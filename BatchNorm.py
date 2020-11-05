from __future__ import division

import torch
from torch.nn import Module
import torch.distributed as dist
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch._jit_internal import weak_module, weak_script_method


# TODO: check contiguous in THNN
# TODO: use separate backend functions?
@weak_module
class _BatchNorm(Module):
    _version = 2
    __constants__ = ['track_running_stats', 'momentum', 'eps', 'weight', 'bias',
                     'running_mean', 'running_var', 'num_batches_tracked']

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True,period=None):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.num_iterations=0
        self.period=period
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.uniform_(self.weight)
            init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    @weak_script_method
    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return self.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)


    def batch_norm(self,input, running_mean, running_var, weight=None, bias=None,
                training=False, momentum=0.1, eps=1e-5):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor], bool, float, float) -> Tensor  # noqa
        """Applies Batch Normalization for each channel across a batch of data.

        See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`,
        :class:`~torch.nn.BatchNorm3d` for details.
        """
        #torch.backends.cudnn.enabled=True
        if training:
            size = input.size()
            # XXX: JIT script does not support the reduce from functools, and mul op is a
            # builtin, which cannot be used as a value to a func yet, so rewrite this size
            # check to a simple equivalent for loop
            #
            # TODO: make use of reduce like below when JIT is ready with the missing features:
            # from operator import mul
            # from functools import reduce
            #
            #   if reduce(mul, size[2:], size[0]) == 1
            size_prods = size[0]
            for i in range(len(size) - 2):
                size_prods *= size[i + 2]
            if size_prods == 1:
                raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
        #---------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
            X=input.clone()
            if len(size)==2:
                X=X.permute(1,0)   
            elif len(size)==3:
                X=X.permute(1,0,2).reshape(self.num_features,-1)
            elif len(size)==4:
                X=X.permute(1,0,2,3).reshape(self.num_features,-1)
            else:
                raise ValueError('there is no suitable function for input size {}'.format(size))
            mean=torch.mean(X,dim=1,keepdim=False)
            var=torch.var(X,dim=1,unbiased=False,keepdim=False)
            if self.training and self.track_running_stats:
                #update running mean and var
                if self.track_running_stats:
                    self.running_mean=(1-momentum)*running_mean+momentum*mean
                    self.running_var=(1-momentum)*running_var+momentum*var
                    #synchronize mean and var across all workers
                    self.num_iterations+=1
                    if self.num_iterations%(80*self.period)==0:
                        self.all_reduce_statistic()
                    
            #normalize input with mean and std of the current mini-batch     ***
            if self.affine:
                if len(size)==2:
                    y=(input-mean.reshape(1,self.num_features))*weight.reshape(1,self.num_features)/(var.reshape(1,self.num_features)+eps)**0.5+bias.reshape(1,self.num_features)
                if len(size)==3:
                    y=(input-mean.reshape(1,self.num_features,1))*weight.reshape(1,self.num_features,1)/(var.reshape(1,self.num_features,1)+eps)**0.5+bias.reshape(1,self.num_features,1)
                if len(size)==4:
                    y=(input-mean.reshape(1,self.num_features,1,1))*weight.reshape(1,self.num_features,1,1)/(var.reshape(1,self.num_features,1,1)+eps)**0.5+bias.reshape(1,self.num_features,1,1)
            else:
                if len(size)==2:
                    y=(input-mean.reshape(1,self.num_features))/(var.reshape(1,self.num_features)+eps)**0.5
                if len(size)==3:
                    y=(input-mean.reshape(1,self.num_features,1))/(var.reshape(1,self.num_features,1)+eps)**0.5
                if len(size)==4:
                    y=(input-mean.reshape(1,self.num_features,1,1))/(var.reshape(1,self.num_features,1,1)+eps)**0.5  
            return y
        else:
            #normalize input with mean and std of running mean and running std while testing ***
            size = input.size()
            if self.affine:
                if len(size)==2:
                    y=(input-running_mean.reshape(1,self.num_features))*weight.reshape(1,self.num_features)/(running_var.reshape(1,self.num_features)+eps)**0.5+bias.reshape(1,self.num_features)
                if len(size)==3:
                    y=(input-running_mean.reshape(1,self.num_features,1))*weight.reshape(1,self.num_features,1)/(running_var.reshape(1,self.num_features,1)+eps)**0.5+bias.reshape(1,self.num_features,1)
                if len(size)==4:
                    y=(input-running_mean.reshape(1,self.num_features,1,1))*weight.reshape(1,self.num_features,1,1)/(running_var.reshape(1,self.num_features,1,1)+eps)**0.5+bias.reshape(1,self.num_features,1,1)
            else:
                if len(size)==2:
                    y=(input-running_mean.reshape(1,self.num_features))/(running_var.reshape(1,self.num_features)+eps)**0.5
                if len(size)==3:
                    y=(input-running_mean.reshape(1,self.num_features,1))/(running_var.reshape(1,self.num_features,1)+eps)**0.5
                if len(size)==4:
                    y=(input-running_mean.reshape(1,self.num_features,1,1))/(running_var.reshape(1,self.num_features,1,1)+eps)**0.5
            return y

        #---------------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------------
        '''
        return torch.batch_norm(
            input, weight, bias, running_mean, running_var,
            training, momentum, eps, torch.backends.cudnn.enabled
        )
        '''
    def all_reduce_statistic(self):
        mean_k=self.running_mean.clone().detach()
        dist.all_reduce(mean_k)
        var_k=self.running_var.clone().detach()
        var_dmean=var_k+(self.running_mean-mean_k)**2
        dist.all_reduce(var_dmean)
        self.running_mean=mean_k
        self.running_var=var_dmean

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


@weak_module
class BatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2D or 3D input (a mini-batch of 1D
    inputs with optional additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are sampled
    from :math:`\mathcal{U}(0, 1)` and the elements of :math:`\beta` are set to 0.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)`
        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm1d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm1d(100, affine=False)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    @weak_script_method
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


@weak_module
class BatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are sampled
    from :math:`\mathcal{U}(0, 1)` and the elements of :math:`\beta` are set to 0.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    @weak_script_method
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
