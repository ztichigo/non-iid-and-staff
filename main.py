import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import os
import argparse
import pickle
from torch import distributed, nn


from src.data import *
import  src.model
from src.my_sampler import UnshuffleDistributedSampler
from src.trainer import Trainer
from src.LocalSGD import DSGD
from src.utils import  progress_bar,format_time,get_alg_name,iid
from src.plot import plot_learning_curve



def work_process(process, args):
    #get the rank of current process within world
    args.rank=args.nr*args.gpus*args.process_num+process
    #get the rank of gpu to which current process will be sent 
    gpu=process//args.process_num+args.st

    os.environ['MASTER_ADDR'] ='202.38.73.168' 
    os.environ['MASTER_PORT'] = '6632'
    #set  parameters of distributed environment
    torch.cuda.set_device(gpu) 
    device=torch.device("cuda:"+str(gpu))
    distributed.init_process_group(  
    	backend='nccl',  
    	world_size=args.world_size,  
    	rank=args.rank
    )  
    #move model to specified gpu 
    torch.manual_seed(0)  
    model=src.model.get_model(args).to(device)
    
    #check whether to resume from checkpoint or not
    start_epoch = 0 
    best_acc=0
    if args.resume:
    # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+args.model+"_init.pth")
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    '''
    #set parameters of each worker to be same
    torch.cuda.manual_seed(1)
    for p in model.parameters():
        if p.requires_grad==False:
            continue
        p.data=torch.randn_like(p.data)*0.1
    '''
    #create disributed optimizer
    optimizer=DSGD(model.parameters(),model,update_period=args.period, lr=args.lr,local=args.local,args=args)
    #load dataset
    train_dataset, test_dataset = get_dataset(args.dataset, args)

    train_sampler = UnshuffleDistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, cluster_data=args.cluster_data,Dirichlet=args.Dirichlet)
    train_loader=data.DataLoader(train_dataset,args.batch_size,sampler=train_sampler)
    test_loader=data.DataLoader(test_dataset,args.batch_size,shuffle=True,num_workers=1)
    
    #train model and record related result
    trainer = Trainer(model, optimizer, train_loader, test_loader,device)
    trainer.fit(best_acc, start_epoch, args.epochs, args)

    #save command line arguements
    if args.rank==0:
        path = "Args/{}_{}_{}_{}.pkl".format(args.model, args.dataset,iid(args),get_alg_name(args))
        with open(path, 'wb') as outfile:
            pickle.dump(args,outfile)





def main():
    parse=argparse.ArgumentParser(description='sparse-preserveing federated learning')
    parse.add_argument('--dataset',type=str,default='mnist',help='the dataset to be used')
    parse.add_argument('--input_shape',nargs='+',type=int,help='the shape of input data')
    parse.add_argument('--num_class',type=int,default=10)
    parse.add_argument('--model',type=str,default='LeNet',help='the model to be used')
    parse.add_argument('--root',type=str,default='./data',help='current root of data dir')  
    parse.add_argument('--data_index',type=int,default=1,help='the index of synthetic dataset,range from 0 to 3')
    parse.add_argument('--rank',type=int,default=0,help='the rank of current process') 
    parse.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parse.add_argument('--epochs', type=int, default=3)
    parse.add_argument('--period',type=int,default=1,help='period of updating on each client')
    parse.add_argument('--batch_size',type=int,default=2,help='batch size' )
    parse.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parse.add_argument('--cluster_data', dest='cluster_data', action='store_true')
    parse.add_argument('--Dirichlet',dest='Dirichlet',action='store_true')
    parse.add_argument('--local',dest='local',action='store_true')
    parse.add_argument('--gpus',type=int,default=8,help='the number of gpu per node ')
    parse.add_argument('--nodes',type=int,default=1,help='the number of node')
    parse.add_argument('--process_num',type=int,default=1,help='the number of process per gpu')
    parse.add_argument('--st', type=int, default=0, help='gpu st')
    parse.add_argument('--nr',type=int,default=0,help='rank within node group')
    args = parse.parse_args()
    if not (args.input_shape==None):
        args.input_shape=tuple(args.input_shape)
    args.world_size=args.nodes*args.gpus*args.process_num
    print(args.world_size)
    #os.environ['MASTER_ADDR'] = '202.38.73.153'              #  
    #os.environ['MASTER_PORT'] = '22'  
    mp.spawn(work_process,nprocs=args.gpus*args.process_num,args=(args,))
    #data_load.get_dataset('synthetic',args)

if __name__ == "__main__":
    main()


