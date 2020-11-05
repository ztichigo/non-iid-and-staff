import torch.nn as nn
import torch.nn.functional as F
import importlib
import math
import torch
import  torchvision.models as  models
from src.BatchNorm import _BatchNorm,BatchNorm1d,BatchNorm2d

class TwoHiddenLayerFc(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(TwoHiddenLayerFc, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, out_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out=F.softmax(out,dim=1)
        return out


class LeNet(nn.Module):
    def __init__(self,input_shape,out_dim):
        super(LeNet,self).__init__()
        self.conv1=nn.Conv2d(input_shape[0],6,5)
        self.conv2=nn.Conv2d(6,15,5)
        self.fc1=nn.Linear(15*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,out_dim)

    def forward(self,input):
        out=F.relu(self.conv1(input))
        out=F.max_pool2d(out,2)
        out=F.relu(self.conv2(out))
        out=F.max_pool2d(out,2)
        out=out.view(out.size(0),-1)
        out=F.relu(self.fc1(out))
        out=F.relu(self.fc2(out))
        out=self.fc3(out)
        return out

class LeNet_BN(nn.Module):
    def __init__(self,input_shape,out_dim):
        super(LeNet_BN,self).__init__()
        self.conv1=nn.Conv2d(input_shape[0],6,5)
        self.bn1=nn.BatchNorm2d(6)
        self.conv2=nn.Conv2d(6,15,5)
        self.bn2=nn.BatchNorm2d(15)
        self.bn31=nn.BatchNorm1d(120)
        self.fc1=nn.Linear(15*5*5,120)
        self.bn32=nn.BatchNorm1d(84)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,out_dim)
        torch.manual_seed(0)

    def forward(self,input):
        out=F.relu(self.bn1(self.conv1(input)))
        out=F.max_pool2d(out,2)
        out=F.relu(self.bn2(self.conv2(out)))
        out=F.max_pool2d(out,2)
        out=out.view(out.size(0),-1)
        out=F.relu(self.bn31(self.fc1(out)))
        out=F.relu(self.bn32(self.fc2(out)))
        out=self.fc3(out)
        return out

class LeNet_FBN(nn.Module):
    def __init__(self,input_shape,out_dim,period):
        super(LeNet_FBN,self).__init__()
        self.conv1=nn.Conv2d(input_shape[0],6,5)
        self.bn1=BatchNorm2d(6,period=period)
        self.conv2=nn.Conv2d(6,15,5)
        self.bn2=BatchNorm2d(15,period=period)
        self.bn31=BatchNorm1d(120,period=period)
        self.fc1=nn.Linear(15*5*5,120)
        self.bn32=BatchNorm1d(84,period=period)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,out_dim)

    def forward(self,input):
        out=F.relu(self.bn1(self.conv1(input)))
        out=F.max_pool2d(out,2)
        out=F.relu(self.bn2(self.conv2(out)))
        out=F.max_pool2d(out,2)
        out=out.view(out.size(0),-1)
        out=F.relu(self.bn31(self.fc1(out)))
        out=F.relu(self.bn32(self.fc2(out)))
        out=self.fc3(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18(num_class):
    return ResNet(ResidualBlock,num_class)

class ResidualBlock_BN(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_BN(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18_BN(num_class):
    return ResNet(ResidualBlock,num_class)




def get_model(options):
    model_name = str(options.model)
    if  model_name=='LeNet':
        return LeNet(options.input_shape,options.num_class)
    elif model_name=='LeNet_BN':
        return LeNet_BN(options.input_shape,options.num_class)
    elif model_name=='LeNet_FBN':
        return LeNet_FBN(options.input_shape,options.num_class,options.period)    
    elif model_name=='ResNet18':
        return ResNet18(options.num_class)
    elif model_name=='ResNet18_BN':
        return ResNet18_BN(options.num_class)
    elif model_name=='TwoHiddenLayerFc':
        return TwoHiddenLayerFc(options.input_shape,options.num_class)
    else:
        raise ValueError("Not support model: {}!".format(model_name))