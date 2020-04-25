import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GradReversalLayer
import torch.nn.init as init

'''
    Implementation of Light CNN
    @Reference: https://github.com/AlfredXiangWu/LightCNN
'''

def weights_init(m):
    init.xavier_normal_(m.weight)
    if m.bias is not None:
        torch.nn.init.zeros_(m.bias)

class mfm_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm_block, self).__init__()
        self.out_channels = out_channels
        self.type = type
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)
        self.filter.apply(weights_init)
    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)  # split channels in CNN or split D in FC
        return torch.max(out[0], out[1])

class glu_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(glu_block, self).__init__()
        self.out_channels = out_channels
        self.type = type
        self.glu = nn.GLU(dim=1)
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)
        self.filter.apply(weights_init)
    def forward(self, x):
        x = self.filter(x)
        out = self.glu(x)
        return out

class mfm_group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(mfm_group, self).__init__()
        self.conv_a = mfm_block(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm_block(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class glu_group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(glu_group, self).__init__()
        self.conv_a = glu_block(in_channels, in_channels, 1, 1, 0)
        self.conv   = glu_block(in_channels, out_channels, kernel_size, stride, padding)
    def forward(self, x):
        res = x
        x = self.conv_a(x)
        x = self.conv(x)
        return x + res

########## LCNN

class LightCNN_9Layers(nn.Module):
    def __init__(self, inputdim=257, outputdim=2, input_channel=1, **kwargs):
        super(LightCNN_9Layers, self).__init__()
        self.input_channel = input_channel
        self.features = nn.Sequential(
                mfm_block(self.input_channel, 16, 5, 1, 2),                     # 257 * T       * 16
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),          # 129 * (T//2)  * 16
                mfm_group(16, 24, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),          # 65  * (T//4)  * 24
                mfm_group(24, 32, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),          # 33  * (T//8)  * 32
                mfm_group(32, 16, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),          # 17  * (T//16) * 16
                mfm_group(16, 16, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),          # 9   * (T//32) * 16
                )
        dim = 16*((inputdim//self.input_channel+31)//32) # 144
        hDim = 64
        self.fc0 = mfm_block(dim, hDim, type=0)
        # spoof_classifier
        self.fc1 = nn.Linear(hDim, hDim)
        self.fc1.apply(weights_init)
        self.fc2 = nn.Linear(hDim, outputdim)
        self.fc2.apply(weights_init)
        self.spoof_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self.fc1,
                nn.Dropout(p=0.5),
                self.fc2,
                )
    def forward(self, x):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        x = self.features(x) # N*16*(T//32)*9
        x = x.transpose(1,2) # N*(T//32)*16*9
        x = x.contiguous().view(x.size(0),x.size(1),-1) # N*(T//32)*144
        x = x.mean(1) # N*144
        x = self.fc0(x) # N*64
        out = self.spoof_classifier(x) # N*2
        return out
    def extract(self, x):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        x = self.features(x)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        x = x.mean(1)
        x = self.fc0(x)
        return x

class LightCNN_9Layers_DAT(nn.Module):
    def __init__(self, inputdim=257, outputdim1=2, outputdim2=2, input_channel=1, **kwargs):
        super(LightCNN_9Layers_DAT, self).__init__()
        self.input_channel = input_channel
        self.features = nn.Sequential(
                mfm_block(self.input_channel, 16, 5, 1, 2),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                mfm_group(16, 24, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                mfm_group(24, 32, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                mfm_group(32, 16, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                mfm_group(16, 16, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                )
        dim = 16*((inputdim//self.input_channel+31)//32)
        hDim = 64
        self.fc0 = mfm_block(dim, hDim, type=0) #16
        # spoof_classifier
        self.fc1 = nn.Linear(hDim, hDim)
        self.fc1.apply(weights_init)
        self.fc2 = nn.Linear(hDim, outputdim1)
        self.fc2.apply(weights_init)
        self.spoof_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self.fc1,
                nn.Dropout(p=0.5),
                self.fc2,
                )
        # domain_classifier
        self.gradrevLayer = GradReversalLayer(alpha=1.0)
        self.fc3 = nn.Linear(hDim, hDim)
        self.fc3.apply(weights_init)
        self.fc4 = nn.Linear(hDim, outputdim2)
        self.fc4.apply(weights_init)
        self.domain_classifier = nn.Sequential(
                self.gradrevLayer,
                self.fc3,
                self.fc4,
                )
    def forward(self, x, domain='outdomain'):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        x = self.features(x)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        x = x.mean(1)
        x = self.fc0(x)
        if domain == 'outdomain':
            # spoof_classifier
            out1 = self.spoof_classifier(x)
        # domain_classifier
        out2 = self.domain_classifier(x)
        if domain == 'outdomain':
            return out1, out2
        else:
            return out2
    def extract(self, x):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        x = self.features(x)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        x = x.mean(1)
        x = self.fc0(x)
        return x

class LightCNN_9Layers_DADA(nn.Module):
    def __init__(self, inputdim=257, outputdim1=2, outputdim2=2, input_channel=1, **kwargs):
        super(LightCNN_9Layers_DADA, self).__init__()
        self.input_channel = input_channel
        self.features = nn.Sequential(
                mfm_block(self.input_channel, 16, 5, 1, 2),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                mfm_group(16, 24, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                mfm_group(24, 32, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                mfm_group(32, 16, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                mfm_group(16, 16, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                )
        dim = 16*((inputdim//self.input_channel+31)//32)
        hDim = 64
        self.fc0 = mfm_block(dim, hDim, type=0) #16
        # spoof_classifier
        self.fc1 = nn.Linear(hDim, hDim)
        self.fc1.apply(weights_init)
        self.fc2 = nn.Linear(hDim, outputdim1)
        self.fc2.apply(weights_init)
        self.spoof_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self.fc1,
                nn.Dropout(p=0.5),
                self.fc2,
                )
        # bonafide_domain_classifier
        self.gradrevLayer1 = GradReversalLayer(alpha=1.0)
        self.fc3 = nn.Linear(hDim, hDim)
        self.fc3.apply(weights_init)
        self.fc4 = nn.Linear(hDim, outputdim2)
        self.fc4.apply(weights_init)
        self.bonafide_domain_classifier = nn.Sequential(
                self.gradrevLayer1,
                self.fc3,
                self.fc4,
                )
        # spoof_domain_classifier
        self.gradrevLayer2 = GradReversalLayer(alpha=1.0)
        self.fc5 = nn.Linear(hDim, hDim)
        self.fc5.apply(weights_init)
        self.fc6 = nn.Linear(hDim, outputdim2)
        self.fc6.apply(weights_init)
        self.spoof_domain_classifier = nn.Sequential(
                self.gradrevLayer2,
                self.fc5,
                self.fc6,
                )

    def forward(self, x, domain='bonafide_outdomain'):
        '''
            domain: bonafide_outdomain, spoof_outdomain, indomain
        '''
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        x = self.features(x)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        x = x.mean(1)
        x = self.fc0(x)

        # spoof_classifier
        with torch.set_grad_enabled(domain != 'indomain'): # do not update spoof_classifier for indomain data
            out1 = self.spoof_classifier(x)
        # bonafide_domain_classifier
        if domain == 'bonafide_outdomain':
            out2 = self.bonafide_domain_classifier(x)
        elif domain == 'spoof_outdomain':
            out2 = self.spoof_domain_classifier(x)
        else: # indomain
            out1 = F.softmax(out1, dim=-1)
            bonafide_features = x #out1[:,0].unsqueeze(1) * x
            spoof_features = x #out1[:,1].unsqueeze(1) * x
            out2 = self.bonafide_domain_classifier(bonafide_features)
            out3 = self.spoof_domain_classifier(spoof_features)
        if domain == 'indomain':
            return out1, out2, out3
        else:
            return out1, out2
    def extract(self, x):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        x = self.features(x)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        x = x.mean(1)
        x = self.fc0(x)
        return x



########## CGCNN

class CGCNN_10Layers(nn.Module):
    def __init__(self, inputdim=257, outputdim=2, input_channel=1, **kwargs):
        super(CGCNN_10Layers, self).__init__()
        self.input_channel = input_channel
        self.features = nn.Sequential(
                glu_block(self.input_channel, 16, 5, 1, 2),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                glu_group(16, 16, 3, 1, 1),
                glu_block(16, 32, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                glu_group(32, 32, 3, 1, 1),
                glu_block(32, 16, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                glu_group(16, 16, 3, 1, 1),
                glu_block(16, 16, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                )
        dim = 16*((inputdim//self.input_channel+15)//16)
        hDim = 64
        self.fc0 = nn.Linear(dim, hDim)
        # spoof_classifier
        self.fc1 = nn.Linear(hDim, hDim)
        self.fc1.apply(weights_init)
        self.fc2 = nn.Linear(hDim, outputdim)
        self.fc2.apply(weights_init)
        self.spoof_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self.fc1,
                nn.Dropout(p=0.5),
                self.fc2,
                )
    def forward(self, x):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        x = self.features(x)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        x = x.mean(1)
        x = self.fc0(x)
        out = self.spoof_classifier(x)
        return out
    def extract(self, x):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        x = self.features(x)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        x = x.mean(1)
        x = self.fc0(x)
        return x

class CGCNN_10Layers_DAT(nn.Module):
    def __init__(self, inputdim=257, outputdim1=2, outputdim2=2, input_channel=1, **kwargs):
        super(CGCNN_10Layers_DAT, self).__init__()
        self.input_channel = input_channel
        self.features = nn.Sequential(
                glu_block(self.input_channel, 16, 5, 1, 2),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                glu_group(16, 16, 3, 1, 1),
                glu_block(16, 32, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                glu_group(32, 32, 3, 1, 1),
                glu_block(32, 16, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                glu_group(16, 16, 3, 1, 1),
                glu_block(16, 16, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                )
        dim = 16*((inputdim//self.input_channel+15)//16)
        hDim = 64
        self.fc0 = nn.Linear(dim, hDim)
        # spoof_classifier
        self.fc1 = nn.Linear(hDim, hDim)
        self.fc1.apply(weights_init)
        self.fc2 = nn.Linear(hDim, outputdim1)
        self.fc2.apply(weights_init)
        self.spoof_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self.fc1,
                nn.Dropout(p=0.5),
                self.fc2,
                )
        # domain_classifier
        self.gradrevLayer = GradReversalLayer(alpha=1.0)
        self.fc3 = nn.Linear(hDim, hDim)
        self.fc3.apply(weights_init)
        self.fc4 = nn.Linear(hDim, outputdim2)
        self.fc4.apply(weights_init)
        self.domain_classifier = nn.Sequential(
                self.gradrevLayer,
                self.fc3,
                self.fc4,
                )
    def forward(self, x, domain='outdomain'):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        x = self.features(x)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        x = x.mean(1)
        x = self.fc0(x)
        if domain == 'outdomain':
            # spoof_classifier
            out1 = self.spoof_classifier(x)
        # domain_classifier
        out2 = self.domain_classifier(x)
        if domain == 'outdomain':
            return out1, out2
        else:
            return out2
    def extract(self, x):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        x = self.features(x)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        x = x.mean(1)
        x = self.fc0(x)
        return x


class CGCNN_10Layers_DADA(nn.Module):
    def __init__(self, inputdim=257, outputdim1=2, outputdim2=2, input_channel=1, **kwargs):
        super(CGCNN_10Layers_DADA, self).__init__()
        self.input_channel = input_channel
        self.features = nn.Sequential(
                glu_block(self.input_channel, 16, 5, 1, 2),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                glu_group(16, 16, 3, 1, 1),
                glu_block(16, 32, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                glu_group(32, 32, 3, 1, 1),
                glu_block(32, 16, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                glu_group(16, 16, 3, 1, 1),
                glu_block(16, 16, 3, 1, 1),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                )
        dim = 16*((inputdim//self.input_channel+15)//16)
        hDim = 64
        self.fc0 = nn.Linear(dim, hDim)
        # spoof_classifier
        self.fc1 = nn.Linear(hDim, hDim)
        self.fc1.apply(weights_init)
        self.fc2 = nn.Linear(hDim, outputdim1)
        self.fc2.apply(weights_init)
        self.spoof_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self.fc1,
                nn.Dropout(p=0.5),
                self.fc2,
                )
        # bonafide_domain_classifier
        self.gradrevLayer1 = GradReversalLayer(alpha=1.0)
        self.fc3 = nn.Linear(hDim, hDim)
        self.fc3.apply(weights_init)
        self.fc4 = nn.Linear(hDim, outputdim2)
        self.fc4.apply(weights_init)
        self.bonafide_domain_classifier = nn.Sequential(
                self.gradrevLayer1,
                self.fc3,
                self.fc4,
                )
        # spoof_domain_classifier
        self.gradrevLayer2 = GradReversalLayer(alpha=1.0)
        self.fc5 = nn.Linear(hDim, hDim)
        self.fc5.apply(weights_init)
        self.fc6 = nn.Linear(hDim, outputdim2)
        self.fc6.apply(weights_init)
        self.spoof_domain_classifier = nn.Sequential(
                self.gradrevLayer2,
                self.fc5,
                self.fc6,
                )

    def forward(self, x, domain='bonafide_outdomain'):
        '''
            domain: bonafide_outdomain, spoof_outdomain, indomain
        '''
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        x = self.features(x)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        x = x.mean(1)
        x = self.fc0(x)

        # spoof_classifier
        with torch.set_grad_enabled(domain != 'indomain'): # do not update spoof_classifier for indomain data
            out1 = self.spoof_classifier(x)
        # bonafide_domain_classifier
        if domain == 'bonafide_outdomain':
            out2 = self.bonafide_domain_classifier(x)
        elif domain == 'spoof_outdomain':
            out2 = self.spoof_domain_classifier(x)
        else: # indomain
            out1 = F.softmax(out1, dim=-1)
            bonafide_features = x #out1[:,0].unsqueeze(1) * x
            spoof_features = x #out1[:,1].unsqueeze(1) * x
            out2 = self.bonafide_domain_classifier(bonafide_features)
            out3 = self.spoof_domain_classifier(spoof_features)
        if domain == 'indomain':
            return out1, out2, out3
        else:
            return out1, out2
    def extract(self, x):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        x = self.features(x)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.size(0),x.size(1),-1)
        x = x.mean(1)
        x = self.fc0(x)
        return x



########## Resnet

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1.apply(weights_init)
        self.bn1 = nn.Sequential() #nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2.apply(weights_init)
        self.bn2 = nn.Sequential() #nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.conv3 = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.conv3.apply(weights_init)
            self.shortcut = nn.Sequential(
                self.conv3,
                #nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_10Layers(nn.Module):
    def __init__(self, inputdim=257, outputdim=2, input_channel=1, **kwargs):
        super(ResNet_10Layers, self).__init__()
        num_blocks = [1, 1, 1, 1]
        m_channels = 16
        embedding_dim = 64
        self.in_planes = m_channels
        self.embedding_dim = embedding_dim
        self.input_channel = input_channel

        self.conv1 = nn.Conv2d(self.input_channel, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.Sequential() #nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(m_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(m_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(m_channels*8, num_blocks[3], stride=2)
        self.embedding = nn.Linear(m_channels * 8 * BasicBlock.expansion, embedding_dim)

        # spoof_classifier
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, outputdim)
        self.spoof_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self.fc1,
                nn.Dropout(p=0.5),
                self.fc2,
                )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        length = x.size(2)
        feat_dim = x.size(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (int(math.ceil(length/8)), int(math.ceil(feat_dim/8))))
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        out = self.spoof_classifier(x)
        return out

    def extract(self, x):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        length = x.size(2)
        feat_dim = x.size(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (int(math.ceil(length/8)), int(math.ceil(feat_dim/8))))
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x

class ResNet_10Layers_DAT(nn.Module):
    def __init__(self, inputdim=257, outputdim1=2, outputdim2=2, input_channel=1, **kwargs):
        super(ResNet_10Layers_DAT, self).__init__()
        num_blocks = [1, 1, 1, 1]
        m_channels = 16
        embedding_dim = 64
        self.in_planes = m_channels
        self.embedding_dim = embedding_dim
        self.input_channel = input_channel

        self.conv1 = nn.Conv2d(self.input_channel, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.Sequential() #nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(m_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(m_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(m_channels*8, num_blocks[3], stride=2)
        self.embedding = nn.Linear(m_channels * 8 * BasicBlock.expansion, embedding_dim)

        # spoof_classifier
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, outputdim1)
        self.spoof_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self.fc1,
                nn.Dropout(p=0.5),
                self.fc2,
                )
        # domain_classifier
        self.gradrevLayer = GradReversalLayer(alpha=1.0)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        self.fc4 = nn.Linear(embedding_dim, outputdim2)
        self.domain_classifier = nn.Sequential(
                self.gradrevLayer,
                self.fc3,
                self.fc4,
                )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x, domain='outdomain'):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        length = x.size(2)
        feat_dim = x.size(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (int(math.ceil(length/8)), int(math.ceil(feat_dim/8))))
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        
        if domain == 'outdomain':
            # spoof_classifier
            out1 = self.spoof_classifier(x)
        # domain_classifier
        out2 = self.domain_classifier(x)
        if domain == 'outdomain':
            return out1, out2
        else:
            return out2

    def extract(self, x):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        length = x.size(2)
        feat_dim = x.size(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (int(math.ceil(length/8)), int(math.ceil(feat_dim/8))))
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x

class ResNet_10Layers_DADA(nn.Module):
    def __init__(self, inputdim=257, outputdim1=2, outputdim2=2, input_channel=1, **kwargs):
        super(ResNet_10Layers_DADA, self).__init__()
        num_blocks = [1, 1, 1, 1]
        m_channels = 16
        embedding_dim = 64
        self.in_planes = m_channels
        self.embedding_dim = embedding_dim
        self.input_channel = input_channel

        self.conv1 = nn.Conv2d(self.input_channel, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.apply(weights_init)
        self.bn1 = nn.Sequential() #nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(m_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(m_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(m_channels*8, num_blocks[3], stride=2)
        self.embedding = nn.Linear(m_channels * 8 * BasicBlock.expansion, embedding_dim)
        self.embedding.apply(weights_init)

        # spoof_classifier
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc1.apply(weights_init)
        self.fc2 = nn.Linear(embedding_dim, outputdim1)
        self.fc2.apply(weights_init)
        self.spoof_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self.fc1,
                nn.Dropout(p=0.5),
                self.fc2,
                )
        # bonafide_domain_classifier
        self.gradrevLayer1 = GradReversalLayer(alpha=1.0)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3.apply(weights_init)
        self.fc4 = nn.Linear(embedding_dim, outputdim2)
        self.fc4.apply(weights_init)
        self.bonafide_domain_classifier = nn.Sequential(
                self.gradrevLayer1,
                self.fc3,
                self.fc4,
                )
        # spoof_domain_classifier
        self.gradrevLayer2 = GradReversalLayer(alpha=1.0)
        self.fc5 = nn.Linear(embedding_dim, embedding_dim)
        self.fc5.apply(weights_init)
        self.fc6 = nn.Linear(embedding_dim, outputdim2)
        self.fc6.apply(weights_init)
        self.spoof_domain_classifier = nn.Sequential(
                self.gradrevLayer2,
                self.fc5,
                self.fc6,
                )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x, domain='bonafide_outdomain'):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        length = x.size(2)
        feat_dim = x.size(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (int(math.ceil(length/8)), int(math.ceil(feat_dim/8))))
        x = x.view(x.size(0), -1)
        x = self.embedding(x)

        # spoof_classifier
        with torch.set_grad_enabled(domain != 'indomain'): # do not update spoof_classifier for indomain data
            out1 = self.spoof_classifier(x)
        # bonafide_domain_classifier
        if domain == 'bonafide_outdomain':
            out2 = self.bonafide_domain_classifier(x)
        elif domain == 'spoof_outdomain':
            out2 = self.spoof_domain_classifier(x)
        else: # indomain
            out1 = F.softmax(out1, dim=-1)
            out2 = self.bonafide_domain_classifier(x)
            out3 = self.spoof_domain_classifier(x)
        if domain == 'indomain':
            return out1, out2, out3
        else:
            return out1, out2

    def extract(self, x):
        x = x.unsqueeze(1) if self.input_channel == 1 else torch.stack(x.chunk(self.input_channel, dim=-1)).transpose(0,1).contiguous()
        length = x.size(2)
        feat_dim = x.size(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (int(math.ceil(length/8)), int(math.ceil(feat_dim/8))))
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x
        

class ResNet_18Layers(nn.Module):
    def __init__(self, inputdim=257, outputdim=2, input_channel=1, **kwargs):
        super(ResNet_18Layers, self).__init__()
        num_blocks = [2, 2, 2, 2]
        m_channels = 16
        embedding_dim = 64
        self.in_planes = m_channels
        self.embedding_dim = embedding_dim
        self.input_channel = input_channel

        self.conv1 = nn.Conv2d(self.input_channel, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.Sequential() #nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(m_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(m_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(m_channels*8, num_blocks[3], stride=2)
        self.embedding = nn.Linear(m_channels * 8 * BasicBlock.expansion, embedding_dim)
        #self.linear = nn.Linear(embedding_dim, outputdim)

        # spoof_classifier
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, outputdim)
        self.spoof_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self.fc1,
                nn.Dropout(p=0.5),
                self.fc2,
                )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        length = x.size(2)
        feat_dim = x.size(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (int(math.ceil(length/8)), int(math.ceil(feat_dim/8))))
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        out = self.spoof_classifier(x)
        return out

    def extract(self, x):
        x = x.unsqueeze(1)
        length = x.size(2)
        feat_dim = x.size(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (int(math.ceil(length/8)), int(math.ceil(feat_dim/8))))
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x

class ResNet_18Layers_DAT(nn.Module):
    def __init__(self, inputdim=257, outputdim1=2, outputdim2=2, input_channel=1, **kwargs):
        super(ResNet_18Layers_DAT, self).__init__()
        num_blocks = [2, 2, 2, 2]
        m_channels = 16
        embedding_dim = 64
        self.in_planes = m_channels
        self.embedding_dim = embedding_dim
        self.input_channel = input_channel

        self.conv1 = nn.Conv2d(self.input_channel, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.Sequential() #nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(m_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(m_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(m_channels*8, num_blocks[3], stride=2)
        self.embedding = nn.Linear(m_channels * 8 * BasicBlock.expansion, embedding_dim)
        #self.linear = nn.Linear(embedding_dim, outputdim)

        # spoof_classifier
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, outputdim1)
        self.spoof_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self.fc1,
                nn.Dropout(p=0.5),
                self.fc2,
                )
        # domain_classifier
        self.gradrevLayer = GradReversalLayer(alpha=1.0)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        self.fc4 = nn.Linear(embedding_dim, outputdim2)
        self.domain_classifier = nn.Sequential(
                self.gradrevLayer,
                self.fc3,
                self.fc4,
                )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x, domain='outdomain'):
        x = x.unsqueeze(1)
        length = x.size(2)
        feat_dim = x.size(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (int(math.ceil(length/8)), int(math.ceil(feat_dim/8))))
        x = x.view(x.size(0), -1)
        x = self.embedding(x)

        if domain == 'outdomain':
            # spoof_classifier
            out1 = self.spoof_classifier(x)
        # domain_classifier
        out2 = self.domain_classifier(x)
        if domain == 'outdomain':
            return out1, out2 # N*(T//32)*(outputdim)
        else:
            return out2

    def extract(self, x):
        x = x.unsqueeze(1)
        length = x.size(2)
        feat_dim = x.size(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (int(math.ceil(length/8)), int(math.ceil(feat_dim/8))))
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x

class ResNet_18Layers_MADA(nn.Module):
    def __init__(self, inputdim=257, outputdim1=2, outputdim2=2, input_channel=1, **kwargs):
        super(ResNet_18Layers_MADA, self).__init__()
        num_blocks = [2, 2, 2, 2]
        m_channels = 16
        embedding_dim = 64
        self.in_planes = m_channels
        self.embedding_dim = embedding_dim
        self.input_channel = input_channel

        self.conv1 = nn.Conv2d(self.input_channel, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1.apply(weights_init)
        self.bn1 = nn.Sequential() #nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(m_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(m_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(m_channels*8, num_blocks[3], stride=2)
        self.embedding = nn.Linear(m_channels * 8 * BasicBlock.expansion, embedding_dim)
        self.embedding.apply(weights_init)
        #self.linear = nn.Linear(embedding_dim, outputdim)

        # spoof_classifier
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc1.apply(weights_init)
        self.fc2 = nn.Linear(embedding_dim, outputdim1)
        self.fc2.apply(weights_init)
        self.spoof_classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                self.fc1,
                nn.Dropout(p=0.5),
                self.fc2,
                )
        # bonafide_domain_classifier
        self.gradrevLayer1 = GradReversalLayer(alpha=1.0)
        self.fc3 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3.apply(weights_init)
        self.fc4 = nn.Linear(embedding_dim, outputdim2)
        self.fc4.apply(weights_init)
        self.bonafide_domain_classifier = nn.Sequential(
                self.gradrevLayer1,
                self.fc3,
                self.fc4,
                )
        # spoof_domain_classifier
        self.gradrevLayer2 = GradReversalLayer(alpha=1.0)
        self.fc5 = nn.Linear(embedding_dim, embedding_dim)
        self.fc5.apply(weights_init)
        self.fc6 = nn.Linear(embedding_dim, outputdim2)
        self.fc6.apply(weights_init)
        self.spoof_domain_classifier = nn.Sequential(
                self.gradrevLayer2,
                self.fc5,
                self.fc6,
                )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x, domain='bonafide_outdomain'):
        x = x.unsqueeze(1)
        length = x.size(2)
        feat_dim = x.size(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (int(math.ceil(length/8)), int(math.ceil(feat_dim/8))))
        x = x.view(x.size(0), -1)
        x = self.embedding(x)

        # spoof_classifier
        with torch.set_grad_enabled(domain != 'indomain'): # do not update spoof_classifier for indomain data
            out1 = self.spoof_classifier(x)
        # bonafide_domain_classifier
        if domain == 'bonafide_outdomain':
            out2 = self.bonafide_domain_classifier(x)
        elif domain == 'spoof_outdomain':
            out2 = self.spoof_domain_classifier(x)
        else: # indomain
            out1 = F.softmax(out1, dim=-1)
            out2 = self.bonafide_domain_classifier(x)
            out3 = self.spoof_domain_classifier(x)
        if domain == 'indomain':
            return out1, out2, out3
        else:
            return out1, out2

    def extract(self, x):
        x = x.unsqueeze(1)
        length = x.size(2)
        feat_dim = x.size(3)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, (int(math.ceil(length/8)), int(math.ceil(feat_dim/8))))
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return x

