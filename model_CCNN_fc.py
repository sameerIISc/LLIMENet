import torch
import torch.nn as nn

cuda = torch.cuda.is_available()
cuda1 = torch.device('cuda:0')

class norm_Block(nn.Module):
    def __init__(self, num_classes=10):
        super(norm_Block, self).__init__()

        n_channels = 64

        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(n_channels,affine=True)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(n_channels,affine=True)
        self.relu2 = nn.PReLU()
        
    def forward(self, x):
        
        conv1 = self.relu1(self.norm1(self.conv1(x)))
        conv2 = self.relu2(self.norm2(self.conv2(conv1)))
        
        out = x + conv2
        
        return out

class res_Block(nn.Module):
    def __init__(self, num_classes=10):
        super(res_Block, self).__init__()

        n_channels = 64

        self.conv3 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.PReLU()
        
    def forward(self, x):
        
        conv3 = self.relu3(self.conv3(x))
        conv4 = self.relu4(self.conv4(conv3))
        
        out = x + conv4
        
        return out

class CNN_c(nn.Module):
    
    def __init__(self, num_classes=10):
        super(CNN_c, self).__init__()
                      
        self.conv01 =  nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        
        self.res1 = norm_Block()
        self.res2 = norm_Block()
        self.res3 = norm_Block()
        self.res4 = norm_Block()
        self.res5 = norm_Block()
        
        # self.conv02 =  nn.Conv2d(67, 64, kernel_size=3, stride=1, padding=1)
                
        self.res6 = res_Block()
        self.res7 = res_Block()
        self.res8 = res_Block()
        self.res9 = res_Block()
        self.res10 = res_Block()
        
        self.apool = nn.AdaptiveAvgPool2d((20,20))
        
        self.linear1 = nn.Linear(25600, 4096, bias=True)
        self.linear2 = nn.Linear(4096, 4096, bias=True)
        self.linear3 = nn.Linear(4096, 111, bias=True)
                
    def forward(self,x):
        
        x2 = self.conv01(x)
        
        res1 = self.res1(x2)
        res2 = self.res2(res1)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)
        
        res6 = self.res6(res5)
        res7 = self.res7(res6)
        res8 = self.res8(res7)
        res9 = self.res9(res8)
        res10 = self.res10(res9)
        
        out = self.apool(res10)

        out = torch.flatten(out, start_dim=1)
      
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        
        return out    