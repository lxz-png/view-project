import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNN_Denoise(nn.Module):
    def __init__(self, channel=3, layer_num=17):
        super().__init__()
        layers = []
        
        layers.append(nn.Conv2d(channel, 64, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(layer_num-2):
            layers.append(nn.Conv2d(64,64,3,padding=1,bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(64, channel, 3, padding=1, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual_noise = self.model(x)
        return x - residual_noise  


class SRCNN_SuperRes(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear")
        # SRCNN三层卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.upsample(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        x = F.adaptive_avg_pool2d(x, 256)
        return x


class SceneRestorePipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise_net = DnCNN_Denoise()   
        self.sr_net = SRCNN_SuperRes()       

    def forward(self, old_noisy_img):
        clean_img = self.denoise_net(old_noisy_img)  
        restore_highres_img = self.sr_net(clean_img) 
        return restore_highres_img
