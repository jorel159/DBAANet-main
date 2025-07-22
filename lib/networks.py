import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from lib.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from lib.decoders import DBAA
from lib.decoders import LGAG, EUCB


class DBAANet(nn.Module):
    def __init__(self, num_classes=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, lgag_ks=3, activation='relu', encoder='pvt_v2_b0', pretrain=True):
        super(DBAANet, self).__init__()

        # conv block to convert single channel to 3 channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # backbone network initialization with pretrained weight
        if encoder == 'pvt_v2_b0':
            self.backbone = pvt_v2_b0()
            path = './pretrained_pth/pvt/pvt_v2_b0.pth'
            channels=[256, 160, 64, 32]
        elif encoder == 'pvt_v2_b1':
            self.backbone = pvt_v2_b1()
            path = './pretrained_pth/pvt/pvt_v2_b1.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b2':
            self.backbone = pvt_v2_b2()
            path = './pretrained_pth/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b3':
            self.backbone = pvt_v2_b3()
            path = './pretrained_pth/pvt/pvt_v2_b3.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b4':
            self.backbone = pvt_v2_b4()
            path = './pretrained_pth/pvt/pvt_v2_b4.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'pvt_v2_b5':
            self.backbone = pvt_v2_b5() 
            path = './pretrained_pth/pvt/pvt_v2_b5.pth'
            channels=[512, 320, 128, 64]
        elif encoder == 'resnet18':
            self.backbone = resnet18(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet34':
            self.backbone = resnet34(pretrained=pretrain)
            channels=[512, 256, 128, 64]
        elif encoder == 'resnet50':
            self.backbone = resnet50(pretrained=pretrain)
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet101':
            self.backbone = resnet101(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        elif encoder == 'resnet152':
            self.backbone = resnet152(pretrained=pretrain)  
            channels=[2048, 1024, 512, 256]
        else:
            print('Encoder not implemented! Continuing with default encoder pvt_v2_b2.')
            self.backbone = pvt_v2_b2()  
            path = './pretrained_pth/pvt/pvt_v2_b2.pth'
            channels=[512, 320, 128, 64]
            
        if pretrain==True and 'pvt_v2' in encoder:
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        
        # -----------------------------------新增两次下采样分支  25.2.25
        self.downsample_branch1 = nn.Sequential(
            nn.Conv2d(3, channels[3], kernel_size=7, stride=4, padding=3), 
            nn.BatchNorm2d(channels[3]),
            nn.ReLU()
        )

        self.downsample_branch2 = nn.Sequential(    
            nn.Conv2d(channels[3], channels[0], kernel_size=7, stride=8, padding=3), 
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        
        #LGAG
        self.lgag0 = LGAG(F_g=channels[0], F_l=channels[0], F_int=channels[0]//2, kernel_size=lgag_ks, groups=channels[0]//2)
        #-------------------------------------------------------  


        # print('Model %s created, param count: %d' %
        #              (encoder+' backbone: ', sum([m.numel() for m in self.backbone.parameters()])))
        # 计算编码器各组件参数量
        encoder_params = sum(p.numel() for p in self.backbone.parameters())
        downsample_params = sum(p.numel() for p in self.downsample_branch1.parameters()) + sum(p.numel() for p in self.downsample_branch2.parameters())
        fusion_params = sum(p.numel() for p in self.lgag0.parameters())

        print('[Encoder Components]')
        print('Model %s created, param count: %.2fM' % 
              (encoder+' backbone: ', encoder_params / 1e6))
        print('Model %s created, param count: %.2fM' % 
              ('Downsample branch(LCE): ', downsample_params / 1e6))
        print('Model %s created, param count: %.2fM' % 
              ('Convolutional-Transformer Attention Gate (CTAG): ', fusion_params / 1e6))
        print('Total encoder params: %.2fM' % 
              ((encoder_params + downsample_params + fusion_params) / 1e6))
            
        #   decoder initialization
        self.decoder = DBAA(channels=channels, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, lgag_ks=lgag_ks, activation=activation)
        
        print('Model %s created, param count: %d' %
                     ('DBAAnet decoder: ', sum([m.numel() for m in self.decoder.parameters()])))
             
        self.out_head4 = nn.Conv2d(channels[0], num_classes, 1)
        self.out_head3 = nn.Conv2d(channels[1], num_classes, 1)
        self.out_head2 = nn.Conv2d(channels[2], num_classes, 1)
        self.out_head1 = nn.Conv2d(channels[3], num_classes, 1)
        
        self.orig_proj = nn.Sequential(
            nn.Conv2d(3, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU()
        )

    def forward(self, x, mode='test'):
        
        # if grayscale input, convert to 3 channels
        if x.size()[1] == 1:
            x = self.conv(x)
        


        # encoder1
        
        x1, x2, x3, x4 = self.backbone(x)
        #print(x1.shape, x2.shape, x3.shape, x4.shape)
        
        down_feat1 = self.downsample_branch1(x)  # 输出尺寸: [B, C, 22, 22]
        down_feat2 = self.downsample_branch2(down_feat1)
        
        # 拼接操作
        # x4 = torch.cat([x4, down_feat], dim=1) # 通道维度拼接
        # x4 = self.channel_proj(x4)             # 调整回原通道数
        # 使用EAG_Fusion动态融合PVT主分支和下采样分支
        #x4 = self.eag_fusion(x4, down_feat)  # 输入形状需均为 [B, C, H, W]

        x4 = self.lgag0(g=x4, x=down_feat2)

        #--------------------------------------------------------------

        # decoder
        dec_outs = self.decoder(x4, [x3, x2, x1], [down_feat1,down_feat2])
        
        # prediction heads  
        p4 = self.out_head4(dec_outs[0])
        p3 = self.out_head3(dec_outs[1])
        p2 = self.out_head2(dec_outs[2])
        p1 = self.out_head1(dec_outs[3])

        p4 = F.interpolate(p4, scale_factor=32, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=16, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=8, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=4, mode='bilinear') 
        
        orig_proj = self.orig_proj(x)

        p1 = p1 + orig_proj
        p2 = p2 + orig_proj
        p3 = p3 + orig_proj
        p4 = p4 + orig_proj 
        
        #输出中，p1最大
        if mode == 'test':
            return [p4, p3, p2, p1]
        
        return [p4, p3, p2, p1]
               

        
if __name__ == '__main__':
    model = DBAANet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    P = model(input_tensor)
    print(P[0].size(), P[1].size(), P[2].size(), P[3].size())

