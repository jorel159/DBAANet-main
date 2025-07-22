import torch
import torch.nn as nn
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from thop import profile



def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

#   Multi-scale depth-wise convolution (MSDC)
class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(self.activation, inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x+dw_out
        # You can return outputs based on what you intend to do with them
        return outputs

class MSCB(nn.Module):
    """
    Multi-scale convolution block (MSCB) 
    """
    def __init__(self, in_channels, out_channels, stride, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        super(MSCB, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_sizes = kernel_sizes
        self.expansion_factor = expansion_factor
        self.dw_parallel = dw_parallel
        self.add = add
        self.activation = activation
        self.n_scales = len(self.kernel_sizes)
        # check stride value
        assert self.stride in [1, 2]
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor
        self.ex_channels = int(self.in_channels * self.expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_channels, self.ex_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_channels),
            act_layer(self.activation, inplace=True)
        )
        self.msdc = MSDC(self.ex_channels, self.kernel_sizes, self.stride, self.activation, dw_parallel=self.dw_parallel)
        if self.add == True:
            self.combined_channels = self.ex_channels*1
        else:
            self.combined_channels = self.ex_channels*self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_channels, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.out_channels),
        )
        if self.use_skip_connection and (self.in_channels != self.out_channels):
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False)
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        msdc_outs = self.msdc(pout1)
        if self.add == True:
            dout = 0
            for dwout in msdc_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(msdc_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels,self.out_channels))
        out = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_channels != self.out_channels:
                x = self.conv1x1(x)
            return x + out
        else:
            return out
        
#   Multi-scale convolution block (MSCB)
def MSCBLayer(in_channels, out_channels, n=1, stride=1, kernel_sizes=[1,3,5], expansion_factor=2, dw_parallel=True, add=True, activation='relu6'):
        """
        create a series of multi-scale convolution blocks.
        """
        convs = []
        mscb = MSCB(in_channels, out_channels, stride, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        convs.append(mscb)
        if n > 1:
            for i in range(1, n):
                mscb = MSCB(out_channels, out_channels, 1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
                convs.append(mscb)
        conv = nn.Sequential(*convs)
        return conv

#   Efficient up-convolution block (EUCB)
class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.in_channels, bias=False),
	        nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        ) 
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x
    
class EDCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu'):
        super().__init__()
        self.down_dwc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                      stride=2, padding=kernel_size//2, 
                      groups=in_channels, bias=False),  # 深度卷积
            nn.BatchNorm2d(in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)  # 1x1 卷积调整通道
        )
    
    def forward(self, x):
        x = self.down_dwc(x)
        x = self.pwc(x)
        return x    


#   Large-kernel grouped attention gate (LGAG)
class LGAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG,self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
                
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x*psi
    
class LGAG3(nn.Module):
    def __init__(self, F_g ,F_k, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG3,self).__init__()

        if kernel_size == 1:
            groups = 1
        # 新增第三个输入的处理分支
        self.W_k = nn.Sequential(
            nn.Conv2d(F_k, F_int, kernel_size, 1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, 1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, 1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # 修改为3通道输出
        self.up3c = nn.Conv2d(F_int, 3, kernel_size=1,stride=1,padding=0,bias=True)  # 2 -> 3
        
        # 扩展为三个处理分支
        self.psi_branch = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            ) for _ in range(3)  # 2 -> 3
        ])

        # 改为三元融合参数
        self.alpha = nn.Parameter(torch.tensor([ 0.3, 0.1 ,0.6]))  # 初始化权重
        
        self.activation = act_layer(activation, inplace=True)
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
                
    # 第一个为解码器特征，第二个为辅助编码器链接 ，第三个为主编码器链接         
    def forward(self, g, k ,x):
        g1 = self.W_g(g)
        k1 = self.W_k(k)  # 新增分支处理
        x1 = self.W_x(x)
        
        # 三路特征融合
        fused = self.activation(g1 + k1 + x1)  # 新增k1参与融合
        up3c = self.up3c(fused)  # [B,3,H,W]
        
        # 分割三通道
        c0 = up3c[:, [0], :, :]  # 通道0
        c1 = up3c[:, [1], :, :]  # 通道1
        c2 = up3c[:, [2], :, :]  # 新增通道2
        
        # 三个注意力权重
        attn_0 = self.psi_branch[0](c0)
        attn_1 = self.psi_branch[1](c1)
        attn_2 = self.psi_branch[2](c2)  # 新增第三个注意力
        
        # 加权融合（保持梯度计算）
        return ( g * attn_0 + 
                k * attn_1 + 
                x * attn_2)
    

#   Channel attention block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out= self.max_pool(x) 
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out) 
    
#   Spatial attention block (SAB)
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size//2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

# 计算参数量和 FLOPs 的函数
def calc_params_flops(model, input_size):
    # 参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 使用 thop 计算 FLOPs（需要安装：pip install thop）
    try:
        input = torch.randn(1, input_size[0], input_size[1], input_size[2])
        flops, params = profile(model, inputs=(input,), verbose=False)
        flops = flops / 1e9  # 转换为 GFLOPs
    except:
        flops = None
        print("FLOPs calculation requires 'thop' library. Install it or compute manually.")
    
    return total_params, flops

#Dual-Branch Attention Aggregation Network (DBAANet)
class DBAA(nn.Module):
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], expansion_factor=6, dw_parallel=True, add=True, lgag_ks=3, activation='relu6'):
        super(DBAA,self).__init__()
        eucb_ks = 3 # kernel size for eucb
        #第四层只有一个MSCAM，多尺度卷积块
        self.mscb4 = MSCBLayer(channels[0], channels[0], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)


        self.up_skip1 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=3//2)
        
        self.down_skip2 = EDCB(in_channels=channels[3], out_channels=channels[2])
        

        #上采样快
        #channels[3]对应原图尺寸通道数/4。  下采样倍数分别为 4 2 2 2  
        self.eucb3 = EUCB(in_channels=channels[0], out_channels=channels[1], kernel_size=eucb_ks, stride=eucb_ks//2)
        #大核分组注意力门
        self.lgag3 = LGAG3(F_g=channels[1],F_k=channels[1], F_l=channels[1], F_int=channels[1]//2, kernel_size=lgag_ks, groups=channels[1]//2)
        self.mscb3 = MSCBLayer(channels[1], channels[1], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)

        self.eucb2 = EUCB(in_channels=channels[1], out_channels=channels[2], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.lgag2 = LGAG3(F_g=channels[2], F_k=channels[2],F_l=channels[2], F_int=channels[2]//2, kernel_size=lgag_ks, groups=channels[2]//2)
        self.mscb2 = MSCBLayer(channels[2], channels[2], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        
        self.eucb1 = EUCB(in_channels=channels[2], out_channels=channels[3], kernel_size=eucb_ks, stride=eucb_ks//2)
        self.lgag1 = LGAG3(F_g=channels[3],F_k=channels[3], F_l=channels[3], F_int=int(channels[3]/2), kernel_size=lgag_ks, groups=int(channels[3]/2))
        self.mscb1 = MSCBLayer(channels[3], channels[3], n=1, stride=1, kernel_sizes=kernel_sizes, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, activation=activation)
        
        #通道注意力快
        self.cab4 = CAB(channels[0])
        self.cab3 = CAB(channels[1])
        self.cab2 = CAB(channels[2])
        self.cab1 = CAB(channels[3])
        
        #空间注意力快块
        self.sab = SAB()
    
      
    def forward(self, x, skips, skip_1):
            
        # MSCAM4
        d4 = self.cab4(x)*x
        d4 = self.sab(d4)*d4 
        d4 = self.mscb4(d4)
        
        # EUCB3
        d3 = self.eucb3(d4)

        #测分支的2层卷积上采样
        ux = self.up_skip1(skip_1[1])
        #测分支的1层卷积下采样
        dx = self.down_skip2(skip_1[0])

        # 计算 EUCB 参数量和 FLOPs
        eucb_params, eucb_flops = calc_params_flops(ux, skip_1[1].shape)
        print(f"EUCB: Parameters = {eucb_params:,}, FLOPs = {eucb_flops:.3f} GFLOPs")

        # 计算 EDCB 参数量和 FLOPs
        edcb_params, edcb_flops = calc_params_flops(dx, skip_1[0].shapse)
        print(f"EDCB: Parameters = {edcb_params:,}, FLOPs = {edcb_flops:.3f} GFLOPs")
       
        # LGAG3
        x3 = self.lgag3(g=d3, k = ux, x=skips[0])
        
        # Additive aggregation 3
        d3 = d3 + x3
        
        # MSCAM3
        d3 = self.cab3(d3)*d3
        d3 = self.sab(d3)*d3  
        d3 = self.mscb3(d3)
        
        # EUCB2
        d2 = self.eucb2(d3)
        
        # LGAG2
        x2 = self.lgag2(g=d2, k = dx,  x=skips[1])
        
        # Additive aggregation 2
        d2 = d2 + x2 
        
        # MSCAM2
        d2 = self.cab2(d2)*d2
        d2 = self.sab(d2)*d2
        d2 = self.mscb2(d2)
        
        # EUCB1
        d1 = self.eucb1(d2)
        
        # LGAG1
        x1 = self.lgag1(g=d1,k = skip_1[0], x=skips[2])
        
        # Additive aggregation 1
        d1 = d1 + x1 
        
        # MSCAM1
        d1 = self.cab1(d1)*d1
        d1 = self.sab(d1)*d1
        d1 = self.mscb1(d1)
        
        return [d4, d3, d2, d1]
