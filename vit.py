import torch
import torch.nn as nn
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        '''
        img_size:图像大小
        patch_size:每个区域大小
        embed_dim:每个token维度
        norm_layer:归一化层
        '''
        super().__init__()
        self.img_size = (img_size, img_size)                                                    # 图像大小
        self.patch_size = (patch_size, patch_size)                                              # 每个区域大小    
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])           # 网格大小(区域网格)
        self.num_patches = self.grid_size[0] * self.grid_size[1]                                # 区域数量

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)   # 3,224,224 -> 768,14,14
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()                      # 若有则使用归一化层，否则保持不变


    def forward(self, x):
        B,C,H,W = x.shape # Batch, Channel, Height, Width
        assert H == self.img_size[0] and W == self.img_size[1],\
        f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale = None, attn_drop=0., proj_drop=0.):
        '''
        dim:输入token维度
        num_heads:注意力头数量
        qkv_bias:生成QKV时是否添加偏置
        qk_scale:缩放因子，None则使用1/sqrt(head_dim)
        attn_drop:注意力dropout概率
        proj_drop:输出dropout概率
        '''
        super().__init__()
        self.num_heads = num_heads                                      # 注意力头数量 
        head_dim = dim // num_heads                                     # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5                       # 缩放因子
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)               # 通过全连接层生成QKV，并行计算，参数更少
        self.attn_drop = nn.Dropout(attn_drop)                          # 注意力dropout层
        self.proj = nn.Linear(dim, dim)                                 # 输出全连接层
        self.proj_drop = nn.Dropout(proj_drop)                          # 输出dropout层
    
    def forward(self, x):
        B,N,C = x.shape                                                         # Batch, num_patch+1(+class token), embed_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads) # 生成QKV并reshape,B,N.3*C->B,N,3,num_heads,head_dim
        qkv = qkv.permute(2,0,3,1,4)                                            # 调整维度顺序->(3,B,num_heads,N,head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]                                        # 切片Q,K,V，形状均为(B,num_heads,N,head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale                           # 计算注意力得分，形状为(B,num_heads,N,N)
        attn = attn.softmax(dim=-1)                                             # 对最后一个维度进行softmax
        x = (attn @ v).transpose(1,2).reshape(B, N, C)                          # (B,num_heads,N,head_dim)->(B,N,num_heads,head_dim)->(B,N,C)
        attn = self.attn_drop(attn)                                             # 注意力dropout
        
        x = self.proj(x)                                                       # 输出全连接层
        x = self.proj_drop(x)                                                  # 输出dropout
        return x
