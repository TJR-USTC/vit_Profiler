from collections import OrderedDict
from functools import partial
import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    '''
    随机深度drop path
    x:输入张量
    drop_prob:drop概率
    training:是否在训练模式
    '''
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)                                     # 在batch维度上进行drop，生成与x的维度匹配的形状，只保持batch维度不同
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)   # 生成与x大小相同的随机张量
    random_tensor.floor_()                                                          # 二值化（小于keep_prob的值为0，大于等于keep_prob的值为1）
    output = x.div(keep_prob) * random_tensor                                       # 将输入张量按keep_prob缩放，并与随机张量相乘，实现drop path
    return output
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        '''
        drop_prob:drop概率
        '''
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
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
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])           # 网格大小(区域网格)
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

class MLP(nn.Module):
    def __init__(self, in_feature, hidden_feature=None, out_feature=None, act_layer=nn.GELU, drop=0.):
        '''
        in_feature:输入特征维度
        hidden_feature:隐藏层特征维度，若为None则等于in_feature * 4
        out_feature:输出特征维度，若为None则等于in_feature
        act_layer:激活函数层
        drop:dropout概率
        '''
        super().__init__()
        hidden_feature = hidden_feature or in_feature * 4                 # 若未指定隐藏层维度，则使用输入维度 * 4
        out_feature = out_feature or in_feature                           # 若未指定输出维度，则使用输入维度
        self.fc1 = nn.Linear(in_feature, hidden_feature)                  # 第一层全连接层
        self.act = act_layer                                              # 激活函数层
        self.fc2 = nn.Linear(hidden_feature, out_feature)                 # 第二层全连接层
        self.drop = nn.Dropout(drop)                                      # dropout层

    def forward(self, x):
        x = self.fc1(x)                                                   # 第一层全连接
        x = self.act(x)                                                   # 激活函数
        x = self.drop(x)                                                  # dropout
        x = self.fc2(x)                                                   # 第二层全连接
        x = self.drop(x)                                                  # dropout
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        '''
        dim:输入token维度
        num_heads:注意力头数
        mlp_ratio:MLP隐藏层维度与输入维度之比
        qkv_bias:生成QKV时是否添加偏置
        qk_scale:缩放因子，None则使用1/sqrt(head_dim)
        drop:MLP输出dropout概率
        attn_drop:注意力dropout概率
        drop_path:drop patch比例，在embedding中使用
        act_layer:激活函数层
        norm_layer:归一化层
        '''
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)                                                # 第一个归一化层
        self.attn = Attention(                                                      # 注意力层
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)  if drop_path > 0. else nn.Identity()  # 随机深度层
        self.norm2 = norm_layer(dim)                                                # 第二个归一化层
        mlp_hidden_dim = int(dim * mlp_ratio)                                       # MLP隐藏层维度
        self.mlp = MLP(in_feature=dim, hidden_feature=mlp_hidden_dim,               # MLP层
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))                            # 残差连接与注意力计算
        x = x + self.drop_path(self.mlp(self.norm2(x)))                             # 残差连接与MLP计算
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_rate=0., 
                 attn_drop_rate=0., drop_path_rate=0.,embed_layer=PatchEmbedding,norm_layer=None,
                 act_layer=None):
        '''
        img_size:图像大小
        patch_size:每个区域大小
        in_chans:输入通道数
        num_classes:分类数量
        embed_dim:每个token维度
        depth:Transformer块数量
        num_heads:注意力头数量
        mlp_ratio:MLP隐藏层维度与输入维度之比
        qkv_bias:生成QKV时是否添加偏置
        qk_scale:缩放因子，None则使用1/sqrt(head_dim)
        representation_size:表示层大小
        distilled:是否使用蒸馏
        drop_rate:MLP输出dropout概率
        attn_drop_rate:注意力dropout概率
        drop_path_rate:drop patch比例，在embedding中使用
        embed_layer:嵌入层
        norm_layer:归一化层
        act_layer:激活函数层
        '''
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes                                              # 分类数量
        self.num_features = self.embed_dim = embed_dim                              # 特征维度等于嵌入维度
        self.num_tokens = 2 if distilled else 1                                     # 分类token数量
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)                  # 归一化层,默认使用LayerNorm,eps防止参数为0
        act_layer = act_layer or nn.GELU()                                          # 激活函数层，
        self.patch_embed = embed_layer(                                             # 图像分块嵌入层
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches                                  # 区域数量

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))                                 # 分类token参数初始化(batch_size=1, token_num=1, embed_dim)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None         # 蒸馏token参数初始化
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))     # 位置嵌入参数初始化
        self.pos_drop = nn.Dropout(p=drop_rate)                                                     # 位置嵌入dropout层

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]                          # 每个块的drop path比例线性变化（从0到drop_path_rate）
        self.blocks = nn.Sequential(*[                                                              # Transformer块列表
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)                                           # 最终归一化层
        if representation_size and not distilled:                                   # 若有表示层且不使用蒸馏
            self.has_logits = True
            self.num_features = representation_size                                 # 特征维度等于表示层大小    
            self.pre_logits = nn.Sequential(OrderedDict([                           # 表示层
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()                                         # 恒等映射
        
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()  # 分类头
        self.head_dist = nn.Linear(self.num_features, num_classes) if distilled and num_classes > 0 else nn.Identity()  # 蒸馏分类头
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)                             # 位置嵌入参数初始化
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)                        # 蒸馏token参数初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)                             # 分类token参数初始化
        self.apply(self._init_weights)                                              # 权重初始化

    def forward_features(self, x):
        B = x.shape[0]                                                              # Batch大小
        x = self.patch_embed(x)                                                     # 图像分块嵌入

        cls_tokens = self.cls_token.expand(B, -1, -1)                               # 扩展分类token以匹配batch大小
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)                                   # 在序列维度上连接分类token与嵌入
        else:
            dist_tokens = self.dist_token.expand(B, -1, -1)                         # 扩展蒸馏token以匹配batch大小
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)                      # 在序列维度上连接分类token、蒸馏token与嵌入

        x = x + self.pos_embed                                                      # 添加位置嵌入
        x = self.pos_drop(x)                                                        # 位置嵌入dropout

        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:                                                 # dist_token为None，默认提取cls_token对应输出
            return self.pre_logits(x[:,0])
        else:
            return x[:,0],x[:,1]
    
    def forward(self, x):
        x = self.forward_features(x)
        if self.dist_token is not None:
            # 蒸馏模式：forward_features 返回两个输出
            x, x_dist = self.head(x[0]), self.head_dist(x[1])                       # 分别通过分类头和蒸馏分类头
            if self.training and not torch.jit.is_scripting():                      # 训练模式且非JIT脚本模式
                return x, x_dist                                                    # 返回两个头部结果
        else:
            # 正常模式：forward_features 返回单个输出
            x = self.head(x)                                                        # 仅通过分类头进行预测
        return x
    def _init_weights(self, m):
        # 判断模块是不是线性层 
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:                                                      # 如果有偏置，则初始化为0
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')                           # 对卷积层权重进行Kaiming正态初始化
            if m.bias is not None:                                                      # 如果有偏置，则初始化为0
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)                                                     # 对LayerNorm的偏置初始化为0
            nn.init.ones_(m.weight)                                                    # 对LayerNorm的权重初始化为1

def vit_base_patch16_224(num_classes:int =1000, pretrained:bool=False):
    '''
    创建一个ViT模型，使用16x16的图像块和224x224的输入图像大小
    '''
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        representation_size=None, num_classes=num_classes)
    return model
