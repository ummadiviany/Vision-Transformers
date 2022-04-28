import torch.nn as nn
import torch

class PatchEmbeeding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size//patch_size) ** 2


        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )


    def forward(self, x):
        
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1,2)

        return x
    
    
class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p = 0., proj_p=0.) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim//n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    
    def forward(self,x):
        n_samples, n_tokens, dim = x.shape 
        # (N, 576, 768)
        # print("n_samples, n_tokens, dim",n_samples, n_tokens, dim)
        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)
        
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )
        
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )
        

        q, k, v = qkv[0], qkv[1], qkv[2]
        # print("q.shape",q.shape)
        # print("k.shape",k.shape)
        k_t = k.transpose(-2,-1)
        # print("k_t shape",k_t.shape)
        dp = (q @ k_t) * self.scale
        # print("dp.shape",dp.shape)
        attn = dp.softmax(dim=-1)
        # print("attn.shape",attn.shape)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        # print("weighted_avg.shape",weighted_avg.shape)
        weighted_avg = weighted_avg.transpose(1, 2)
        # print("weighted_avg.shape",weighted_avg.shape)

        weighted_avg = weighted_avg.flatten(start_dim=2)
        # print("weighted_avg.shape",weighted_avg.shape)
        x = self.proj(weighted_avg)
        # print("x.shape",x.shape)
        x = self.proj_drop(x)
        # print("x.shape",x.shape)
        return x

    
class MLP(nn.Module):
    def __init__(self,in_features, hidden_fetures, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_fetures)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_fetures, out_features)
        self.drop = nn.Dropout(p)

    def forward(self,x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


    
class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            n_heads = n_heads,
            qkv_bias = qkv_bias,
            attn_p = attn_p,
            proj_p = p 
        )
        self.norm2 = nn.LayerNorm(dim,eps=1e-6)
        hidden_features = int(dim*mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_fetures=hidden_features,
            out_features=dim
        )


    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
    
class VisionTransformer(nn.Module):

    def __init__(
        self,
        img_size = 384,
        patch_size = 16,
        in_channels = 3,
        n_classes = 1000,
        embed_dim = 768,
        depth = 12,
        n_heads = 12,
        mlp_ratio = 4,
        qkv_bias = True,
        p = 0.,
        attn_p = 0.,
    ):
        super().__init__()

        self.patch_embed = PatchEmbeeding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.patch_embed.n_patches, embed_dim))
        
        self.pos_drop = nn.Dropout(p)
        
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads = n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias = qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )

                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)


    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )

        x = torch.cat((cls_token,x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)

        return x