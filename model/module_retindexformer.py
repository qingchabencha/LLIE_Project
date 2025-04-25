import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class ColorComplementorCVAE(nn.Module):
    def __init__(self, lattent_dim, encode_layer , feature_in=4, feature_out=3, **kwargs):
        super(ColorComplementorCVAE, self).__init__()
        self.lattent_dim = lattent_dim
        self.feature_in = feature_in
        self.feature_out = feature_out
        self.encode_layer = encode_layer
        self.decoder_dim_in = self.lattent_dim

        self.color_amplifier = nn.Sequential(
            nn.Conv2d(feature_out, feature_out, kernel_size=1),
            nn.ReLU()
        )

        self.encoder_shape_list = [int(self.lattent_dim / 2 ** i) for i in range(self.encode_layer)]
        self.encoder_shape_list.sort()
        self.decoder_shape_list = self.encoder_shape_list[::-1]

        self.encoder = nn.ModuleList()
        for i in range(self.encode_layer):
            in_channels = self.feature_in if i == 0 else self.encoder_shape_list[i-1]
            self.encoder.append(nn.Conv2d(in_channels, self.encoder_shape_list[i], kernel_size=3, stride=2, padding=1, bias=True))
            self.encoder.append(nn.ReLU())
            in_channels = self.encoder_shape_list[i]

        self.encoder_mu = nn.Linear(self.encoder_shape_list[-1], self.lattent_dim)
        self.encoder_log_var = nn.Linear(self.encoder_shape_list[-1], self.lattent_dim)

        self.decoder_same_representation = nn.ModuleList()
        for i in range(self.encode_layer):
            in_channels = self.feature_in if i == 0 else self.encoder_shape_list[i-1]
            self.decoder_same_representation.append(nn.Conv2d(in_channels, self.encoder_shape_list[i], kernel_size=3, stride=2, padding=1, bias=True))
            if i != self.encode_layer - 1:
                self.decoder_same_representation.append(nn.ReLU())

        self.decoder = nn.ModuleList()
        for i in range(self.encode_layer):
            in_channels = self.decoder_dim_in + self.encoder_shape_list[-1] if i == 0 else self.decoder_shape_list[i-1]
            self.decoder.append(nn.ConvTranspose2d(in_channels, self.decoder_shape_list[i], kernel_size=3, stride=2, padding=1, output_padding=1, bias=True))
            if i != self.encode_layer - 1:
                self.decoder.append(nn.ReLU())

        self.restore = nn.Conv2d(self.decoder_shape_list[-1], self.feature_out, kernel_size=3, stride=1, padding=1, bias=True)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return torch.randn(1)

    def encode(self, target_img, illu_fea):
        amplified_img = self.color_amplifier(target_img)
        fused_input = torch.cat((amplified_img, illu_fea), dim=1)
        latent = fused_input
        for layer in self.encoder:
            latent = layer(latent)
        latent = latent.permute(0, 2, 3, 1).contiguous()
        mu = self.encoder_mu(latent)
        log_var = self.encoder_log_var(latent)
        z = self.reparameterize(mu, log_var)
        z = z.permute(0, 3, 1, 2).contiguous()
        return mu, log_var, z

    def decode(self, latent, lited_up_image, illu_fea):
        feature = torch.cat((lited_up_image, illu_fea), dim=1)
        for layer in self.decoder_same_representation:
            feature = layer(feature)
        restored_img = torch.cat((latent, feature), dim=1)
        for layer in self.decoder:
            restored_img = layer(restored_img)
        restored_img = self.restore(restored_img)
        return restored_img

    def forward(self, lited_up_image, illu_fea, target=None):
        if self.training:
            mu, log_var, latent = self.encode(target, illu_fea)
        else:
            mu = None
            log_var = None
            len_batch = lited_up_image.size(0)
            hidden = self.lattent_dim
            H = lited_up_image.size(2) // 2 ** self.encode_layer
            W = lited_up_image.size(3) // 2 ** self.encode_layer
            latent = torch.randn(len_batch, hidden, H, W).to(lited_up_image.device)
        restored_img = self.decode(latent, lited_up_image, illu_fea)
        return restored_img, mu, log_var

class ColorComplementorVAE(nn.Module):
    def __init__(self, lattent_dim, encode_layer , feature_in=4, feature_out=3, **kwargs):
        super(ColorComplementorVAE, self).__init__()
        self.lattent_dim = lattent_dim
        self.feature_in = feature_in
        self.feature_out = feature_out # output image size
        self.encode_layer = encode_layer
        self.decoder_dim_in = self.lattent_dim * 2
        
        # encode the target picture into latent space
        self.encoder_shape_list = [int(self.lattent_dim / 2 ** i) for i in range(self.encode_layer)]
        self.encoder_shape_list.sort() # [8, 16, 32]
        self.decoder_shape_list = self.encoder_shape_list[::-1] # [32, 16, 8]
        
        self.encoder = nn.ModuleList()
        for i in range(self.encode_layer):
            if i == 0:
                self.encoder.append(nn.Conv2d(self.feature_out, self.encoder_shape_list[i], kernel_size=3, stride=2, padding=1, bias=True))
            else:
                self.encoder.append(nn.Conv2d(int(self.encoder_shape_list[i-1]), int(self.encoder_shape_list[i]), kernel_size=3, stride=2, padding=1, bias=True))
            self.encoder.append(nn.ReLU())
            
        self.encoder_mu = nn.Linear(self.encoder_shape_list[-1], self.lattent_dim) # 32 -> 32
        self.encoder_log_var = nn.Linear(self.encoder_shape_list[-1], self.lattent_dim) # 32 -> 32
        
        
        # decode part
        self.decoder_same_representation = nn.ModuleList()
        for i in range(self.encode_layer):
            if i == 0:
                self.decoder_same_representation.append(nn.Conv2d(self.feature_in, self.encoder_shape_list[i], kernel_size=3, stride=2, padding=1, bias=True))
            else:
                self.decoder_same_representation.append(nn.Conv2d(int(self.encoder_shape_list[i-1]), int(self.encoder_shape_list[i]), kernel_size=3, stride=2, padding=1, bias=True))
            if i != self.encode_layer - 1:
                self.decoder_same_representation.append(nn.ReLU())
        self.decoder = nn.ModuleList()
        for i in range(self.encode_layer):
            if i == 0:
                self.decoder.append(nn.ConvTranspose2d(self.decoder_dim_in, self.decoder_shape_list[i], kernel_size=3, stride=2, padding=1, output_padding=1, bias=True))
            else:
                self.decoder.append(nn.ConvTranspose2d(int(self.decoder_shape_list[i-1]), int(self.decoder_shape_list[i]), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True))
            if i != self.encode_layer - 1:
                self.decoder.append(nn.ReLU())
        
        # output restored image
        self.restore = nn.Conv2d(self.decoder_shape_list[-1], self.feature_out, kernel_size=3, stride=1, padding=1, bias=True)
            

    
    def reparameterize(self, mu, log_var):  # 从编码器输出的均值和对数方差中采样得到潜在变量z
        if self.training:
            std = torch.exp(0.5 * log_var)  # 计算标准差
            eps = torch.randn_like(std)  # 从标准正态分布中采样得到随机噪声
            return mu + eps * std  # 根据重参数化公式计算潜在变量z
        else:
            return torch.randn(1)
    
    def encode(self, target_img):
        latent = target_img
        for layer in self.encoder:
            latent = layer(latent)
        latent = latent.permute(0, 2, 3, 1).contiguous()
        mu = self.encoder_mu(latent)  # 将特征图展平并通过线性层计算均值
        log_var = self.encoder_log_var(latent)  # 将特征图展平并通过线性层计算对数方差
        latent = self.reparameterize(mu, log_var)  # 通过重参数化采样得到潜在变量z
        return mu, log_var, latent.permute(0, 3, 1, 2).contiguous()  # 将潜在变量z重新排列为原始形状并返回
    
    def decode(self, latent, lited_up_image, illu_fea):
        feature = torch.cat((lited_up_image, illu_fea), dim=1)
        for layer in self.decoder_same_representation:
            feature = layer(feature)
        restored_img = torch.cat((latent, feature), dim=1)  # 将潜在变量z和特征图拼接在一起
        # U-Net 恢复图像
        for layer in self.decoder:
            restored_img = layer(restored_img)
        restored_img = self.restore(restored_img)  # 将恢复后的图像通过卷积层映射到输出通道数
        return restored_img
    
    def forward(self, lited_up_image, illu_fea, target=None):
        """
        restore the color

        Args:
            lited_up_image (_type_): image after the illumination estimation, for instructing generation
            illumination_map (_type_): illumination of each pixel in the image, for instructing generation
            target (_type_, optional): the output image, None if not training. Defaults to None.
        Returns:
            _type_: _description_
        """
        if self.training:
            mu, log_var, latent = self.encode(target)
        else:
            mu = None
            log_var = None
            len_batch =  lited_up_image.size(0)
            hidden = self.lattent_dim
            H = lited_up_image.size(2) // 2 ** self.encode_layer
            W = lited_up_image.size(3) // 2 ** self.encode_layer
            latent = torch.randn(len_batch, hidden, H, W).to(lited_up_image.device)  # 随机生成潜在变量z
        restored_img = self.decode(latent, lited_up_image, illu_fea)
        return  restored_img, mu, log_var

class Illumination_Estimator(nn.Module):
    def __init__(self, model_type , hidden_channel_dim, feature_in=4, feature_out=3, **kwargs):
        """

        Args:
            model_type (_type_): CNN or FFC; CNN means the original method implmented according to the sota paper
            hidden_channel_dim (_type_): number of channels in the hidden layer.
            feature_in (int, optional): _description_. Defaults to 4 = 1 illumination channel + 3 RGB channels.
            feature_out (int, optional): _description_. Defaults to 3 = 3 light-up channels for RGB

        Raises:
            ValueError: Light-up Feature
            ValueError: Light-up Map
        """
        
        super(Illumination_Estimator, self).__init__()
        self.hidden = hidden_channel_dim
        self.feature_in = feature_in
        self.feature_out = feature_out

        if model_type == "CNN":
            self.conv1 = nn.Conv2d(self.feature_in, self.hidden, kernel_size=1, stride=1, padding=0, bias=True)
            if kwargs.get("CNN_Kernel_size", None) is None:
                print("Illumination estimator module do not initiate kernel size, using default value 5")
            self.conv_fea = nn.Conv2d(self.hidden, self.hidden, kernel_size=kwargs.get('CNN_Kernel_size', 5), groups=self.hidden, stride=1, padding=2, bias=True)
            self.conv_map = nn.Conv2d(self.hidden, self.feature_out, kernel_size=1, stride=1, padding=0, bias=True)
            if kwargs.get("CNN_Feature_active", None) is None:
                self.active = None
            elif kwargs.get("CNN_Feature_active", None) == "None":
                self.active = None
            elif kwargs.get("CNN_Feature_active", None) == "sigmoid":
                self.active = nn.Sigmoid()
            elif kwargs.get("CNN_Feature_active", None) == "relu":
                self.active = nn.ReLU()
            else:
                raise ValueError("CNN_Feature_active should be None or sigmoid, the method {} is not supported".format(kwargs.get("CNN_Feature_active", None)))
        
        
        elif model_type == "FFC":
            pass
        else:
            raise ValueError("model_illumination_estimator_model should be CNN or FFC the method {} is not supported".format(IE_Type))
        
    def forward(self, img, img_illumination):
        """
        Forward pass of the model.

        Args:
            img (torch.Tensor): The original low-light image.
            img_illumination (torch.Tensor): The illumination image. (calculated by the perceived brightness)

        Returns:
            _type_: Low, High
        """
        x = torch.concat((img, img_illumination), dim=1)
        # ! original paper do not use any active function
        x = self.conv1(x)
        if self.active is not None:
            x = self.active(x)
        illu_fea  = self.conv_fea(x)
        if self.active is not None:
            x = self.active(x)
        illu_map = self.conv_map(illu_fea)
        
        return illu_fea, illu_map
    
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
    
class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x) # 不同channel相同位置pixel进行fussion
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2))) # multihead attention preparation, split all q, k, v and illu_attn into multihead (B, n (H*W), C) -> (B, head_num, n, d)
        v = v * illu_attn # (batch, head_hum, h*w, dim_head) * (batch, head_num, h*w, dim_head)
        # q: b,heads,hw,c
        q = q.transpose(-2, -1) # (b, head_num, dim_head, h*w)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2) # L2 normalization on feature 
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q, (batch, head_num, dim_head, dim_head), 本质上不还是相同位置的像素进行fussion么？没有不同位置的pixel进行合并
        attn = attn * self.rescale # learnable parameter to rescale attention
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw [hw 进行fusion，根据不同的attn]，attention依旧是一句不同channel的占比进行fussion？
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c) # linear transform一下，融合一下feature
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1) # 根据输入的v来计算一个可以训练的position embedding？
        out = out_c + out_p

        return out
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn # feed forward, have activate function here
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)  
    
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)
 
    
class IGAB(nn.Module):
    # Illumination Guided Attention Block
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))
        
    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

 
class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=30, level=2, num_blocks=[2,4,4]):
        super(Denoiser, self).__init__()
        """
        Denoiser module
        
        The attention is happenning in the channel dimension
        
        :Args:
            in_dim (int, optional): Number of channels in the input image. Defaults to 3.
            out_dim (int, optional): _description_.  Number of channels of the input image
            dim (int, optional): _description_. Number of channels of the light-up feature image
            level (int, optional): _description_. U-shaped encoder-decoder structure, number of levels. Defaults to 2.
        """
        self.dim= dim
        self.level=level
        # Trainable Embedding, embedding the light-uped image into high-dimensional space
        # to represent the noise
        self.embedding = nn.Conv2d(in_dim, self.dim, kernel_size=3, padding=1, stride=1, bias=False)
        
        # encode the image and feature map into different resolution levels representation
        self.encoder_layers = nn.ModuleList()
        dim_level = dim
        for i in range(self.level):
            self.encoder_layers.append(nn.ModuleList(
                [
                    IGAB(dim=dim_level, dim_head=dim,num_blocks=num_blocks[i], heads=dim_level//dim), 
                    # number of heads = dim_level // dim, 1 in the first level, 2 in the second and 4 etc.
                    nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False), # downsample the image
                    nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False) # Illumination feature map downsampleing
                ]
            ))
            dim_level *= 2
            
        self.bottleneck = IGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])
        
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0), # Upsample the image
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False), # encode the image in higher resolution
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2
            
        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # activation function
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, lit_up_image, illu_fea):
         # Embedding
        fea = self.embedding(lit_up_image)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea,illu_fea)  # bchw, 不同channel，相同位置像素的融合通过MHA融合之后，再通过forward部分往前传进行，进行不同像素之间的融合，不过还是较为底层的
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea,illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level-1-i]
            fea = LeWinBlcok(fea,illu_fea)

        # Mapping
        img_restoration = self.mapping(fea)
        out = torch.clamp(img_restoration + lit_up_image, 0, 1)

        return out