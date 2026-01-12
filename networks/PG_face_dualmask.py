import torch
import torch.nn as nn
from timm.models.convmixer import Residual
from torch.nn import init
import functools
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tools.adsfa_module import Adp_spa_freq_attention
from tools.lde_module import DMlpPlug
from tools.phase_mix import PhaseMix


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, config: dict = None):
        super().__init__()

        base_config = {
            'nhead': 4,
            'dropout': 0.1,
            'num_layers': 2
        }

        config = config or base_config
        self.nhead = config['nhead']
        self.dropout_rate = config['dropout']
        self.num_layers = config['num_layers']

        self.hid_dim = d_model
        self.dim_feedforward = self.hid_dim * 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            self.num_layers
        )

        self.pos_embedding = nn.Embedding(10000, d_model)
        self.scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(self.device)
        self.dropout = nn.Dropout(0.1)
        self.BatchNorm = nn.BatchNorm2d(d_model)

    def reshape(self, x):
        B, C, H, W = x.shape
        patch_size = 1
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        num_patches = x.shape[2] * x.shape[3]
        x = x.contiguous().view(B, num_patches, C)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W

        x_seq = self.reshape(x)

        pos = torch.arange(0, seq_len, device=self.device).unsqueeze(0).repeat(B, 1)
        pos_embed = self.pos_embedding(pos)

        if pos_embed.shape[1] != x_seq.shape[1]:
            raise ValueError(f"Position code length {pos_embed.shape[1]} sequence length {x_seq.shape[1]} mismatch.")

        z = self.dropout((x_seq * self.scale) + pos_embed)

        out = self.transformer_encoder(z)

        out = out.permute(0, 2, 1).view(B, self.hid_dim, H, W)
        out = self.BatchNorm(out)

        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, position: int, config: dict = None):
        super().__init__()

        base_config = {
            'n_heads': 4,
            'return_attn': False
        }

        if position == 1:
            base_config['n_heads'] = 8

        config = config or base_config
        self.position = position
        self.n_heads = config['n_heads']
        self.return_attn = config['return_attn']

        self.fc_q = None
        self.fc_k = None
        self.fc_v = None
        self.fc_o = None
        self.scale = None

    def forward(self, x):
        batch_size, in_dim, height, width = x.shape
        seq_len = height * width

        if self.fc_q is None:
            self.head_dim = in_dim // self.n_heads
            assert self.head_dim > 0, f"head_dim must > 0, in_dim={in_dim}, n_heads={self.n_heads}"

            self.fc_q = nn.Linear(in_dim, in_dim).to(x.device)
            self.fc_k = nn.Linear(in_dim, in_dim).to(x.device)
            self.fc_v = nn.Linear(in_dim, in_dim).to(x.device)
            self.fc_o = nn.Linear(in_dim, in_dim).to(x.device)
            self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(x.device)

        x_flat = x.view(batch_size, in_dim, seq_len).permute(0, 2, 1)

        Q = self.fc_q(x_flat)
        K = self.fc_k(x_flat)
        V = self.fc_v(x_flat)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        weighted = torch.matmul(attention, V)

        weighted = weighted.permute(0, 2, 1, 3).contiguous()
        weighted = weighted.view(batch_size, -1, in_dim)
        weighted = self.fc_o(weighted)

        weighted = weighted + x_flat

        weighted = weighted.permute(0, 2, 1).view(batch_size, in_dim, height, width)

        if self.return_attn:
            return weighted, attention
        else:
            return weighted


class Identity(nn.Module):
    def forward(self, x):
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        self.conv1x1 = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False)
    def forward(self, x):
        return self.conv1x1(x) + self.residual(x)


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain)
    return net


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError(f'Normalization layer {norm_type} not found')
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'unet_64':
        net = UNetGenerator(input_nc, output_nc, 3, ngf,
                                     norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


class UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNetGenerator, self).__init__()

        self.ngf = ngf

        self.phase_swap = PhaseMix(swap_strength=1.0)
        self.residual = ResidualBlock(3, 3)
        self.lde = DMlpPlug()

        self.enc_original_conv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 4, 2, 1),
            norm_layer(ngf),
            nn.LeakyReLU(0.2, True)
        )
        self.enc_original_conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            norm_layer(ngf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.enc_original_conv3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            norm_layer(ngf * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.enc_original_conv4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )

        self.enc_reference_conv1 = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 4, 2, 1),
            norm_layer(ngf),
            nn.LeakyReLU(0.2, True)
        )
        self.enc_reference_conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            norm_layer(ngf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.enc_reference_conv3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            norm_layer(ngf * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.enc_reference_conv4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, True)
        )

        self.mid_transformer = TransformerBlock(d_model=ngf * 8)

        self.channel_adjust1 = nn.Conv2d(ngf * 12, ngf * 8, 1)
        self.channel_adjust2 = nn.Conv2d(ngf * 6, ngf * 4, 1)
        self.channel_adjust3 = nn.Conv2d(ngf * 3, ngf * 2, 1)

        self.dec_up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )

        self.dec_up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )

        self.dec_up3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1),
            norm_layer(ngf),
            nn.ReLU(True)
        )

        self.dec_up4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(0.5) if use_dropout else None
        self.up_high_freq_aware = Adp_spa_freq_attention(feat_dim=output_nc, patch_size=16)

    def forward(self, original_img, reference_img, face_mask):
        phase_swapped_ori, phase_swapped_ref = self.phase_swap(original_img, reference_img)
        phase_fusion = phase_swapped_ori + phase_swapped_ref
        phase_swapped_feature = self.residual(phase_fusion)
        phase_swapped_feature = phase_swapped_feature * face_mask
        original_img = original_img * face_mask
        phase_swapped_features = self.lde(phase_swapped_feature)

        orig_enc1 = self.enc_original_conv1(original_img)
        orig_enc2 = self.enc_original_conv2(orig_enc1)
        orig_enc3 = self.enc_original_conv3(orig_enc2)
        orig_enc4 = self.enc_original_conv4(orig_enc3)

        phase_enc1 = self.enc_reference_conv1(phase_swapped_features)
        phase_enc2 = self.enc_reference_conv2(phase_enc1)
        phase_enc3 = self.enc_reference_conv3(phase_enc2)
        phase_enc4 = self.enc_reference_conv4(phase_enc3)

        # F_fusion
        combined_features = orig_enc4 + phase_enc4
        refined_features = self.mid_transformer(combined_features)

        dec1 = self.dec_up1(refined_features)
        skip3 = torch.cat([orig_enc3, phase_enc3], dim=1)
        dec1 = torch.cat([dec1, skip3], dim=1)
        dec1 = self.channel_adjust1(dec1)
        if self.dropout:
            dec1 = self.dropout(dec1)

        dec2 = self.dec_up2(dec1)
        skip2 = torch.cat([orig_enc2, phase_enc2], dim=1)
        dec2 = torch.cat([dec2, skip2], dim=1)
        dec2 = self.channel_adjust2(dec2)
        if self.dropout:
            dec2 = self.dropout(dec2)

        dec3 = self.dec_up3(dec2)
        skip1 = torch.cat([orig_enc1, phase_enc1], dim=1)
        dec3 = torch.cat([dec3, skip1], dim=1)
        dec3 = self.channel_adjust3(dec3)
        residual = self.dec_up4(dec3)
        residual = self.up_high_freq_aware(residual)

        adv_img = original_img + residual
        adv_img = torch.clamp(adv_img, -1, 1)

        return adv_img * face_mask