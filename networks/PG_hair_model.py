import torch
import torch.nn as nn
from torch.nn import init
import functools
from tools.wtconv2d import WTConv2d


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, position: int, config: dict = None):
        super().__init__()

        base_config = {
            'nhead': 4,
            'dropout': 0.1,
            'num_layers': 2
        }

        if position == 1:
            base_config['nhead'] = 8
            base_config['num_layers'] = 3

        config = config or base_config
        self.nhead = config['nhead']
        self.dropout_rate = config['dropout']
        self.num_layers = config['num_layers']
        self.position = int(position)

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
        pos = torch.arange(0, seq_len, device=self.device).unsqueeze(0).repeat(B, 1)  # [B, seq_len]
        pos_embed = self.pos_embedding(pos)  # [B, seq_len, d_model]

        if pos_embed.shape[1] != x_seq.shape[1]:
            raise ValueError(f"Position code length {pos_embed.shape[1]} sequence length {x_seq.shape[1]} mismatch.")

        z = self.dropout((x_seq * self.scale) + pos_embed)

        out = self.transformer_encoder(z)  # [B, seq_len, C]
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
             gpu_ids=[], transf_enc_pos=[], transf_dec_pos=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'unet_64':
        net = UnetGenerator(input_nc, output_nc, 3, ngf,
                            norm_layer=norm_layer, use_dropout=use_dropout,
                            transf_enc_pos=transf_enc_pos,
                            transf_dec_pos=transf_dec_pos)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 transf_enc_pos=[], transf_dec_pos=[]):
        super().__init__()

        self.transf_enc_pos = transf_enc_pos
        self.transf_dec_pos = transf_dec_pos
        current_depth = 0

        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None,
            submodule=None,
            innermost=True,
            norm_layer=norm_layer,
            position=current_depth,
            is_encoder=True,
            transf_enc_pos=self.transf_enc_pos,
            transf_dec_pos=self.transf_dec_pos
        )
        current_depth += 1

        for i in range(1, num_downs):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                position=current_depth,
                is_encoder=True,
                transf_enc_pos=self.transf_enc_pos,
                transf_dec_pos=self.transf_dec_pos
            )
            current_depth += 1

        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            position=current_depth,
            is_encoder=False,
            transf_enc_pos=self.transf_enc_pos,
            transf_dec_pos=self.transf_dec_pos
        )
        current_depth -= 1

        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            position=current_depth,
            is_encoder=False,
            transf_enc_pos=self.transf_enc_pos,
            transf_dec_pos=self.transf_dec_pos
        )
        current_depth -= 1

        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
            position=current_depth,
            is_encoder=False,
            transf_enc_pos=self.transf_enc_pos,
            transf_dec_pos=self.transf_dec_pos
        )

        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            position=0,
            is_encoder=False,
            transf_enc_pos=self.transf_enc_pos,
            transf_dec_pos=self.transf_dec_pos
        )

    def forward(self, x, x_r, mask):
        # x = x * mask
        result = self.model(x)
        if isinstance(result, tuple):
            return result[0]
        return result * mask


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 position=None, is_encoder=True,
                 transf_enc_pos=[], transf_dec_pos=[]):
        super().__init__()

        self.outermost = outermost
        self.innermost = innermost
        self.use_dropout = use_dropout
        self.position = position
        self.is_encoder = is_encoder
        self.transf_enc_pos = transf_enc_pos
        self.transf_dec_pos = transf_dec_pos

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        dwtconv = WTConv2d(inner_nc, inner_nc, kernel_size=4, wt_levels=3)  # optional 4x4 Conv
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        transformer_modules = []

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv, dwtconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            transformer_blk = False

            if self.is_encoder and self.position in self.transf_enc_pos:
                transformer_blk = True
            elif not self.is_encoder and self.position in self.transf_dec_pos:
                transformer_blk = True

            if transformer_blk:
                self.transformer_enc = TransformerBlock(
                    d_model=inner_nc,
                    position=position
                )
                self.self_att = SelfAttentionBlock(position=position)
                transformer_modules = [self.transformer_enc, self.self_att]
            else:
                self.transformer_enc = None
                self.self_att = None

            model = down + transformer_modules + [submodule] + up

            if use_dropout:
                model += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            result = x
            attention_weights = None

            for layer in self.model:
                if isinstance(layer, SelfAttentionBlock) and layer.return_attn:
                    result, attention_weights = layer(result)
                else:
                    result = layer(result)

            if attention_weights is not None:
                return torch.cat([x, result], 1), attention_weights
            else:
                return torch.cat([x, result], 1)