import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.hicmamba_utils import *


class HiCMamba(nn.Module):
    def __init__(self, img_size=40, out_chans=1, in_chans=1,
                 embed_dim=32, d_state=16, layer=2, num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 # embed_dim=32, d_state=16, depths=[1, 1, 1, 1, 1, 1, 1, 1, 1], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear',
                 dowsample=Downsample, upsample=Upsample, shift_flag=True, modulator=False,
                 cross_modulator=False, **kwargs):
        super().__init__()
        depths = [layer for _ in range(9)]
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.in_chans = in_chans

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.seq_input_proj = InputProj(in_channel=256, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=out_chans, kernel_size=3, stride=1)

        self.encoderlayer_0 = BasicHolisticScanLayer(dim=embed_dim,
                                                    input_resolution=(img_size,
                                                                      img_size),
                                                    depth=depths[0],
                                                    mlp_ratio=self.mlp_ratio,
                                                    drop=drop_rate,
                                                    drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint, d_state=d_state)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicHolisticScanLayer(dim=embed_dim * 2,
                                                    input_resolution=(img_size // 2,
                                                                      img_size // 2),
                                                    depth=depths[1],
                                                    mlp_ratio=self.mlp_ratio,
                                                    drop=drop_rate,
                                                    drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint, d_state=d_state)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)

        # Bottleneck
        self.conv = BasicHolisticScanLayer(dim=embed_dim * 4,
                                          input_resolution=(img_size // (2 ** 1),
                                                            img_size // (2 ** 1)),
                                          depth=depths[4],
                                          mlp_ratio=self.mlp_ratio,
                                          drop=drop_rate,
                                          drop_path=conv_dpr,
                                          norm_layer=norm_layer,
                                          use_checkpoint=use_checkpoint, d_state=d_state)

        self.upsample_2 = upsample(embed_dim * 4, embed_dim * 2)
        self.decoderlayer_2 = BasicHolisticScanLayer(dim=embed_dim * 4,
                                                    input_resolution=(img_size // 2,
                                                                      img_size // 2),
                                                    depth=depths[7],
                                                    mlp_ratio=self.mlp_ratio,
                                                    drop=drop_rate,
                                                    drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint, d_state=d_state)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicHolisticScanLayer(dim=embed_dim * 2,
                                                    input_resolution=(img_size,
                                                                      img_size),
                                                    depth=depths[8],
                                                    mlp_ratio=self.mlp_ratio,
                                                    drop=drop_rate,
                                                    drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint, d_state=d_state)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, map_data, mask=None):
        x = map_data
        # Input Projection
        # y: (B H*W C)
        y = self.input_proj(x)
        y = self.pos_drop(y)
        # Encoder
        # torch.Size([64, 1600, 32])
        conv0 = self.encoderlayer_0(y, mask=mask)
        # torch.Size([64, 400, 64])
        pool0 = self.dowsample_0(conv0)
        # torch.Size([64, 400, 64])
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        # torch.Size([64, 100, 128])
        pool1 = self.dowsample_1(conv1)

        # Bottleneck
        # torch.Size([64, 100, 128])
        conv4 = self.conv(pool1, mask=mask)

        # torch.Size([64, 400, 64])
        up2 = self.upsample_2(conv4)
        deconv2 = torch.cat([up2, conv1], -1)
        # torch.Size([64, 400, 128])
        deconv2 = self.decoderlayer_2(deconv2, mask=mask)

        # torch.Size([64, 1600, 32])
        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        # torch.Size([64, 1600, 64])
        deconv3 = self.decoderlayer_3(deconv3, mask=mask)

        # Output Projection
        y = self.output_proj(deconv3)

        return x + y if self.in_chans == 3 else y