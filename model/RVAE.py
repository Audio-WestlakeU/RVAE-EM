import torch
import torch.nn as nn
from .module import (
    build_conv2d,
    ResBlock2D,
    build_GRU,
    build_MLP,
    reparametrization,
)


class RVAE(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        gru_dim_x_enc: int,
        gru_dim_z_enc: int,
        gru_dim_x_dec: int,
        pre_conv_enc: list,
        pre_conv_dec: list,
        resblock_enc: list,
        resblock_dec: list,
        post_conv_enc: list,
        post_conv_dec: list,
        num_resblock: int,
        num_GRU_layer_enc: int,
        num_GRU_layer_dec: int,
        dense_zmean_zlogvar: list,
        dense_activation_type: str,
        dropout_p: float,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.net_pre_conv_enc = build_conv2d(pre_conv_enc, "Conv_pre_enc", False)
        self.net_conv_enc = ResBlock2D(
            resblock_enc, batch_norm, num_resblock, dropout_p
        )
        self.net_post_conv_enc = build_conv2d(post_conv_enc, "Conv_post_enc", False)
        self.gru_x_enc = build_GRU(dim_x, gru_dim_x_enc, num_GRU_layer_enc, True)
        self.gru_z_enc = build_GRU(dim_z, gru_dim_z_enc, num_GRU_layer_enc, False)
        dim_h_enc = gru_dim_x_enc * 2 + gru_dim_z_enc
        assert dense_zmean_zlogvar[-1] == dim_z
        self.mlp_zmean_enc, _ = build_MLP(
            dim_h_enc, dense_zmean_zlogvar, dense_activation_type, dropout_p
        )
        self.mlp_zlogvar_enc, _ = build_MLP(
            dim_h_enc, dense_zmean_zlogvar, dense_activation_type, dropout_p
        )
        self.gru_x_dec = build_GRU(dim_z, gru_dim_x_dec, num_GRU_layer_dec, True)
        self.net_pre_conv_dec = build_conv2d(pre_conv_dec, "Conv_pre_dec", False)
        self.net_conv_dec = ResBlock2D(
            resblock_dec, batch_norm, num_resblock, dropout_p
        )
        self.net_post_conv_dec = build_conv2d(post_conv_dec, "Conv_post_dec", False)
        self.num_GRU_layer_enc = num_GRU_layer_enc
        self.num_GRU_layer_dec = num_GRU_layer_dec
        self.dim_hz_enc = gru_dim_z_enc
        self.dim_z = dim_z

    def encoder(self, x):

        device = x.device
        x = (x + 1e-8).log()
        bs, seq_len, dim_feature = x.shape

        z = torch.zeros([bs, seq_len, self.dim_z]).to(device)
        z_t = torch.zeros([bs, self.dim_z]).to(device)
        h_hz_t_enc = torch.zeros([self.num_GRU_layer_enc, bs, self.dim_hz_enc]).to(
            device
        )

        zmean = torch.zeros([bs, seq_len, self.dim_z]).to(device)
        zlogvar = torch.zeros([bs, seq_len, self.dim_z]).to(device)

        # x的部分
        x = x.unsqueeze(1)
        x_temp = self.net_pre_conv_enc(x)
        x_temp = self.net_conv_enc(x_temp)
        x_temp = self.net_post_conv_enc(x_temp)
        hx_in_enc = x_temp.squeeze(1)
        hx_enc, _ = self.gru_x_enc(torch.flip(hx_in_enc, [1]))
        hx_enc = torch.flip(hx_enc, [1])

        # z和h_enc的部分
        for t in range(seq_len):
            hz_t_in_enc = z_t.unsqueeze(1)
            hz_t_enc, h_hz_t_enc = self.gru_z_enc(hz_t_in_enc, h_hz_t_enc)
            hz_t_enc = hz_t_enc.squeeze(1)
            h_t_enc = torch.cat([hx_enc[:, t, :], hz_t_enc], -1)
            zmean_t = self.mlp_zmean_enc(h_t_enc)
            zlogvar_t = self.mlp_zlogvar_enc(h_t_enc)
            z_t = reparametrization(zmean_t, zlogvar_t)
            z[:, t, :] = z_t
            zmean[:, t, :] = zmean_t
            zlogvar[:, t, :] = zlogvar_t

        return z, zmean, zlogvar

    def decoder(self, z):
        [bs, seq_len, _] = z.shape
        h_dec, _ = self.gru_x_dec(z)
        h_dec = h_dec.reshape([bs, seq_len, 2, -1])
        h_dec = h_dec.permute(0, 2, 1, 3)

        h_dec = self.net_pre_conv_dec(h_dec)
        h_dec = self.net_conv_dec(h_dec)
        h_dec = self.net_post_conv_dec(h_dec)
        logx = h_dec.squeeze(1)
        x = logx.exp()
        return x

    def forward(self, x):

        z, zmean, zlogvar = self.encoder(x)
        x_reconstruct = self.decoder(z)
        return x_reconstruct, zmean, zlogvar, z
