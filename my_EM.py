"""
实现自己算法中的采样与EM，实现去混响
"""
import torch
import torch.nn.functional
from torch import vmap


class MyEM:
    def __init__(self, model, EM_kparams, chunk_size=None):
        """
        Class: EM algorithm
        Params:
            model: RVAE network
            EM kparams: parameter dict
                CTF_len: CTF filter length
                max_steps: max iterations
                noi_pow_scale: initial Gaussian noise power
            chuck_size: chunk size for torch.vmap. Try smaller parameter when facing memory issues.
        """
        super().__init__()

        self.model = model
        del model
        self.noi_pow_scale = EM_kparams["noi_pow_scale"]
        self.CTF_P = EM_kparams["CTF_len"]
        self.max_steps = EM_kparams["max_steps"]
        self.update_speech_post_func = vmap(
            self.update_speech_post, chunk_size=chunk_size
        )
        self.update_CTF_func = vmap(self.update_CTF, chunk_size=chunk_size)
        self.update_noise_var_func = vmap(self.update_noise_var, chunk_size=chunk_size)

    def EM(self, Observed_data):
        """
        Func: EM algorithm
        """
        [self.bs, self.F, self.T] = Observed_data.shape
        self.device = Observed_data.device
        self.model = self.model.to(self.device)

        # Initialize the CTF filter and noise variance
        CTF, Noi_var = self.init_CTF_noi_var(
            Observed_data
        )  # [bs,F,P+1] complex,[bs,F] real
        Noi_var_reshape = Noi_var.reshape([-1])  # [bs*F], real
        Observed_data_reshape = Observed_data.reshape([-1, self.T])  # [bs*F,T], complex
        CTF_reshape = CTF.reshape([-1, self.CTF_P + 1])  # [bs*F,P+1], complex

        # 初始化隐变量
        Speech_prior_var, _, _, _ = self.model(Observed_data.abs().permute(0, 2, 1))
        Speech_prior_var = Speech_prior_var.permute(0, 2, 1).pow(2)
        Speech_prior_var_reshape = Speech_prior_var.reshape(
            [-1, self.T]
        )  # [bs*F,T], real

        for _ in range(self.max_steps):

            # E step: get the posterior of clean speech
            (
                Speech_post_mean_reshape,
                Speech_post_var_reshape,
            ) = self.update_speech_post_func(
                Observed_data_reshape,
                CTF_reshape,
                Speech_prior_var_reshape,
                Noi_var_reshape,
            )  # [bs*F,T] complex, [bs*F,T,T] real

            # 更新CTF参数
            CTF_reshape = self.update_CTF_func(
                Observed_data_reshape, Speech_post_mean_reshape, Speech_post_var_reshape
            )
            # 更新Sigma_noi参数
            Noi_var_reshape = self.update_noise_var_func(
                Observed_data_reshape,
                CTF_reshape,
                Speech_post_mean_reshape,
                Speech_post_var_reshape,
            )

        return Speech_post_mean_reshape.reshape([self.bs, self.F, self.T])

    def init_CTF_noi_var(self, Observed_data):
        """
        Func: Initialize CTF filter and noise variance
        Params:
            Observed_data: observation, shape of [bs, F, T]
        """
        bs, F, T = Observed_data.shape
        device = self.device

        Y_power = Observed_data.abs().pow(2).mean(2)  # [bs,F]
        Sigma_noi = Y_power * self.noi_pow_scale  # [bs,F]

        H_init = torch.zeros([bs, F, self.CTF_P + 1]) + 0j
        H_init[:, :, 0] = H_init[:, :, 0] + 1

        return H_init.to(device), Sigma_noi.to(device)  # [bs,F,P+1],[bs,F]

    def update_speech_post(self, Observed_data, CTF_filter, speech_var, noise_var):
        """
        Func: E-step, update posterior of clean speech
        Params:
            Observed_data: reverberant recordings, shape of [bs, F, T]
            CTF_filter: CTF filter, shape of [bs, F, P+1]
            speech_var: prior variance of clean speech
            noise_var: noise variance
        """

        device = self.device
        T = Observed_data.shape[0]
        P = self.CTF_P

        H_tilde = torch.zeros([T, T])  # [T,T]
        H_tilde = H_tilde + 0j
        H_tilde = H_tilde.to(device)

        speech_var = speech_var + 0j

        for p in range(P + 1):  # [T,T]
            H_tilde = H_tilde + CTF_filter[p] * torch.diag_embed(
                torch.ones_like(H_tilde.diag(-p)), -p
            )

        UBVA = noise_var.pow(-1) * torch.mm(
            torch.mm(H_tilde.H, H_tilde), speech_var.diag_embed(0)
        )

        Sigma_f = speech_var.diag_embed(0) - torch.mm(
            torch.mm(speech_var.diag_embed(0), UBVA),
            ((torch.eye(UBVA.shape[0], device=UBVA.device) + UBVA).inverse()),
        )

        Mu_f = noise_var.pow(-1) * torch.mm(
            torch.mm(Sigma_f, H_tilde.H), Observed_data.unsqueeze(1)
        ).squeeze(
            1
        )  # [T]

        return Mu_f, Sigma_f  # [T] complex, [T,T] complex

    def update_CTF(self, Observed_data, Post_mean, Post_var):
        """
        Func: M-step, update CTF filter, shape of [P+1], complex
        Params:
            Observed_data: observation in single band [T], complex
            Post_mean: posterior mean of clean speech in single band [T], complex
            Post_var: posterior variance of clean speech in single band [T,T], complex
        """
        numerator = (
            torch.zeros([1, self.CTF_P + 1]).to(self.device) + 0j
        )  # [1,P+1], complex
        denominator = (
            torch.zeros([self.CTF_P + 1, self.CTF_P + 1]).to(self.device) + 0j
        )  # [P+1,P+1], complex
        for t in range(self.CTF_P, self.T):
            Mu_t = (
                Post_mean[t - self.CTF_P : t + 1].flip(0).unsqueeze(0)
            )  # [1,P+1], complex
            Sigma_t = (
                Post_var[t - self.CTF_P : t + 1, t - self.CTF_P : t + 1].flip(0).flip(1)
            )  # [P+1,P+1], complex

            numerator = numerator + Observed_data[t] * Mu_t.conj()
            denominator = denominator + torch.mm(Mu_t.T, Mu_t.conj()) + Sigma_t

        H = torch.mm(numerator, denominator.inverse()).squeeze()  # [P+1], complex

        return H

    def update_noise_var(self, Observed_data, CTF_filter, Post_mean, Post_var):
        """
        Func: M-step, update noise variance in a single band
        Params:
            Observed_data: observation in a single band, [T] complex
            CTF_filter: CTF filter, [P+1] complex
            Post_mean: posterior mean of clean speech, [T] complex
            Post_var: posterior variance of clean speech, [T,T] complex
        """
        T = Observed_data.shape[0]
        H_tilde = torch.zeros([T, T]) + 0j  # [T,T] complex
        H_tilde = H_tilde.to(self.device)

        for p in range(self.CTF_P + 1):
            H_tilde = H_tilde + CTF_filter[p] * torch.diag_embed(
                torch.ones_like(H_tilde.diag(-p)), -p
            )

        H_tilde_times_H_tilde = torch.mm(H_tilde.H, H_tilde)  # [T,T] complex

        numerator = torch.norm(
            Observed_data.unsqueeze(1) - torch.mm(H_tilde, Post_mean.unsqueeze(1)),
            "fro",
        ).pow(2)

        numerator = (
            numerator
            + torch.trace(torch.mm(torch.mm(H_tilde, Post_var), H_tilde.H)).real
        )

        return numerator / T + 1e-8
