import torch


class LossF:
    def loss_ISD(self, output: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
        """
        Return: IS loss
        Params:
            output: magnitude spectrogram (>=0), output of RVAE decoder, shape of [bs,T,F]
            target: magnitude spectrogram (>=0), clean, shape of [bs,T,F]
            eps: small constant, avoid numerical problems
        """
        bs, n_seq, _ = output.shape
        target = target**2 + eps
        output = output**2 + eps
        ret = torch.sum(target / output - torch.log(target / output) - 1)

        return ret / bs / n_seq

    def loss_KLD(self, zmean: torch.Tensor, zlogvar: torch.Tensor):
        """
        Return: KL loss
        Params:
            zmean: mean of latent variables, output of RVAE encoder, shape of [bs,T,D]
            zlogvar: log variance of latent variables, output of RVAE encoder, shape of [bs,T,D]
        """
        bs, n_seq, _ = zmean.shape
        zmean_p = torch.zeros_like(zmean)
        zlogvar_p = torch.zeros_like(zlogvar)
        ret = -0.5 * torch.sum(
            zlogvar
            - zlogvar_p
            - torch.div(zlogvar.exp() + (zmean - zmean_p).pow(2), zlogvar_p.exp())
            + 1
        )
        return ret / bs / n_seq

    def cal_KL_scale(
        self,
        cur_step: int,
        beta: float,
        zero_step: int,
        warmup_step: int,
        hold_step: int,
    ):
        """
        Return: the scale of KL loss
        Params:
            cur_step: current training step
            beta: base scale (default: 1)
            zero_step: keep scale = 0
            warmup_step: scale linear increased
            hold_step: keep scale=1
        """
        period = warmup_step + hold_step
        if cur_step < zero_step:
            return 0
        else:
            epoch_mod = (cur_step - zero_step) % period
            if epoch_mod < warmup_step:
                return beta * epoch_mod / warmup_step
            else:
                return beta
