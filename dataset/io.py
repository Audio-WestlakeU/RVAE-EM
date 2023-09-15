import torch
import torchaudio
import numpy as np


def stft(audio, nfft, hop, win):
    """
    输入numpy.array音频，返回torch.tensor复数谱
    """
    if win.upper() == "HANN":
        window = torch.hann_window(nfft).to(audio.device)
    elif win.upper() == "HAMMING":
        window = torch.hamming_window(nfft).to(audio.device)

    spch_sptm = torch.stft(
        audio,
        n_fft=nfft,
        hop_length=hop,
        onesided=True,
        window=window,
        return_complex=True,
    )
    return spch_sptm


def istft(sptm, nfft, hop, win):
    """
    输入torch.tensor复数谱，返回numpy.array音频
    """
    if win.upper() == "HANN":
        window = torch.hann_window(nfft).to(sptm.device)
    elif win.upper() == "HAMMING":
        window = torch.hamming_window(nfft).to(sptm.device)

    audio = torch.istft(sptm, n_fft=nfft, hop_length=hop, onesided=True, window=window)

    return audio.to("cpu")


def sptm2sptm_preprocess(x):
    """
    输入复数频谱，返回实数功率谱，并调换维度
    """
    return x.abs().pow(2).permute(0, 2, 1)


def sptm2sptm_postprocess(x):
    """
    输入实数功率谱，返回实数幅度谱，并调换维度
    """
    return x.sqrt().permute(0, 2, 1)


def sptm2mel_preprocess(x):
    """
    输入复数频谱，返回实数功率谱，并调换维度
    """
    x = x.abs().pow(2)
    device = x.device
    n_stft = x.shape[1]
    mel_trans = torchaudio.transforms.MelScale(
        n_mels=64, sample_rate=16000, n_stft=n_stft
    )
    data_mel = mel_trans(x.to("cpu")).to(device)

    return data_mel.permute(0, 2, 1)


def audio2input_woDC_preprocess(audio, nfft, hop, win, *args, **kwargs):
    sptm = stft(
        audio,
        nfft=nfft,
        hop=hop,
        win=win,
    )
    sptm = sptm[:, 1:, :]
    sptm = sptm2sptm_preprocess(sptm)
    return sptm

def aud2sptm_woDC(audio, nfft, hop, win, *args, **kwargs):
    '''
    audio to magnitude spectrogram
    audio: [bs,samples]
    nfft: int,
    hop: int,
    '''
    sptm = stft(
        audio,
        nfft=nfft,
        hop=hop,
        win=win,
    )
    sptm = sptm[:, 1:, :].permute(0,2,1) 
    return sptm.abs(), torch.angle(sptm) # [bs, T, F]

def sptm_woDC2aud(mag_woDC,phase_woDC,nfft,hop,win,*args, **kwargs):
    mag_woDC = mag_woDC.permute(0,2,1) # [bs,F,T]
    phase_woDC = phase_woDC.permute(0,2,1)
    bs,F,T = mag_woDC.shape
    sptm_woDC = mag_woDC * torch.exp(1j*phase_woDC)
    F = F+1
    sptm = torch.zeros([bs,F,T]).to(sptm_woDC.device)+0j
    sptm[:,1:,:]=sptm_woDC
    wav = istft(sptm,nfft=nfft,hop=hop,win=win)
    return wav # [bs,sample]
    

def audio2EMinput_woDC_preprocess(audio, nfft, hop, win, *args, **kwargs):
    sptm = stft(
        audio,
        nfft=nfft,
        hop=hop,
        win=win,
    )
    sptm_wo_DC = sptm[:, 1:, :]
    return sptm_wo_DC


def EMoutput_woDC2audio_postprocess(sptm_wo_DC, nfft, hop, win, *args, **kwargs):

    bs, F, T = sptm_wo_DC.shape
    F = F + 1
    sptm_w_DC = torch.zeros([bs, F, T]).to(sptm_wo_DC.device) + 0j
    sptm_w_DC[:, 1:, :] = sptm_wo_DC

    wav = istft(sptm_w_DC, nfft=nfft, hop=hop, win=win)
    return wav


def recon_audio2input_woDC_preprocess(audio, nfft, hop, win, *args, **kwargs):
    sptm = stft(
        audio,
        nfft=nfft,
        hop=hop,
        win=win,
    )
    sptm = sptm[:, 1:, :]
    phase = torch.angle(sptm)
    sptm = sptm2sptm_preprocess(sptm)
    return sptm, phase


def recon_output_woDC2audio_postprocess(
    power_sptm_wo_DC, phase, nfft, hop, win, *args, **kwargs
):

    abs_sptm_wo_DC = power_sptm_wo_DC.sqrt().permute(0, 2, 1)
    bs, F, T = abs_sptm_wo_DC.shape
    sptm_wo_DC = abs_sptm_wo_DC * torch.exp(1j * phase)

    F = F + 1
    sptm_w_DC = torch.zeros([bs, F, T]).to(abs_sptm_wo_DC.device) + 0j
    sptm_w_DC[:, 1:, :] = sptm_wo_DC
    wav = istft(sptm_w_DC, nfft=nfft, hop=hop, win=win)
    return wav
