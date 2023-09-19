import torch
import torchaudio
import numpy as np


def stft(audio, nfft, hop, win):
    """
    Func: STFT
    Params:
        nfft: nfft
        hop: hop size
        win: type of STFT analysis window, str, 'hann' or 'hamming'
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
    Func: iSTFT
    Params: same as stft
    """
    if win.upper() == "HANN":
        window = torch.hann_window(nfft).to(sptm.device)
    elif win.upper() == "HAMMING":
        window = torch.hamming_window(nfft).to(sptm.device)

    audio = torch.istft(sptm, n_fft=nfft, hop_length=hop, onesided=True, window=window)

    return audio.to("cpu")


def aud2sptm_woDC(audio, nfft, hop, win, *args, **kwargs):
    """
    Func: from audio to magnitude spectrogram without DC component
    Return: magnitude and phase spectrograms
    Params:
        audio: audio [bs,samples]
        others: same as stft
    """
    sptm = stft(
        audio,
        nfft=nfft,
        hop=hop,
        win=win,
    )
    sptm = sptm[:, 1:, :].permute(0, 2, 1)
    return sptm.abs(), torch.angle(sptm)  # [bs, T, F]


def sptm_woDC2aud(mag_woDC, phase_woDC, nfft, hop, win, *args, **kwargs):
    """
    Func: from magnitude spectrogram without DC component to audio
    Return: Audio
    Params:
        mag_woDC: magnitude spectrogram
        phase_woDC: phase spectrogram
        others: same as istft
    """
    mag_woDC = mag_woDC.permute(0, 2, 1)  # [bs,F,T]
    phase_woDC = phase_woDC.permute(0, 2, 1)
    bs, F, T = mag_woDC.shape
    sptm_woDC = mag_woDC * torch.exp(1j * phase_woDC)
    F = F + 1
    sptm = torch.zeros([bs, F, T]).to(sptm_woDC.device) + 0j
    sptm[:, 1:, :] = sptm_woDC
    wav = istft(sptm, nfft=nfft, hop=hop, win=win)
    return wav  # [bs,sample]


def audio2EMinput_woDC_preprocess(audio, nfft, hop, win, *args, **kwargs):
    """
    Func: from audio to RVAE-EM input
    Params:
        same as aud2sptm_woDC
    """
    sptm = stft(
        audio,
        nfft=nfft,
        hop=hop,
        win=win,
    )
    sptm_wo_DC = sptm[:, 1:, :]
    return sptm_wo_DC


def EMoutput_woDC2audio_postprocess(sptm_wo_DC, nfft, hop, win, *args, **kwargs):
    """
    Func: From EM output to audio
    Params:
        sptm_wo_DC: RVAE-EM output, complex spectrogram
        others: same as istft
    """
    bs, F, T = sptm_wo_DC.shape
    F = F + 1
    sptm_w_DC = torch.zeros([bs, F, T]).to(sptm_wo_DC.device) + 0j
    sptm_w_DC[:, 1:, :] = sptm_wo_DC

    wav = istft(sptm_w_DC, nfft=nfft, hop=hop, win=win)
    return wav
