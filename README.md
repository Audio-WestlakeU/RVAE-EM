# RVAE-EM

Official PyTorch implementation of "**RVAE-EM: Generative speech dereverberation based on recurrent variational auto-encoder and convolutive transfer function**" which has been accepted by ICASSP 2024.

[Paper](https://arxiv.org/abs/2309.08157) | [Code](https://github.com/Audio-WestlakeU/RVAE-EM) | [DEMO](https://github.com/Audio-WestlakeU/RVAE-EM) 

## 1. Introduction

RVAE is a speech dereverberation algorithm.
It has two versions, RVAE-EM-U (unsupervised trained) and RVAE-EM-S (supervised trained).

The overview of RVAE-EM is

<div align="center">
<image src="/figures/overview.png"  width="500" alt="Overview of RVAE-EM" />
</div>

## 2. Get started
### 2.1 Requirements

See `requirements.txt`.

### 2.2 Prepare datasets

For training and validating, you should prepare directories of 
 - clean speech (from WSJ0 corpus in our experiments)
 - reverberant-dry-paired RIRs (simulated with gpuRIR toolbox in our experiments)

with `.wav` files.
The directory of paired RIRs should have two subdirectories `noisy/` and `target/`, with the same filenames present in both.

For testing, you should prepare directory of reverberant recordings with `.wav` files.

We provide tools for simulating RIRs and generating testset, try 
```
python prepare_data/gen_rirs.py -c config/gen_trainset.yaml
python prepare_data/gen_rirs.py -c config/gen_testset.yaml
python prepare_data/gen_testset.py -c config/gen_trainset.yaml
```

### 2.3 Train proposed RVAE-EM-U (unsupervised training)

Unsupervised training with multiple GPUs:
```
# GPU setting
export CUDA_VISIBLE_DEVICES=[GPU_ids]

# start a new training process or resume training (if possible)
python train_u.py -c [config_U.json] -p [save_path]

# use pretrained model parameters
python train_u.py -c [config_U.json] -p [save_path] --start_ckpt [pretrained_checkpoint]
```
Fixed learning rate of 1e-4 is recommended. 
You can also use smaller learning rate at the end of training for better performance.

### 2.4 Train proposed RVAE-EM-S (supervised fine-tuning)

Supervised training with multiple GPUs:
```
# GPU setting
export CUDA_VISIBLE_DEVICES=[GPU_ids]

# start a new training process
python train_s.py -c [config_S.json] -p [save_path] --start_ckpt [checkpoint_from_unsupervised_training]

# resume training
python train_s.py -c [config_S.json] -p [save_path]

# use pretrained model parameters
python train_s.py -c [config_S.json] -p [save_path] --start_ckpt [pretrained_checkpoint]
```
Fixed learning rate of 1e-5 is recommended. 
You can also use smaller learning rate at the end of training for better performance.




### 2.5 Test & evaluate
Both RVAE-EM-U and RVAE-EM-S use the same command to test and evaluate:
```
# GPU setting
export CUDA_VISIBLE_DEVICES=[GPU_ids]

# test
python enhance.py -c [config.json] -p [save_path] --ckpt [checkpoint_path]

# evaluate (SISDR, PESQ, STOI)
python eval.py -i [input .wav folder] -o [output .wav folder] -r [reference .wav folder]

# evaluate (DNSMOS)
python DNSMOS/dnsmos_local.py -t [output .wav folder]
```

If you are facing memory issues, try smaller `batch_size` or smaller `chunk_size` in class `MyEM`.

## 3. Performance

The performance reported in our paper is
<div align="center">
<image src="/figures/performance.jpg"  width="900" alt="Performance table" />
</div>

Our RVAE network should be trained sufficiently.
In this table, we train and finetune networks in RVAE-EM-U and RVAE-EM-S for over 180k steps and 80k steps, respectively.
<!-- We also provide the checkpoints of RVAE-EM-U and RVAE-EM-S in `/logs/ckpt`.
Different from the reported version, both the checkpoints are trained/finetuned for about 200k steps. -->

The typical validation total/IS/KL loss curves of unsupervised training are
<div align="center">
<image src="/figures/RVAE-U_val_loss.svg"  width="200" alt="Total loss" />
<image src="/figures/RVAE-U_val_loss_IS.svg"  width="200" alt="IS loss" />
<image src="/figures/RVAE-U_val_loss_KL.svg"  width="200" alt="KL loss" />
</div>

The typical validation total/IS/KL loss curves of supervised finetuning are
<div align="center">
<image src="/figures/RVAE-S_val_loss.svg"  width="200" alt="Total loss" />
<image src="/figures/RVAE-S_val_loss_IS.svg"  width="200" alt="IS loss" />
<image src="/figures/RVAE-S_val_loss_KL.svg"  width="200" alt="KL loss" />
</div>


## 4. Citation

If you find our work helpful, please cite
```
@misc{wang2023rvaeem,
      title={RVAE-EM: Generative speech dereverberation based on recurrent variational auto-encoder and convolutive transfer function}, 
      author={Pengyu Wang and Xiaofei Li},
      year={2023},
      eprint={2309.08157},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
