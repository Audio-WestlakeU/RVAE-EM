# RVAE-EM

Official PyTorch implementation of "**RVAE-EM: Generative speech dereverberation based on recurrent variational auto-encoder and convolutive transfer function**" which has been submitted to ICASSP 2024.

[Paper](blah.com) | [Code](https://github.com/Audio-WestlakeU/RVAE-EM) | [DEMO](blah.com)

## 1. Introduction

As a speech dereverberation algorithm, RVAE-EM has two versions, RVAE-EM-U (unsupervised) and RVAE-EM-S (supervised).

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
The directory of paired RIRs should have two subdirectories `noisy/` and `clean/`, with the same filenames present in both.

For testing, you should prepare directory of reverberant recordings with `.wav` files.

We provide tools for simulating RIRs and generating testset, see `prepare_data/gen_rirs.py` and `prepare_data/gen_testset.py`.

### 2.3 Train proposed RVAE-EM-U (unsupervised)

Unsupervised training with GPUs:
```
# GPU setting
export CUDA_VISIBLE_DEVICES=0,1 # for 2 gpus
export CUDA_VISIBLE_DEVICES=0, # for 1 gpu

# start a new training process or resume training (if possible)
python train_u.py -c [config.json] -p [save_path]

# use pretrained model parameters
python train_u.py -c [config.json] -p [save_path] --start_ckpt [pretrained_checkpoint]
```

### 2.4 Train proposed RVAE-EM-S (supervised)

Supervised training with GPUs:
```
# GPU setting
export CUDA_VISIBLE_DEVICES=0,1 # for 2 gpus
export CUDA_VISIBLE_DEVICES=0, # for 1 gpu

# start a new training process
python train_s.py -c [config.json] -p [save_path] --start_ckpt [checkpoint_from_unsupervised_training]

# resume training
python train_s.py -c [config.json] -p [save_path]

# use pretrained model parameters
python train_s.py -c [config.json] -p [save_path] --start_ckpt [pretrained_checkpoint]
```





### 2.5 Test & evaluate
## 3. Performance
## 4. Citation

If you fine the code is helpful, please site the following article.
```

```

## 5. Reference
