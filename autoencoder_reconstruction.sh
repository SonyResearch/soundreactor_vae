#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python autoencoder_reconstruction.py \
    --model_variant soundctm_vae \
    --dtype float32 --batch_size 1 \
    --ckpt path/to/ckpt \
    --output_dir test/output/ \
    --config modules_causal/stable_audio_tools/configs/model_configs/autoencoders/sr_ds_1600_dim_64_causal.json --csv_file test/test.csv  \
    --input_file test/-0vPFx-wRRI.flac \
    # --csv_file test/test.csv  \

# --input_file
# --output_file
# --input_dir
# --output_dir
# --csv_file
# --batch_size 1
# --duration 8.0
# --model_variant choices=['sr_noncausal', 'sr_causal', 'soundctm_vae'])
# sr_noncausal_ds_1600_dim_64.ckpt
# sr_causal_ds_1600_dim_64.ckpt