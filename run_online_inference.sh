#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python run_online_inference.py \
    --dtype float32 --target_duration 15 \
    --ckpt /data2/share/generative_sound_engine/ckpt/stage1/exported_model_ds_1600_dim_64_causal_20250721.ckpt \
    --output_dir test/output/ \
    --input_file test/output/-0ewPjoBBNI_scene-13_sfx_latents.pt \
    --mode decode --use_cuda_graph

# --mode", type=str, default="reconstruction", choices=["reconstruction", "encode", "decode"]