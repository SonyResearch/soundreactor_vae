# Continous audio VAEs codebase

Audio VAEs from [SoundCTM](https://openreview.net/forum?id=KrK6zXbjfO) and [SoundReactor](https://koichi-saito-sony.github.io/soundreactor/).

We have two variants of 48kHz stereo VAEs (noncausal and causal) from SoundReactor and one 44.1kHz mono VAE (noncausal) from SoundCTM.

All the models are trained with only [Audioset](https://research.google.com/audioset/) and [WavCaps](https://arxiv.org/pdf/2303.17395), so models might not work well on speech, music, and studio-level recordings of SFX.
We have a plan to include those data for training in the future.

## Installation

We have only tested this on Ubuntu.

## Prerequisites

Install docker and build docker container.

```sh
docker build -t tag .
```

Or you can install with pip (Other versions might work but we only tested on this setup.)

- Python 3.9+
- PyTorch 2.5.1+ and corresponding torchvision/torchaudio (pick your CUDA version https://pytorch.org/)


```bash
cd continous_waveform_vae
pip install -e .
```

## Run inference

```bash
bash autoencoder_reconstruction.sh
```

### Arguments
- --input_file: Path to a single audio or video file (supported extensions: .wav, .flac, .mp3, .mp4, .mkv, etc.).

- --input_dir: Path to a directory containing audio/video files. All supported files inside will be processed.

- --csv_file: Path to a CSV file. Note: The CSV must contain a column named 'filepath' with absolute paths to the audio files.

- --output_dir: Root directory to save the reconstructed audio files.

- --output_file: (Optional, Single-file mode only) Specific path to save the output file. If not set, it defaults to the input filename with _recon.flac suffix.

##### Model Configuration
- --model_variant: (Required) The VAE model architecture to use.
    - sr_noncausal: SoundReactor's non-causal VAE. (48kHz stereo)
        - Temporal downsampling rate: 1600, Latent channel: 64 dim.
    - sr_causal: SoundReactor's causal VAE.  (48kHz stereo)
        - Temporal downsampling rate: 1600, Latent channel: 64 dim.
    - soundctm_vae: SoundCTM non-casual VAE.  (44.1kHz mono)
        - Temporal downsampling rate: 1024, Latent channel: 64 dim.

- --config: Path to the model JSON config file. Required for sr_noncausal and sr_causal variants.
    - sr_noncausal: modules/stable_audio_tools/configs/model_configs/autoencoders/sr_ds_1600_dim_64.json
    - sr_causal: modules_causal/stable_audio_tools/configs/model_configs/autoencoders/sr_ds_1600_dim_64_causal.json

- --ckpt: Path to the model weights.
    - sr_noncausal: path to "sr_noncausal_ds_1600_dim_64.ckpt". (Please download checkpoint from [google drive](https://drive.google.com/drive/folders/1rlROnePyQU4b8GzjkB07W5vlb297p5_c?usp=sharing))
    - sr_causal: path to "sr_causal_ds_1600_dim_64.ckpt". (Please download checkpoint from [google drive](https://drive.google.com/drive/folders/1rlROnePyQU4b8GzjkB07W5vlb297p5_c?usp=sharing))
    - soundctm_vae: dir to store downloaded checkpoint. (Checkpoint will be automatically downloaded from [huggingface](https://huggingface.co/koichisaito/soundctm_dit).)

##### Misc.
- --batch_size: Number of files to process simultaneously (Default: 1).

- --dtype: Precision for inference. Choices: float32, float16, bfloat16 (Default: float32).

- --duration: Duration in seconds to process from the start of the audio. Set to -1.0 to process the full audio length (Default: -1.0).


## Citation

```bibtex
@inproceedings{saito2025soundctm,
  title={Sound{CTM}: Unifying Score-based and Consistency Models for Full-band Text-to-Sound Generation},
  author={Koichi Saito and Dongjun Kim and Takashi Shibuya and Chieh-Hsin Lai and Zhi Zhong and Yuhta Takida and Yuki Mitsufuji},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=KrK6zXbjfO}
}
```

```bibtex
@article{saito2025soundreactor,
  title={SoundReactor: Frame-level Online Video-to-Audio Generation},
  author={Koichi Saito and Julian Tanke and Christian Simon and Masato Ishii and Kazuki Shimada and Zachary Novack and Zhi Zhong and Akio Hayakawa and Takashi Shibuya and Yuki Mitsufuji},
  year={2025},
  eprint={2510.02110},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2510.02110}, 
  journal={arXiv preprint arXiv:2510.02110},
}
```

