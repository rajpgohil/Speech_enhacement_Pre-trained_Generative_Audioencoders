# Speech_enhacement_Pre-trained_Generative_Audioencoders

# 🎙️ Speech Enhancement with WavLM + BiLSTM Denoiser + HiFi-GAN

This repository implements a **speech enhancement system** inspired by [Efficient Speech Enhancement via Embeddings from Pre-trained Generative Audioencoders (Sun et al., 2025)](https://arxiv.org/pdf/2506.11514).  

We use:

- **WavLM-Large** (frozen) as the audio encoder (1024-dim embeddings).  
- A **Projection + BiLSTM Denoiser** to map noisy embeddings → clean embeddings (512-dim).  
- **HiFi-GAN** vocoder (trained on embeddings) to reconstruct clean waveforms.  
- **Realistic augmentations** (MP3, AAC, EnCodec, μ-law, A-law, vocoder artifacts) instead of simple noise mixing.  

---

## 📂 Project Structure

```text
├── dataloader_libri.py              # Dataset with codec/vocoder augmentations
├── train.py                # Training script for denoiser
├── inference.py            # Inference script (denoiser + HiFi-GAN)
├── knn_vc/
│   └── hifigan/            # HiFi-GAN implementation (from bshall/knn-vc) Please clone from https://github.com/bshall/knn-vc.git
├── requirements.txt
└── README.md
└── download the models from https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt
```

---

## 🔧 Installation

```bash
git clone https://github.com/yourusername/Speech_enhacement_Pre-trained_Generative_Audioencoders.git
cd Speech_enhacement_Pre-trained_Generative_Audioencoders
pip install torch torchaudio transformers encodec
sudo apt-get install ffmpeg
```

## 📊 Datasets

- LibriSpeech train-clean-5 (clean speech)
- Augmentations applied on-the-fly to simulate noisy/degraded speech:
	- Low-bitrate MP3 re-encoding
	- Low-bitrate AAC re-encoding
	- Low-codebook EnCodec re-encoding
	- Small HiFi-GAN re-encoding (or simulated lowpass filter)
	- μ-law companding
	- A-law companding
This produces (degraded_audio, clean_audio) pairs for training.

## 🚀 Training

Train the Projection + BiLSTM Denoiser:
```text
python3 train.py
```
## 🎧 Inference
Enhance a noisy/degraded audio file:
```text
python3 inference.py \
  --input_wav noisy_example.wav \
  --output_wav enhanced_example.wav \
  --denoiser_ckpt best_denoiser.pth \
  --hifigan_ckpt g_02500000.pt \
  --hifigan_config config_v1_wavlm.json
```

Pipeline:
```text
Noisy waveform → WavLM (1024) → Projection+BiLSTM (512) → HiFi-GAN (512) → Enhanced waveform
```

## 🧩 Augmentations Explained

- MP3 (low bitrate): simulates streaming/compression artifacts.
- AAC (low bitrate): common in YouTube/streaming audio.
- EnCodec (low codebook): simulates neural codec artifacts.
- Small HiFi-GAN: simulates vocoder artifacts.
- μ-law / A-law: simulates telephone-quality speech.
These augmentations make the model robust to real-world degradations, not just additive noise.

## 📈 Future Work

- Add support for Whisper encoder (512-dim) as an alternative to WavLM.
- Add multi-task training (denoising + dereverberation).
