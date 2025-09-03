# import torch
# import torchaudio
# import torch.nn as nn
# from transformers import WavLMModel, Wav2Vec2FeatureExtractor
# import sys, json

# # -----------------------------
# # 1. Load WavLM-Large (frozen)
# # -----------------------------
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
# wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().cuda()
# for p in wavlm.parameters():
#     p.requires_grad = False

# def get_embeddings(waveform, sr=16000):
#     if waveform.dim() == 1:
#         waveform = waveform.unsqueeze(0)  # [1, T]
#     inputs = feature_extractor(
#         waveform.cpu().numpy(),
#         sampling_rate=sr,
#         return_tensors="pt",
#         padding=True
#     )
#     with torch.no_grad():
#         outputs = wavlm(**{k: v.cuda() for k, v in inputs.items()})
#     return outputs.last_hidden_state  # [B, T, 1024]


# # -----------------------------
# # 2. Projection + BiLSTM Denoiser
# # -----------------------------
# class ProjectionBiLSTMDenoiser(nn.Module):
#     def __init__(self, input_dim=1024, proj_dim=512, hidden_dim=256, output_dim=512, num_layers=3):
#         super().__init__()
#         # Projection layer (1024 → 512)
#         self.proj_in = nn.Linear(input_dim, proj_dim)

#         # BiLSTM denoiser
#         self.lstm = nn.LSTM(
#             proj_dim, hidden_dim, num_layers=num_layers,
#             batch_first=True, bidirectional=True
#         )
#         self.proj_out = nn.Linear(hidden_dim * 2, output_dim)

#     def forward(self, x):
#         # x: [B, T, 1024]
#         x = self.proj_in(x)          # [B, T, 512]
#         out, _ = self.lstm(x)        # [B, T, 2*hidden]
#         out = self.proj_out(out)     # [B, T, 512]
#         return out


# # -----------------------------
# # 3. Load HiFi-GAN (expects 512-dim input)
# # -----------------------------
# sys.path.append("knn-vc/hifigan")  # adjust path if needed
# from models import Generator
# from utils import AttrDict

# def load_hifigan(checkpoint_path, config_path):
#     with open(config_path) as f:
#         h = AttrDict(json.load(f))   # convert dict → AttrDict
#     hifigan = Generator(h).cuda()
#     state = torch.load(checkpoint_path, map_location="cuda")
#     hifigan.load_state_dict(state["generator"])
#     hifigan.eval()
#     return hifigan


# # -----------------------------
# # 4. Inference Pipeline
# # -----------------------------
# def enhance_audio(input_wav, output_wav,
#                   denoiser_ckpt="/data/raj/cabm_ass/best_denoiser.pth",
#                   hifigan_ckpt="/data/raj/cabm_ass/g_02500000.pt",
#                   hifigan_config="/data/raj/cabm_ass/knn-vc/hifigan/config_v1_wavlm.json",
#                   sr=16000):

#     # Load noisy audio
#     noisy, sr_in = torchaudio.load(input_wav)
#     if sr_in != sr:
#         noisy = torchaudio.functional.resample(noisy, sr_in, sr)

#     # Load denoiser
#     denoiser = ProjectionBiLSTMDenoiser().cuda()
#     checkpoint = torch.load(denoiser_ckpt, map_location="cuda")
#     denoiser.load_state_dict(checkpoint["model_state"])
#     denoiser.eval()

#     # Load HiFi-GAN
#     hifigan = load_hifigan(hifigan_ckpt, hifigan_config)

#     # Extract embeddings
#     noisy_emb = get_embeddings(noisy.cuda(), sr=sr)  # [1, T, 1024]

#     # Denoise + Vocoder
#     with torch.no_grad():
#         enhanced_emb = denoiser(noisy_emb)  # [1, T, 512]
#         emb = enhanced_emb.transpose(1, 2)  # [B, 512, T]
#         audio = hifigan(emb).squeeze().cpu()

#     # Save enhanced audio
#     torchaudio.save(output_wav, audio.unsqueeze(0), sr)
#     print(f"✅ Saved enhanced audio to {output_wav}")


# # -----------------------------
# # 5. Run Example
# # -----------------------------
# if __name__ == "__main__":
#     enhance_audio(
#         input_wav="/data/raj/cabm_ass/noisy_example.wav",
#         output_wav="enhanced_example.wav",
#         denoiser_ckpt="/data/raj/cabm_ass/best_denoiser.pth",
#         hifigan_ckpt="/data/raj/cabm_ass/prematch_g_02500000.pt",
#         hifigan_config="/data/raj/cabm_ass/knn-vc/hifigan/config_v1_wavlm.json"
#     )




# ==================================================================
# ==================================================================
# ==================================================================


import torch
import torchaudio
import torch.nn as nn
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

# -----------------------------
# 1. Load WavLM-Large (frozen)
# -----------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().cuda()
for p in wavlm.parameters():
    p.requires_grad = False

def get_embeddings(waveform, sr=16000):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]
    inputs = feature_extractor(
        waveform.cpu().numpy(),
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = wavlm(**{k: v.cuda() for k, v in inputs.items()})
    return outputs.last_hidden_state  # [B, T, 1024]


# -----------------------------
# 2. BiLSTM Denoiser
# -----------------------------
class BiLSTMDenoiser(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True
        )
        self.proj = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.proj(out)
        return out


# -----------------------------
# 3. Load HiFi-GAN (trained on 1024 embeddings)
# -----------------------------
import sys
sys.path.append("knn-vc/hifigan")  # adjust path if needed
from models import Generator
from utils import AttrDict
import json
# from knn_vc.hifigan.models import Generator

def load_hifigan(checkpoint_path, config_path):
    with open(config_path) as f:
        h = AttrDict(json.load(f))   # convert dict → AttrDict
    hifigan = Generator(h).cuda()
    state = torch.load(checkpoint_path, map_location="cuda")
    hifigan.load_state_dict(state["generator"])
    hifigan.eval()
    return hifigan


# -----------------------------
# 4. Inference Pipeline
# -----------------------------
def enhance_audio(input_wav, output_wav,
                  denoiser_ckpt="best_denoiser.pth",
                  hifigan_ckpt="/data/raj/cabm_ass/prematch_g_02500000.pt",
                  sr=16000):

    # Load noisy audio
    noisy, sr_in = torchaudio.load(input_wav)
    if sr_in != sr:
        noisy = torchaudio.functional.resample(noisy, sr_in, sr)

    # Load models
    denoiser = BiLSTMDenoiser().cuda()
    checkpoint = torch.load(denoiser_ckpt, map_location="cuda")
    denoiser.load_state_dict(checkpoint["model_state"])
    denoiser.eval()

    hifigan = load_hifigan(hifigan_ckpt,"/data/raj/cabm_ass/knn-vc/hifigan/config_v1_wavlm.json")

    # Extract embeddings
    noisy_emb = get_embeddings(noisy.cuda(), sr=sr)  # [1, T, 1024]

    # Denoise + Vocoder
    with torch.no_grad():
        enhanced_emb = denoiser(noisy_emb)  # [1, T, 1024]
        # HiFi-GAN expects [B, C, T] → transpose
        # emb = enhanced_emb.transpose(1, 2)  # [B, 1024, T]
        audio = hifigan(enhanced_emb).squeeze().cpu()

    # Save enhanced audio
    torchaudio.save(output_wav, audio.unsqueeze(0), sr)
    print(f"✅ Saved enhanced audio to {output_wav}")


# -----------------------------
# 5. Run Example
# -----------------------------
if __name__ == "__main__":
    enhance_audio(
        input_wav="noisy_example.wav",
        output_wav="enhanced_example.wav",
        denoiser_ckpt="best_denoiser.pth",
        hifigan_ckpt="/data/raj/cabm_ass/prematch_g_02500000.pt"
    )