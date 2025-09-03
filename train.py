# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from transformers import WavLMModel, Wav2Vec2FeatureExtractor
# import torch
# from dataloader_libri import SpeechEnhancementDataset  # dataset.py should contain the class

# # -----------------------------
# # 1. Load WavLM-Large (frozen)
# # -----------------------------
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
# wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().cuda()
# for p in wavlm.parameters():
#     p.requires_grad = False

# def get_embeddings(waveform, sr=16000):
#     """
#     waveform: torch.Tensor [B, T]
#     returns: torch.Tensor [B, T, 1024]
#     """
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
# # 2. BiLSTM Denoiser
# # -----------------------------
# # class BiLSTMDenoiser(nn.Module):
# #     def __init__(self, input_dim=1024, hidden_dim=512, output_dim=1024, num_layers=3):
# #         super().__init__()
# #         self.lstm = nn.LSTM(
# #             input_dim, hidden_dim, num_layers=num_layers,
# #             batch_first=True, bidirectional=True
# #         )
# #         self.proj = nn.Linear(hidden_dim * 2, output_dim)

# #     def forward(self, x):
# #         out, _ = self.lstm(x)  # [B, T, 2*hidden]
# #         out = self.proj(out)   # [B, T, output_dim]
# #         return out


# import torch
# import torch.nn as nn

# class ProjectionBiLSTMDenoiser(nn.Module):
#     def __init__(self, input_dim=1024, proj_dim=512, hidden_dim=256, output_dim=512, num_layers=3):
#         super().__init__()
#         # Trainable projection layer
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
# # 3. Training Loop
# # -----------------------------
# def train():
#     # Dataset
#     dataset = SpeechEnhancementDataset(
#         clean_dir="/data/raj/diarization/dataset/LibriSpeech/train-clean-5",
#         noise_dir="/data/raj/cabm_ass/libri-musan/data/local/musan_bgnoise",
#         sample_rate=16000,
#         segment_length=5.0
#     )

#     # Train/val split
#     val_size = int(0.05 * len(dataset))
#     train_size = len(dataset) - val_size
#     train_set, val_set = random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=2)

#     # Model, optimizer, loss
#     # denoiser = BiLSTMDenoiser().cuda()
#     denoiser = ProjectionBiLSTMDenoiser().cuda() 
#     optimizer = optim.Adam(denoiser.parameters(), lr=1e-4)
#     criterion = nn.MSELoss()

#     best_val_loss = float("inf")
#     save_path = "best_denoiser.pth"

#     for epoch in range(10):
#         denoiser.train()
#         total_loss = 0.0

#         for noisy, clean in train_loader:
#             noisy, clean = noisy.cuda(), clean.cuda()

#             # -----------------------------
#             # Extract embeddings from WavLM (frozen)
#             # -----------------------------
#             noisy_emb = get_embeddings(noisy)   # [B, T, 1024]
#             clean_emb = get_embeddings(clean)   # [B, T, 1024]

#             # -----------------------------
#             # Forward pass through projection + BiLSTM
#             # -----------------------------
#             enhanced_emb = denoiser(noisy_emb)  # [B, T, 512]

#             # Project clean embeddings into 512-dim space (using the same projection layer)
#             with torch.no_grad():
#                 clean_proj = denoiser.proj_in(clean_emb)  # [B, T, 512]

#             # -----------------------------
#             # Compute loss
#             # -----------------------------
#             min_len = min(enhanced_emb.size(1), clean_proj.size(1))
#             loss = criterion(enhanced_emb[:, :min_len, :], clean_proj[:, :min_len, :])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_train_loss = total_loss / len(train_loader)

#         # -----------------------------
#         # Validation
#         # -----------------------------
#         denoiser.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for noisy, clean in val_loader:
#                 noisy, clean = noisy.cuda(), clean.cuda()

#                 noisy_emb = get_embeddings(noisy)   # [B, T, 1024]
#                 clean_emb = get_embeddings(clean)   # [B, T, 1024]

#                 enhanced_emb = denoiser(noisy_emb)  # [B, T, 512]
#                 clean_proj = denoiser.proj_in(clean_emb)  # [B, T, 512]

#                 min_len = min(enhanced_emb.size(1), clean_proj.size(1))
#                 loss = criterion(enhanced_emb[:, :min_len, :], clean_proj[:, :min_len, :])
#                 val_loss += loss.item()

#         avg_val_loss = val_loss / len(val_loader)

#         print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

#         # -----------------------------
#         # Save best model
#         # -----------------------------
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save({
#                 "epoch": epoch+1,
#                 "model_state": denoiser.state_dict(),
#                 "optimizer_state": optimizer.state_dict(),
#                 "val_loss": best_val_loss
#             }, save_path)
#             print(f"✅ Saved best model at epoch {epoch+1} with val_loss={best_val_loss:.4f}")


# if __name__ == "__main__":
#     train()





import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import torch
from dataloader_libri import SpeechEnhancementDataset

# Load feature extractor + model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large").eval().cuda()

# Freeze parameters
for p in wavlm.parameters():
    p.requires_grad = False

def get_embeddings(waveform, sr=16000):
    """
    waveform: torch.Tensor [B, T] or [T]
    returns: torch.Tensor [B, T, 1024]
    """
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
        out, _ = self.lstm(x)  # [B, T, 2*hidden]
        out = self.proj(out)   # [B, T, output_dim]
        return out


# -----------------------------
# 3. Training Loop
# -----------------------------
def train():
    # Dataset
    dataset = SpeechEnhancementDataset(
        clean_dir="/data/raj/diarization/dataset/LibriSpeech/train-clean-5",
        noise_dir="/data/raj/cabm_ass/libri-musan/data/local/musan_bgnoise",
        sample_rate=16000,
        segment_length=5.0
    )

    # Train/val split
    val_size = int(0.05 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    # Model, optimizer, loss
    denoiser = BiLSTMDenoiser().cuda()
    optimizer = optim.Adam(denoiser.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    save_path = "best_denoiser.pth"

    for epoch in range(1):
        denoiser.train()
        total_loss = 0.0

        for i,(noisy, clean,path_audio) in enumerate(train_loader):
            if i>=10:
                break
            noisy, clean = noisy.cuda(), clean.cuda()
            print(path_audio)
            # Extract embeddings
            noisy_emb = get_embeddings(noisy)   # [B, T, 1024]
            clean_emb = get_embeddings(clean)   # [B, T, 1024]

            # Forward pass
            enhanced_emb = denoiser(noisy_emb)

            # Match sequence length
            min_len = min(enhanced_emb.size(1), clean_emb.size(1))
            loss = criterion(enhanced_emb[:, :min_len, :], clean_emb[:, :min_len, :])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # -----------------------------
        # Validation
        # -----------------------------
        denoiser.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean,val_audio_path in val_loader:
                noisy, clean = noisy.cuda(), clean.cuda()
                noisy_emb = get_embeddings(noisy)
                clean_emb = get_embeddings(clean)
                enhanced_emb = denoiser(noisy_emb)
                min_len = min(enhanced_emb.size(1), clean_emb.size(1))
                loss = criterion(enhanced_emb[:, :min_len, :], clean_emb[:, :min_len, :])
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch+1,
                "model_state": denoiser.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": best_val_loss
            }, save_path)
            print(f"✅ Saved best model at epoch {epoch+1} with val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    train()





