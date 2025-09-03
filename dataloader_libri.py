import torch
import torchaudio
import random
import os

class SpeechEnhancementDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dir, noise_dir, sample_rate=16000, segment_length=3.0):
        self.clean_files = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(clean_dir)
            for f in fn if f.endswith(".flac") or f.endswith(".wav")
        ]
        self.noise_files = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(noise_dir)
            for f in fn if f.endswith(".wav")
        ]
        self.sr = sample_rate
        self.seg_len = int(segment_length * sample_rate)

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]
        clean, sr = torchaudio.load(clean_path)
        clean = torchaudio.functional.resample(clean, sr, self.sr)

        # Random crop
        if clean.size(1) > self.seg_len:
            start = random.randint(0, clean.size(1) - self.seg_len)
            clean = clean[:, start:start+self.seg_len]
        else:
            clean = torch.nn.functional.pad(clean, (0, self.seg_len - clean.size(1)))

        # Pick random noise
        noise_path = random.choice(self.noise_files)
        noise, sr = torchaudio.load(noise_path)
        noise = torchaudio.functional.resample(noise, sr, self.sr)

        # Random crop noise
        if noise.size(1) > self.seg_len:
            start = random.randint(0, noise.size(1) - self.seg_len)
            noise = noise[:, start:start+self.seg_len]
        else:
            noise = torch.nn.functional.pad(noise, (0, self.seg_len - noise.size(1)))

        # Mix with random SNR
        snr_db = random.uniform(-5, 20)
        clean_power = clean.norm(p=2)
        noise_power = noise.norm(p=2)
        factor = (clean_power / (10**(snr_db/20))) / (noise_power + 1e-8)
        noisy = clean + factor * noise

        return noisy.squeeze(0), clean.squeeze(0),clean_path

# =======================================================================================
# =======================================================================================
# =======================================================================================
# =======================================================================================
# import torch
# import torchaudio
# import random
# import tempfile
# import subprocess
# import os
# from encodec import EncodecModel
# from encodec.utils import convert_audio

# # -----------------------------
# # MP3 re-encoding
# # -----------------------------
# def mp3_reencode(waveform, sr=16000, bitrate="32k"):
#     with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_in, tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_mp3, tempfile.NamedTemporaryFile(suffix=".wav") as tmp_out:
#         torchaudio.save(tmp_in.name, waveform.unsqueeze(0).cpu(), sr)
#         subprocess.run(["ffmpeg", "-y", "-i", tmp_in.name, "-b:a", bitrate, tmp_mp3.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         subprocess.run(["ffmpeg", "-y", "-i", tmp_mp3.name, tmp_out.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         degraded, _ = torchaudio.load(tmp_out.name)
#     return degraded.squeeze(0)

# # -----------------------------
# # AAC re-encoding
# # -----------------------------
# def aac_reencode(waveform, sr=16000, bitrate="32k"):
#     with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_in, tempfile.NamedTemporaryFile(suffix=".aac") as tmp_aac, tempfile.NamedTemporaryFile(suffix=".wav") as tmp_out:
#         torchaudio.save(tmp_in.name, waveform.unsqueeze(0).cpu(), sr)
#         subprocess.run(["ffmpeg", "-y", "-i", tmp_in.name, "-b:a", bitrate, tmp_aac.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         subprocess.run(["ffmpeg", "-y", "-i", tmp_aac.name, tmp_out.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         degraded, _ = torchaudio.load(tmp_out.name)
#     return degraded.squeeze(0)

# # -----------------------------
# # EnCodec low-codebook re-encoding
# # -----------------------------
# encodec_model = EncodecModel.encodec_model_24khz()
# encodec_model.set_target_bandwidth(1.5)  # very low bitrate
# encodec_model.eval()

# def encodec_reencode(waveform, sr=16000):
#     wav = convert_audio(waveform.unsqueeze(0), sr, encodec_model.sample_rate, encodec_model.channels)
#     with torch.no_grad():
#         encoded_frames = encodec_model.encode(wav)
#         decoded = encodec_model.decode(encoded_frames)
#     return decoded.squeeze(0).cpu()

# # -----------------------------
# # μ-law or A-law companding
# # -----------------------------
# def mu_law(waveform, quantization_channels=256):
#     return torchaudio.functional.mu_law_encoding(waveform, quantization_channels).float()

# def a_law(waveform):
#     return torchaudio.functional.a_law_encoding(waveform).float()

# def mu_law_decode(encoded, quantization_channels=256):
#     return torchaudio.functional.mu_law_decoding(encoded, quantization_channels).float()

# def a_law_decode(encoded):
#     return torchaudio.functional.a_law_decoding(encoded).float()

# def mu_law_reencode(waveform):
#     return mu_law_decode(mu_law(waveform))

# def a_law_reencode(waveform):
#     return a_law_decode(a_law(waveform))

# # # -----------------------------
# # # Placeholder: small HiFi-GAN re-encoding
# # # -----------------------------
# # # You would need a pretrained "small" HiFi-GAN generator
# # # For now, we can just simulate with a lowpass filter
# # def hifigan_small_reencode(waveform, sr=16000):
# #     return torchaudio.functional.lowpass_biquad(waveform, sr, cutoff_freq=3000)

# class SpeechEnhancementDataset(torch.utils.data.Dataset):
#     def __init__(self, clean_dir, sample_rate=16000, segment_length=3.0):
#         self.clean_files = [
#             os.path.join(dp, f)
#             for dp, dn, fn in os.walk(clean_dir)
#             for f in fn if f.endswith(".flac") or f.endswith(".wav")
#         ]
#         self.sr = sample_rate
#         self.seg_len = int(segment_length * sample_rate)

#         # List of augmentations
#         self.augs = [
#             lambda x: mp3_reencode(x, self.sr, "32k"),
#             lambda x: aac_reencode(x, self.sr, "32k"),
#             lambda x: encodec_reencode(x, self.sr),
#             lambda x: mu_law_reencode(x),
#             lambda x: a_law_reencode(x),
#             lambda x: hifigan_small_reencode(x, self.sr),
#         ]

#     def __len__(self):
#         return len(self.clean_files)

#     def __getitem__(self, idx):
#         clean_path = self.clean_files[idx]
#         clean, sr = torchaudio.load(clean_path)
#         clean = torchaudio.functional.resample(clean, sr, self.sr)

#         # Random crop
#         if clean.size(1) > self.seg_len:
#             start = random.randint(0, clean.size(1) - self.seg_len)
#             clean = clean[:, start:start+self.seg_len]
#         else:
#             clean = torch.nn.functional.pad(clean, (0, self.seg_len - clean.size(1)))

#         # Apply random augmentation
#         aug_fn = random.choice(self.augs)
#         degraded = aug_fn(clean.squeeze(0))

#         return degraded, clean.squeeze(0)