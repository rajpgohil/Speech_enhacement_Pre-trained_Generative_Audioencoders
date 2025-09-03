import torch
import torchaudio
from torch.utils.data import DataLoader
from  dataloader_libri import SpeechEnhancementDataset  # assuming dataset class is in dataset.py

def save_audio(waveform, path, sr=16000):
    torchaudio.save(path, waveform.unsqueeze(0).cpu(), sr)

if __name__ == "__main__":
    dataset = SpeechEnhancementDataset(
        clean_dir="/data/raj/diarization/dataset/LibriSpeech/train-clean-5",
        noise_dir="/data/raj/cabm_ass/libri-musan/data/local/musan_bgnoise",
        sample_rate=16000,
        segment_length=5.0
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    noisy, clean = next(iter(loader))  # get one sample

    print("Noisy shape:", noisy.shape)
    print("Clean shape:", clean.shape)

    save_audio(noisy[0], "noisy_example.wav")
    save_audio(clean[0], "clean_example.wav")

    print("Saved noisy_example.wav and clean_example.wav")