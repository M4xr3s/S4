import os
import soundfile as sf
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch

# Load Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def extract_features(audio_file, target_sample_rate=16000):
    audio_input, sample_rate = sf.read(audio_file)
    if sample_rate != target_sample_rate:
        audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=target_sample_rate)
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=target_sample_rate).input_values
    input_values = input_values.to(device)
    with torch.no_grad():
        hidden_states = model(input_values).last_hidden_state.squeeze().cpu().numpy()
    return hidden_states

def concatenate_features(features, window_size=2):
    concatenated_features = []
    for i in range(0, len(features) - window_size + 1, window_size):
        concatenated_features.append(np.concatenate(features[i:i+window_size], axis=0))
    return np.array(concatenated_features)

def process_files(audio_dir, csv_dir, output_dir):
    for filename in os.listdir(audio_dir):
        if filename.endswith(".wav"):
            base_name = os.path.splitext(filename)[0]
            audio_file = os.path.join(audio_dir, filename)
            csv_file = os.path.join(csv_dir, f"{base_name}.csv")
            output_file = os.path.join(output_dir, f"{base_name}_output.csv")

            features = extract_features(audio_file)
            concat_features = concatenate_features(features)

            arousal_df = pd.read_csv(csv_file, usecols=['Valence'], delimiter=';', skiprows=0)
            features_df = pd.DataFrame(concat_features)
            combined_df = pd.concat([features_df, arousal_df], axis=1)
            combined_df.to_csv(output_file, index=False)
            print(f"Processed and saved: {output_file}")
