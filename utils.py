import torch
import torchaudio
import re

def clean_punctuation(text):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', '', text)).strip()

def loadwav2vec2():
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()
    return model, processor

def load_whisper():
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    model_id = "openai/whisper-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")
    model.eval()
    return model, processor

def load_model_processor(model_name, verbose=True):
    if model_name == 'wav2vec2':
        model, processor = loadwav2vec2()
        if verbose:
            print('Loaded Wav2Vec2 model successfully')

    elif model_name == 'whisper':
        model, processor = load_whisper()
        if verbose:
            print('Loaded Whisper model successfully')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, processor

def preprocess_waveform(wv, sample_rate, target_sample_rate=16000):
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        wv = resampler(wv)
        sample_rate = target_sample_rate

    wv = wv.mean(dim=0)  # Convert to mono
    wv = wv.unsqueeze(0)  # Add batch dimension

    return wv, target_sample_rate
