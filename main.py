from pickle import FALSE
# coding: utf-8
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os
from transformers import pipeline
import soundfile as sf
from datasets import load_dataset
import re
from whisper_utils import _torch_extract_fbank_features
import logging

# Suppress warnings from transformers
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("TTS").setLevel(logging.ERROR)

class BIM:
    def __init__(self, model, model_name, images, target_label, eps, alpha,
                 num_iters=0, random_state=False, processor=None, beta=0.1, verbose=False):
        self.model = model
        self.model_name = model_name
        self.device = next(model.parameters()).device
        self.orig_img = images.clone().detach().to(self.device)
        self.eps = eps
        self.target_label = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', '', target_label)).strip()
        if self.model_name == 'wav2vec2':
            self.target_label = self.target_label.upper()
        elif self.model_name == 'whisper':
            self.target_label = self.target_label.lower()
        self.alpha = alpha
        self.rand = random_state
        self.img_bim = images.clone().detach().to(torch.float32).requires_grad_(True)
        self.img_bim = self.img_bim.to(self.device)
        self.img_bim.retain_grad()
        self.processor = processor
        self.beta = beta
        self.verbose=verbose

        if not random_state:
            self.num_iters = math.ceil(min((self.eps / self.alpha) + 4, 1.25 * (self.eps / self.alpha)))
        else:
            self.num_iters = num_iters

        if self.verbose:
            print("Num iters", self.num_iters)
            print('Target String:', self.target_label )

    def compute_ctc_loss(self, logits, target_ids):
        target_lengths = torch.tensor([target_ids.size(1)], dtype=torch.long, device=self.device)
        input_lengths = torch.tensor([logits.size(1)], dtype=torch.long, device=self.device)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        ctc_loss_fn = nn.CTCLoss(blank=self.processor.tokenizer.pad_token_id, zero_infinity=True)
        return ctc_loss_fn(log_probs, target_ids, input_lengths, target_lengths)

    def attack(self):
        loss_arr = []

        progress_bar = tqdm(range(self.num_iters), desc="BIM Attack Progress", unit="step", disable = not self.verbose)
        best_loss = float('inf')
        best_adv_audio = None

        target_ids = self.processor(text=self.target_label, return_tensors="pt").input_ids.to(self.device)
        decoder_input_ids = target_ids[:, :-1]
        labels = target_ids[:, 1:].clone()
        
        for i in progress_bar:
            self.img_bim = self.img_bim.clone().detach().requires_grad_(True)

            if self.model_name == 'wav2vec2':
                self.img_bim.requires_grad = True
                input_values = self.img_bim.squeeze(0)
                input_values = (input_values - input_values.mean()) / (input_values.std() + 1e-6)
                input_values = input_values.unsqueeze(0)
                assert input_values.requires_grad, "Input values must require gradients"

                output = self.model(input_values)
                logits = output.logits
                ctc_loss = self.compute_ctc_loss(logits, target_ids)

                ## l2 penalty
                l2_loss = torch.norm(self.img_bim - self.orig_img, p=2)
                loss = ctc_loss + self.beta * l2_loss
                alignment_loss = ctc_loss

            elif self.model_name == 'whisper':
                input_values = _torch_extract_fbank_features(self.img_bim)
                input_values = input_values.to(self.device)
                assert input_values.requires_grad, "Input values must require gradients"
                output = self.model(
                    input_features=input_values,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels
                )
                loss = output.loss
                alignment_loss = loss

            ## backprop
            loss.backward()

            if self.img_bim.grad is None:
                raise ValueError("Gradients for img_bim are None.")

            loss_arr.append(round(loss.item(), 4))
            progress_bar.set_postfix(loss=loss.item(), alignment_loss=alignment_loss.item())

            ## BIM Attack
            grads = self.img_bim.grad
            self.img_bim = self.img_bim - self.alpha * grads.data.detach().sign()
            self.img_bim = torch.clamp(self.img_bim, min=-1, max=1)
            self.img_bim = torch.clamp(self.img_bim, min=self.orig_img - self.eps, max=self.orig_img + self.eps)
            self.img_bim = self.img_bim.clone().detach().requires_grad_(True)

            ## copy best value
            if alignment_loss.item() < best_loss:
                best_loss = alignment_loss.item()
                best_adv_audio = self.img_bim.clone().detach()

            # ### Transcription after each iteration
            if self.verbose and i%50 == 0:
                with torch.no_grad():
                    if self.model_name == 'wav2vec2':
                        norm_audio = self.img_bim.squeeze(0)
                        norm_audio = (norm_audio - norm_audio.mean()) / (norm_audio.std() + 1e-6)
                        input_values = self.processor(norm_audio, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
                        logits = self.model(input_values).logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                    elif self.model_name == 'whisper':
                        input_values = _torch_extract_fbank_features(self.img_bim)
                        input_values = input_values.to(self.device)
                        predicted_ids = self.model.generate(input_values)

                    transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    print(f"\n[Step {i+1}] Transcription: {transcription}\n")

                    if transcription.upper().strip() == self.target_label.upper().strip():
                        print("Target reached")
                        best_adv_audio = self.img_bim.clone().detach()
                        clipped_delta = torch.clamp(best_adv_audio.data - self.orig_img.data, -self.eps, self.eps)
                        return best_adv_audio, best_adv_audio - self.orig_img, loss_arr

        clipped_delta = torch.clamp(best_adv_audio.data - self.orig_img.data, -self.eps, self.eps)
        return best_adv_audio, clipped_delta, loss_arr

def loadwav2vec2():
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

def get_predictions(audio, model, processor, target_sample_rate,  model_name):
    if model_name == 'wav2vec2':
        input_values = processor(audio.squeeze(0), sampling_rate=target_sample_rate, return_tensors="pt").input_values.to(model.device)
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    elif model_name == 'whisper':
        input_values = processor(audio.squeeze(0), sampling_rate=target_sample_rate, return_tensors="pt").input_features.to(model.device)
        predicted_ids = model.generate(input_values)

    return predicted_ids

def transcribe_audio(audio_file_path, model_name='wav2vec2', model=None, processor=None):
    target_sample_rate = 16000
    adv_audio, sample_rate = torchaudio.load(audio_file_path)
    adv_audio, target_sample_rate = preprocess_waveform(adv_audio, sample_rate, target_sample_rate=target_sample_rate)

    if model is None or processor is None:
        model, processor = load_model_processor(model_name, verbose=True)
    
    predicted_ids = get_predictions(adv_audio, model, processor, target_sample_rate, model_name)

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

def user_input_text(input_string, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating speech for: {input_string} -> {output_path}")
    # Load Bark TTS pipeline
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device="cpu")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    # You can replace this embedding with your own as well.

    output = synthesiser(input_string, forward_params={"speaker_embeddings": speaker_embedding})

    # Save audio to file
    sf.write(output_path, output["audio"], 16000)

    print(f"Audio file saved in '{output_path}'")
    return output_path

def attack(target_string, input_file_path, adv_audio_path, model_name):

    if model_name == 'wav2vec2':
        eps = 0.05
        alpha = 0.005
    elif model_name == 'whisper':
        eps = 0.05
        alpha = 0.01
    num_iters = 600
    random_state = True
    beta = 0.1  # L2 penalty coefficient
    verbose=True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    audio, sample_rate = torchaudio.load(input_file_path)
    audio, target_sample_rate = preprocess_waveform(audio, sample_rate, target_sample_rate=16000)

    model, processor = load_model_processor(model_name, verbose=True)
    model = model.to(device)

    bim = BIM(model, model_name, audio, target_string, eps, alpha, num_iters, random_state, processor=processor, beta=beta, verbose=verbose)
    adv_audio, delta, loss_arr = bim.attack()

    torchaudio.save(adv_audio_path, adv_audio.detach().cpu(), target_sample_rate)
    return adv_audio_path


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_file_path = "./input_audio.wav"
    adv_audio_path = "./adv_audio.wav"
    model_name = 'wav2vec2'

    input_string = "I will win the national championship this year."
    target_string = 'Kill those bastards'  # NEVER use harmful or offensive content

    ## step1
    input_file_path = user_input_text(input_string, input_file_path)

    ## step2
    print("Transcription Original:", transcribe_audio(input_file_path, model_name=model_name))

    ## step3
    adv_audio_path = attack(target_string, input_file_path, adv_audio_path, model_name=model_name)

    ## step4
    print("Transcription Adversarial:", transcribe_audio(adv_audio_path, model_name=model_name))
