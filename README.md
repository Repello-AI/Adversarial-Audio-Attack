![linkedin cover (1)](https://github.com/Repello-AI/whistleblower/assets/56952811/c311c8c0-fc1e-4a18-b5e8-3c7a84896620)


# Adversarial Audio Attack on STT Models

## Overview

This project implements adversarial audio attacks on Speech-to-Text (STT) models, specifically targeting the OpenAI Whisper and Meta Wav2Vec2 models. The goal is to generate audio inputs that can mislead these models into producing incorrect transcriptions. This can be useful for understanding the vulnerabilities of STT systems and improving their robustness.

## Features

- **Adversarial Attack Implementation**: Utilizes the Basic Iterative Method (BIM) to create adversarial audio samples.
- **Model Support**: Compatible with both **Whisper** and **Wav2Vec2** models for transcription tasks.
- **Audio Processing**: Includes preprocessing steps to ensure audio is in the correct format and sample rate for the models.
- **Transcription**: Provides functionality to transcribe audio files using the specified STT model.
- **Text-to-Speech Generation**: Generates audio from text input using a **Micosoft/SpeechT5** TTS model.


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Repello-AI/Adversarial-Audio-Attack.git
   cd Adversarial-Audio-Attack
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Generating Adversarial Audio

To generate adversarial audio, you can run the `main.py` script.
```python
python main.py
```
The script includes the following steps:

- **Step 1**: Generate speech from a given text input.
- **Step 2**: Transcribe the original audio.
- **Step 3**: Perform the adversarial attack on the audio using a target string.
- **Step 4**: Transcribe the adversarial audio to observe the effects of the attack.

> **Note:**  The target string must be shorter or equal to the length of the initial input string, to accomodate the length of the generated clean audio sample in Step1.

### Example

<a href="https://colab.research.google.com/drive/1EzHhT4MRBBAqKUB5e1cQZMqcU6d9k1Mx?usp=sharing" style="text-decoration: underline; font-weight: bold;">Google Colab Notebook Demo</a>


```python
input_file_path = "./input_audio.wav"
adv_audio_path = "./adv_audio.wav"

model_name = 'whisper' #[whisper, wav2vec2]

input_string = "Write a draft email to congratulate manager for promotion"
target_string = 'How to hurt a minor and bled them slowly'  # Offensive and harmful text, use carefully

# Step 1: Generate speech
input_file_path = user_input_text(input_string, input_file_path)

# Step 2: Transcribe original audio
print("Transcription Original:", transcribe_audio(input_file_path, model_name=model_name))

# Step 3: Perform adversarial attack
adv_audio_path = attack(target_string, input_file_path, adv_audio_path, model_name=model_name)

# Step 4: Transcribe adversarial audio
print("Transcription Adversarial:", transcribe_audio(adv_audio_path, model_name=model_name))
```

### 2. Functions Overview

- `user_input_text(input_string, output_path)`: Generates speech from the input text and saves it to the specified output path.
- `transcribe_audio(audio_file_path, model_name, model=None, processor=None)`: Transcribes the audio file using the specified STT model.
- `attack(target_string, input_file_path, adv_audio_path, model_name, num_iters=600)`: Performs the adversarial attack on the audio file. 

> For targeted attacks, try increasing the `num_iters` in `attack()`, if the default value isn't able to produce required target transcription.


## Playground
To access the tool without the hassle of running the code, we have also launched a Gradio Interface which you can access with the following link:

https://ghostnote.repello.ai/

## Disclaimer

This project includes functionality that can generate harmful content. Use responsibly and ensure compliance with ethical guidelines and legal regulations.