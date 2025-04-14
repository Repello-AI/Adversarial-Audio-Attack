import gradio as gr
import torch
from main import user_input_text, transcribe_audio, attack

# File paths (these files are overwritten on each run)
ORIGINAL_AUDIO_PATH = "./input_audio.wav"
ATTACKED_AUDIO_PATH = "./attacked_audio.wav"

def convert_text_to_audio(input_text):
    """
    Convert input text to speech (TTS), saving the output to a file.
    Returns the file path for playback and state.
    """
    file_path = user_input_text(input_text, ORIGINAL_AUDIO_PATH)
    return file_path, file_path  # (output for audio component, update hidden state)

def transcribe_original_audio(audio_path, model_name):
    """
    Transcribe the clean, TTS-generated audio.
    """
    model_name = model_name.lower()
    transcription = transcribe_audio(audio_path, model_name)
    return transcription, model_name

def generate_attacked_audio(target_text, original_audio_path, model_name):
    """
    Generate adversarial (attacked) audio based on the target text and the original audio.
    Returns the attacked audio file path.
    """
    model_name = model_name.lower()
    attacked_path = attack(target_text.upper(), original_audio_path, ATTACKED_AUDIO_PATH, model_name)
    return attacked_path, attacked_path, model_name  # (output for audio component, update hidden state)

def transcribe_attacked_audio(audio_path, model_name):
    """
    Transcribe the adversarial (attacked) audio.
    """
    model_name = model_name.lower()
    transcription = transcribe_audio(audio_path, model_name)
    return transcription, model_name


with gr.Blocks(theme=gr.themes.Base(),
    css="""
        #step1 { background-color: #E3F2FD !important; }
        #step2 { background-color: #FFF3E0 !important; }
        #step3 { background-color: #E8F5E9 !important; }
        #step4 { background-color: #FCE4EC !important; }
        .rounded-column {
            padding: 20px; 
            border-radius: 12px; 
            border: 1px solid #4662f0;
        }
        .rounded-button {
            border-radius: 25px !important;
            width: 50% !important;
            margin: 10px auto !important;
            display: block !important;
            background-color: #4662f0 !important;
            color: white !important; 
        }
        .transparent-box {
            background-color: transparent !important;
            box-shadow: none !important;
            border: none !important;
        }
        .transparent-box input,
        .transparent-box textarea,
        .transparent-box select {
            background-color: white !important;  /* keeps text area readable */
            border: 1px solid #ccc !important;   /* optional subtle border */
        }
        .transparent-markdown {
            background-color: transparent !important;
            box-shadow: none !important;
        }
        .transparent-dropdown {
            background-color: transparent;
            box-shadow: none !important;
            padding: 10px !important;
        }
    """) as demo:
    gr.Markdown("<h1 style='color:#4B8BBE'>🎧 Adversarial Audio Attack Demo</h1>")
    gr.Markdown("""
        <div style='margin-bottom: 20px;'>
            This demo shows how subtle targeted adversarial perturbations can fool a speech recognition system. 
        </div>
    """)

    # --- Step 1: Convert Text to Audio ---
    with gr.Column(elem_classes="rounded-column", elem_id="step1"):
        gr.Markdown("### 📝 Step 1: Convert Text to Audio", elem_classes="transparent-markdown")
        input_text = gr.Textbox(label="🗣️ Enter text to synthesize", placeholder="Type your sentence here...", elem_classes="transparent-box", elem_id="step1")
        convert_button = gr.Button("🎙️ Convert to Audio", elem_classes="rounded-button")
        original_audio = gr.Audio(label="Generated Audio", interactive=True)
        original_audio_state = gr.State()

        convert_button.click(
            fn=convert_text_to_audio,
            inputs=input_text,
            outputs=[original_audio, original_audio_state]
        )

    # --- Step 2: Transcribe Clean Audio ---
    with gr.Column(elem_classes="rounded-column", elem_id="step2"):
        gr.Markdown("### 🔍 Step 2: Transcribe the Generated Audio", elem_classes="transparent-markdown")
        model_selector = gr.Dropdown(choices=["Wav2Vec2", "Whisper"], value="Wav2Vec2", label="Select Model", elem_classes="transparent-dropdown", elem_id="step2")
        transcribe_clean_button = gr.Button("🧠 Transcribe Clean Audio", elem_classes="rounded-button")
        clean_transcription = gr.Textbox(label="Clean Audio Transcription", lines=1, elem_classes="transparent-box", elem_id="step2")
        model_name_state = gr.State()

        transcribe_clean_button.click(
            fn=transcribe_original_audio,
            inputs=[original_audio_state, model_selector],
            outputs=[clean_transcription, model_name_state]
        )

    # --- Step 3: Generate Attacked Audio ---
    with gr.Column(elem_classes="rounded-column", elem_id="step3"):
        gr.Markdown("### 🧨 Step 3: Generate Attacked Audio", elem_classes="transparent-markdown")
        target_text = gr.Textbox(label="🎯 Enter target transcription", placeholder="Type target text here...", elem_classes="transparent-box", elem_id="step3")
        attack_button = gr.Button("⚔️ Generate Attacked Audio", elem_classes="rounded-button")
        attacked_audio = gr.Audio(label="Attacked Audio", interactive=True)
        attacked_audio_state = gr.State()

        attack_button.click(
            fn=generate_attacked_audio,
            inputs=[target_text, original_audio_state, model_name_state],
            outputs=[attacked_audio, attacked_audio_state, model_name_state]
        )

    # --- Step 4: Transcribe Attacked Audio ---
    with gr.Column(elem_classes="rounded-column", elem_id="step4"):
        gr.Markdown("### 🧪 Step 4: Transcribe the Attacked Audio", elem_classes="transparent-markdown")
        transcribe_attacked_button = gr.Button("🔍 Transcribe Attacked Audio", elem_classes="rounded-button")
        attacked_transcription = gr.Textbox(label="Attacked Audio Transcription", lines=1, elem_classes="transparent-box", elem_id="step4")

        transcribe_attacked_button.click(
            fn=transcribe_attacked_audio,
            inputs=[attacked_audio_state, model_name_state],
            outputs=[attacked_transcription, model_name_state]
        )

if __name__ == "__main__":
    demo.launch()