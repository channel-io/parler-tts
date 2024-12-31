import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSConfig
from transformers import AutoTokenizer
import soundfile as sf
import gradio as gr
import os

# Set device map for multiple GPUs
def get_device(gpu_id):
    return f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

gpu_count = torch.cuda.device_count()
gpus = [get_device(i) for i in range(gpu_count)]

# Load model and tokenizer
model_path = "/home/work/channel/dobby/TTS/modules/parler-tts/output_dir_training"
tokenizer_path = "/home/work/channel/dobby/TTS/modules/parler-tts/output_dir_training"

models = [
    ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(gpus[i])
    for i in range(gpu_count)
]
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def generate_audio(text, age="[20s]", gender="[male]"):
    # Determine which GPU to use (simple round-robin or load-balancing mechanism)
    gpu_id = torch.randint(0, gpu_count, (1,)).item()
    device = gpus[gpu_id]
    model = models[gpu_id]

    # Add special tokens to prompt
    prompt = f"{gender} {age} {text}"

    # Tokenize input
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate audio
    generation = model.generate(input_ids=None, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()

    # Save audio to temporary file
    output_path = "parler_tts_out.wav"
    sf.write(output_path, audio_arr, model.config.sampling_rate)

    return output_path

# Gradio Interface
def tts_interface(text, age, gender):
    output_audio_path = generate_audio(text, age, gender)
    return output_audio_path

audio_demo = gr.Interface(
    fn=tts_interface,
    inputs=[
        gr.Textbox(label="Enter text for TTS", placeholder="화면에 보이시는 워크플로우 좀 캡처 해주시겠어요?"),
        gr.Dropdown([f"[{i}s]" for i in range(0, 101, 10)], label="Age", value="[20s]"),
        gr.Dropdown(["[male]", "[female]"], label="Gender", value="[male]")
    ],
    outputs=gr.Audio(label="Generated Audio"),
    title="Parler TTS Multi-GPU",
    description="Enter text to generate speech using Parler TTS with customizable age and gender."
)

if __name__ == "__main__":
    audio_demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
