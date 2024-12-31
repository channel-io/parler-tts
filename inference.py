import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSConfig
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:2" if torch.cuda.is_available() else "cpu"

model_path = "/home/work/channel/dobby/TTS/modules/parler-tts/output_dir_training"
tokenizer_path = "/home/work/channel/dobby/TTS/modules/parler-tts/output_dir_training"

model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# prompt = "[male] [20s] 안녕하세요. 잘 부탁드려요."
prompt = "[male] [20s] 화면에 보이시는 워크플로우 좀 캡처 해주시겠어요?"
description = None

# input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
input_ids = None
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
print(prompt_input_ids)
print(tokenizer.decode(prompt_input_ids[0]))

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
# model.config.sampling_rate = 8000
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)