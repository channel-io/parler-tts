# import torch
# from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSConfig
# from transformers import AutoTokenizer
# import soundfile as sf

# device = "cuda:2" if torch.cuda.is_available() else "cpu"

# model_path = "/home/work/channel/dobby/TTS/modules/parler-tts/output_dir_training"
# tokenizer_path = "/home/work/channel/dobby/TTS/modules/parler-tts/output_dir_training"

# model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# # prompt = "[male] [20s] 안녕하세요. 잘 부탁드려요."
# prompt = "안녕하세요 일찍 출근하시네요"
# description = None

# # input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
# input_ids = None
# prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
# audio_arr = generation.cpu().numpy().squeeze()
# # model.config.sampling_rate = 8000
# sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)


import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

prompt = "Hey, how are you doing today?"
# description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

# input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
input_ids = None
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, 8000)