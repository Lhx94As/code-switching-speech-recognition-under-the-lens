import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, load_wav_16k
from transformers import AutoModel, AutoTokenizer
import torch, torchaudio
import json, time, logging
from argparse import ArgumentParser

# # Load the FLAC file
# data, samplerate = librosa.load('/home3/hexin/espnet/egs2/seame/asr1/dump/raw/org/train_sp/data/format.1/nc01f-01nc01fbx_0101-002638-002771.flac', sr=None)

# # Write as WAV file
# sf.write('output.wav', data, samplerate)



audio_path = '/home4/asr_corpora/original/limited/asru_data/asru_cs200/data/category/G0001/session01/T0001G0001_S01010002.wav'
prompt_speech = load_wav(audio_path, 16000)
prompt_text = "新 年 第 一 天 来 了 个 reject 扎 心 啊"
text = '没 有 找 到 lost'


# initialize cosyvoice model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, device=device)

for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech, stream=False)):
    torchaudio.save('result_zero_shot.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
print(f'zero shot done!')

# load finetuned checkpoint
ft_ckpt = torch.load('results_asru/epoch_020.pth')
new_ft_ckpt = {k.replace("module.", ""): v for k, v in ft_ckpt.items()}
cosyvoice.model.llm.load_state_dict(new_ft_ckpt, strict=True)
print(f'loaded finetuned checkpoint!')

for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech, stream=False)):
    torchaudio.save('result_ft.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
print(f'ft eval done!')