
import ChatTTS
import torch
import torchaudio
from tools.audio import load_audio

from utils import save_speaker_tensor_to_csv, load_speaker_tensor_from_csv

audio_file = 'result00.wav'

chat = ChatTTS.Chat()
chat.load(
    source = 'custom',
    compile = True,
    custom_path = '.'
)

# save speaker
speaker_tensor = chat.dvae.sample_audio(load_audio(audio_file, 24000))
save_speaker_tensor_to_csv('test_speaker00-1', speaker_tensor)
