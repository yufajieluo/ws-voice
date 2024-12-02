
import ChatTTS
import torch
import torchaudio
from tools.audio import load_audio

from utils import save_speaker_tensor_to_csv, load_speaker_tensor_from_csv

chat = ChatTTS.Chat()
chat.load(
    source = 'custom',
    compile = True,
    custom_path = '.'
)

#spk_smp = chat.sample_audio_speaker(load_audio(audio_file, 24000))
#print(spk_smp)


# save speaker
# speaker_tensor = chat.dvae.sample_audio(load_audio(audio_file, 24000))
# save_speaker_tensor_to_csv('test_speaker', speaker_tensor)

# load speaker
speaker_tensor = load_speaker_tensor_from_csv('test_speaker00-1')
spk_smp = chat.speaker.encode_prompt(speaker_tensor)

texts = ["我想吃红烧肉、炖肘子、涮肉"]

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_smp = spk_smp,
    txt_smp = "我想吃红烧肉、炖肘子、涮肉",
)

wavs = chat.infer(
    texts,
    params_infer_code = params_infer_code
)

torchaudio.save(
    'result.wav',
    torch.from_numpy(wavs[0]).unsqueeze(0),
    24000
)
