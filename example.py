import ChatTTS
import torch
import torchaudio
from tools.audio import load_audio

chat = ChatTTS.Chat()
chat.load(
    source = 'custom',
    compile = True,
    custom_path = '.'
)

spk_smp = chat.sample_audio_speaker(load_audio("result00.wav", 24000))
#print(spk_smp)  # save it in order to load the speaker without sample audio next time

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_smp=spk_smp,
    #txt_smp="与sample.mp3内容完全一致的文本转写。",
)

wavs = chat.infer(
    "四川美食确实以辣闻名，但也有不辣的选择。比如甜水面、赖汤圆、蛋烘糕、叶儿粑等。",
    params_infer_code=params_infer_code,
)

torchaudio.save(
    'result.wav',
    torch.from_numpy(wavs[0]).unsqueeze(0),
    24000
)