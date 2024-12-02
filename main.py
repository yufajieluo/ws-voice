
import ChatTTS
import torch
import torchaudio
from utils import save_speaker_tensor_to_csv, load_speaker_tensor_from_csv

chat = ChatTTS.Chat()
chat.load(
    source = 'custom',
    compile = True,
    custom_path = '.'
)

texts = ["明天会不会下雪呢，如果下雪我就出去玩，如果不下雪我就去上班"]

speaker_tensor = chat.speaker._sample_random()
speaker_emb = chat.speaker._encode(speaker_tensor)
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = speaker_emb,   # add sampled speaker
    temperature=0.3,   # using custom temperature
    top_P=0.7,         # top P decode
    top_K=20,          # top K decode
    show_tqdm=False,   # no tqdm
    manual_seed=1234,  # seed
)
save_speaker_tensor_to_csv('test_speaker00', speaker_tensor)

wavs = chat.infer(
    texts, 
    params_infer_code = params_infer_code
)

torchaudio.save(
    'result00.wav',
    torch.from_numpy(wavs[0]).unsqueeze(0),
    24000
)