
from pathlib import Path
import pandas as pd
import torch

def save_speaker_tensor_to_csv(speaker_name: str, tensor: torch.Tensor) -> str:
    msg = "succeed"
    speaker_path = f"{Path(__file__).resolve().parent}/sampled_speaker"
    try:
        #df = pd.DataFrame({"speaker": [float(i) for i in tensor]})
        df = pd.DataFrame(tensor.numpy())
        df.to_csv(f"{speaker_path}/{speaker_name}.csv", index=False, header=False)
    except Exception as e:
        print(f"存储 speaker_tensor 时发生错误：{e}")
        msg = "fail"
    finally:
        return msg

def load_speaker_tensor_from_csv(speaker_name: str) -> torch.Tensor:
    speaker_path = f"{Path(__file__).resolve().parent}/sampled_speaker"
    #d_s = pd.read_csv(f"{speaker_path}/{speaker_name}.csv", header=None).iloc[:, 0]
    d_s = pd.read_csv(f"{speaker_path}/{speaker_name}.csv", header=None)
    _speaker_tensor = torch.tensor(d_s.values)
    return _speaker_tensor