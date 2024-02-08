import torch
from pathlib import Path
import gdown
from .utils import prepare_input

def initModel(device=None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # load PAl model
    weights_file = Path(__file__).parent/"deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt"
    if not weights_file.exists():
        url = "https://drive.google.com/file/d/1bGjEKquWa4cZJ6v52NoRfViwKK15U7vd/view?usp=sharing"
        gdown.download(url, weights_file.as_posix(), quiet=False,fuzzy=True)
    model = torch.load(weights_file).to(device)
    return model, None, prepare_input
