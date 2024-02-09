import torch
from pathlib import Path
import gdown
from .utils import prepare_input
try:
    # older pip version
    from pip.utils.appdirs import user_cache_dir
except:
    # newer pip version
    from pip._internal.utils.appdirs import user_cache_dir

def initModel(device=None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # load PAl model
    weights_file = Path(user_cache_dir('pip'))/"pal4vst/swin-large_upernet_unified_512x512/end2end.pt"
    weights_file.parent.mkdir(parents=True, exist_ok=True)
    if not weights_file.exists():
        url = "https://drive.google.com/file/d/1bGjEKquWa4cZJ6v52NoRfViwKK15U7vd/view?usp=sharing"
        gdown.download(url, weights_file.as_posix(), quiet=False,fuzzy=True)
    model = torch.load(weights_file).to(device)
    return model, None, prepare_input
