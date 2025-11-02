from pathlib import Path
import runpod
from .converter import extract_input
from .controller import run_inference
from .utils.utils import setup_logger
import entropy_training
import torch
from entropy_stable_audio_open.stable_audio_open import get_model
from entropy_training.src.utils.constants import CKPT_KEY_MODEL
from entropy_training.src.utils.utils import environment_setup
from omegaconf import OmegaConf

cfg = OmegaConf.load(Path(entropy_training.__file__).parent / "config.yaml")
environment_setup(cfg)
model = get_model(cfg, download_pretrained_weights=False)
model.load_state_dict(
    torch.load(Path(__file__).parent.parent / "ckpt/checkpoint_1000.pth", map_location="cpu", weights_only=False)[CKPT_KEY_MODEL], strict=True
)
model.to(cfg.environment.device)
model.eval()

def handler(event):
    logger = setup_logger()
    logger.info(f"Received event: {event}")
    input = extract_input(event)
    result = run_inference(cfg, input, model)
    return result

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
