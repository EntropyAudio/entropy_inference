import io
import os
import logging
import torch
import torchaudio
from entropy_data.src.dataset.utils.audio_utils import trim_silence
from entropy_data.src.dataset.data_models import AudioConditioning
from entropy_training.src.utils.enum import SamplerType
from .utils import constants as c
import base64

logger = logging.getLogger(c.LOGGER_NAME)

def run_inference(cfg, model, prompt, batch_size):
    """
    Runs sampling by calling model.generate() on our denoising diffusion model.

    Args:
        cfg: The model inference config.
        model: The denoising model.
        prompt: The prompt.
        batch_size: The batch size.

    Returns:
        Base64 encoded audio sampled from the model.
    """
    logger.info("Running inference...")

    with torch.inference_mode(), torch.autocast(device_type=str(cfg.environment.device), dtype=eval(cfg.training.dtype)):
        output = model.generate(
            steps=100,
            cfg_scale=6.0,
            conditioning=[
                AudioConditioning(description=prompt, key=None, bpm=None, loop=None) for _ in range(batch_size)
            ],
            latent_size=cfg.audio.latent_size,
            sigma_min=0.001,
            sigma_max=1000,
            rho=1.0,
            eta=0.1,
            sampler_type=SamplerType.DPMPP_3M_SDE,
        )

        output_list = []
        output_cpu = output.cpu()

        for idx, audio_tensor in enumerate(output_cpu):
            trimmed_audio = trim_silence(audio_tensor, threshold=0.005)

            max_val = torch.max(torch.abs(trimmed_audio)) + 1e-8
            norm_audio = trimmed_audio / max_val
            final_audio = (norm_audio.clamp(-1, 1).mul(32767).round().to(torch.int16))

            # save_dir = "./outputs_2"
            # os.makedirs(save_dir, exist_ok=True)
            # file_path = os.path.join(save_dir, f"output_{idx}.wav")
            # torchaudio.save(
            #     file_path,
            #     final_audio,
            #     cfg.audio.sample_rate,
            #     format="wav"
            # )

            buffer = io.BytesIO()
            torchaudio.save(
                buffer,
                final_audio,
                cfg.audio.sample_rate,
                format="wav"
            )
            buffer.seek(0)
            base64_audio = base64.b64encode(buffer.read()).decode('utf-8')
            output_list.append(base64_audio)

        return output_list