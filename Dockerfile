FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

WORKDIR /

RUN pip install --no-cache-dir runpod

COPY src/ ./src/
COPY entropy_data/ ./entropy_data/
COPY entropy_metrics/ ./entropy_metrics/
COPY entropy_stable_audio_open/ ./entropy_stable_audio_open/
COPY entropy_training/ ./entropy_training/

RUN pip install -e ./entropy_data/
RUN pip install -e ./entropy_metrics/
RUN pip install -e ./entropy_stable_audio_open/
RUN pip install -e ./entropy_training/

RUN pip install transformers bitsandbytes accelerate einops julius omegaconf k_diffusion
RUN pip install flash-attn --no-build-isolation

CMD ["python", "-m", "src.handler"]