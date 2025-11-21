### Input Schema
```angular2html
{
    "input": {
        "prompt": The text prompt for the model,
        "batch_size": The generation batch size,
    }
}
```

### Output Schema
```angular2html
{
    "prompt": The input prompt.
    "audio_base64": A list of base 64 encoded audio files,
}
```

___
```angular2html
python -m src.handler
```

```angular2html
docker build --platform linux/amd64 --tag entropyaudio/inference .
```

```angular2html
docker push entropyaudio/inference:latest
```

```angular2html
docker run --rm -it --gpus all --platform linux/amd64 entropyaudio/inference 
```

```angular2html
docker.io/entropyaudio/inference:latest
```