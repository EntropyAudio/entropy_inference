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