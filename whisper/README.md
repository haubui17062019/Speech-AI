# 1. Create Docker
```
docker run -it --name "whisper-server" --gpus '"device=2"' --net host -v /home1/data/haubui/Speech-AI:/mnt --shm-size=2g soar97/triton-whisper:24.05
```