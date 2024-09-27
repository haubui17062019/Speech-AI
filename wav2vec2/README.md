# 1. Run docker triton
```bash
docker pull nvcr.io/nvidia/tritonserver:22.12
docker run --gpus '"device=1"' -it --name triton_wav2vec2 \
  -p8818:8000 -p8819:8001 -p8820:8002 \
 -v /home1/data/haubui/Speech-AI/wav2vec2:/mnt \
 --shm-size=16G nvcr.io/nvidia/tritonserver:22.12-py3
```

# 2. Convert onnx2trt
```bash
/usr/src/tensorrt/bin/trtexec --onnx=./ckpt/wav2vec-fix-ng-h.onnx \
                              --saveEngine=./ckpt/wav2vec-fix-ng-h.plan \
                              --explicitBatch \
                              --minShapes=input:1x1000 \
                              --optShapes=input:4x1000 \
                              --maxShapes=input:4x10000 \
                              --verbose=0 \
                              --device=0
```

# 3. Run Triton Inference Server
```bash
tritonserver --model-repository=/mnt/model/ --model-control-mode=explicit --load-model=wav2vec2
```

# TODO
- Cần chỉnh lại length khi convert triton
- Coding word-level timestamps: sử dụng forced alignments
- Cắt thành chunk 30s cho mỗi lần infer tránh out of memory: tham khảo whisper