


# 1. Nếu chưa pull docker and convert model (Sử dụng model pretrained của NGC)
    - B1: bash /riva_quickstart/riva_init.sh
    - B2: bash /riva_quickstart/riva_start.sh

# 2. Cài đặt bổ sung

## 2.1 Nếu chưa có thư viện ffmpeg and ffmpeg-python để download luồng rstp:
```
apt-get update &&     apt-get install -y ffmpeg &&     pip install ffmpeg-python
```
Install lib
```
pip3 install soundfile
```
Note: Nếu chưa cài đặt lib grpc
```
b1: chmod +x gRPC_installation.sh in deepstream_asr_app
b2: ./gRPC_installation.sh
```

# 3. Model deploy
Model
```rmir_tts_fastpitch_hifigan_en_us_ipa_v2.12.0```

## 1 . Convert .nemo custom to .riva
- Sử dụng thư viện nemo2riva


docker:

```docker run --init -it --rm --gpus '"device=4"' -v riva-model-repo:/data -e MODEL_DEPLOY_KEY=tlt_encode --name riva-service-maker nvcr.io/nvidia/riva/riva-speech:2.12.0-servicemaker deploy_all_models /data/rmir /datamodels ```

Note: riva-model-repo set-up in ./riva_quickstart/config.sh

## 1 . Convert .nemo to .riva
- Sử dụng thư viện nemo2riva
-

## 3. Deploy one file .rmir
soon
