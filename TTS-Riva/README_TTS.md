```
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python fastpitch_finetune.py \
--config-name=fastpitch_align_v1.05.yaml \
train_dataset=dataset/manifest_train_1.json \
validation_datasets=dataset/manifest_val_1.json \
sup_data_path=sup_data \
exp_manager.exp_dir=./checkpoint_tts \
+init_from_pretrained_model="tts_en_fastpitch" \
trainer.check_val_every_n_epoch=10 \
model.train_ds.dataloader_params.batch_size=8 \
model.validation_ds.dataloader_params.batch_size=8 \
model.n_speakers=1 \
model.pitch_mean=125 model.pitch_std=26 \
model.optim.lr=2e-4 \
~model.optim.sched \
model.optim.name=adam \
trainer.devices=1 \
trainer.strategy=auto \
+model.text_tokenizer.add_blank_at=true \
```


```
CUDA_VISIBLE_DEVICES=0 python generate_mels.py \
--fastpitch-model-ckpt checkpoint_tts/FastPitch/2024-07-25_10-51-48/checkpoints/FastPitch--val_loss=0.3460-epoch=19-last.ckpt \
--input-json-manifests dataset/manifest_train.json \
--output-json-manifest-root mel_dir
```


```
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python hifigan_finetune.py \
--config-name=hifigan.yaml \
model.train_ds.dataloader_params.batch_size=8 \
model.optim.lr=0.0001 \
~model.optim.sched \
train_dataset=mel_dir/manifest_train_1_mel.json \
validation_datasets=mel_dir/manifest_val_1_mel.json \
exp_manager.exp_dir=checkpoint_tts \
+init_from_pretrained_model="tts_hifigan" \
trainer.check_val_every_n_epoch=10 \
model/train_ds=train_ds_finetune \
model/validation_ds=val_ds_finetune
```

```
docker run -it --rm --gpus '"device=2"' --name tts -p8186:8000 -p8187:8001 -p8188:8002 \
-v /home1/data/haubui/Speech-AI/TTS-Riva:/data \
nvcr.io/nvidia/tritonserver:22.12-py3
```

```
tritonserver  --model-repository=/data/model_tts/ --model-control-mode=explicit \
--load-model=tts
```