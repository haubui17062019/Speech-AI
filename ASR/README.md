# Training ASR with Nemo and Deploy using Nemo Riva


## I. Prapare Data
- Sử dụng bộ VLSP 2020: https://www.kaggle.com/datasets/tuannguyenvananh/vin-big-data-vlsp-2020-100h
- Coding xử lý dataset với bộ VLSP 2020 chuẩn format
```bash
python ./prapare_data/prepare_data.py
```
- Với bộ bud500
```bash
python ./prapare_data/prepare_data_bud500.py
```

## II. Training 
- Git clone source
```bash
git clone https://github.com/NVIDIA/NeMo FIX_ME/path/to/NeMo
```

- Create the tokenizer
```bash
python FIX_ME/path/to/NeMo/scripts/tokenizers/process_asr_text_tokenizer.py \
                --manifest=prapare_data/train_manifest.json \
                --data_root=prapare_data/vlsp2020_train_set_02 \
                --vocab_size=105 \
                --tokenizer=spe \
                --spe_type=unigram
```

- Training Conformer-CTC
```bash
python FIX_ME/path/to/NeMo/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
    --config-path=../conf/conformer/ --config-name=conformer_ctc_bpe \
    +init_from_pretrained_model=stt_en_conformer_ctc_large \
    model.train_ds.manifest_filepath=prapare_data/train_manifest.json \
    model.validation_ds.manifest_filepath=prapare_data/test_manifest.json \
    model.tokenizer.dir=prapare_data/tokenizer_spe_unigram_v128 \
    trainer.devices=1 \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.weight_decay=0.001 \
    model.optim.sched.warmup_steps=2000 \
    ++exp_manager.exp_dir=checkpoints \
    ++exp_manager.version=test \
    ++exp_manager.use_datetime_version=False
```

## III. Deploy Riva
NOTE: Hiện tại đang dùng docker theo riva-quickstart 2.15.0, có thể download riva-quickstart từ link sau
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/resources/riva_quickstart
### 1. Nemo2Riva
- Cần convert ckpt đuôi ".nemo" sang ".riva"
- Cài đặt thư viện nemo2riva: ```pip install nemo2riva```
- Chạy đoạn code sau
```bash
nemo2riva -out ./riva_model/Conformer-CTC-BPE.riva ./checkpoints/Conformer-CTC-BPE/test/checkpoints/Conformer-CTC-BPE.nemo
```

### 2. Riva2RMIR
- Chạy lệnh sau
```bash
docker run --init -it --rm --gpus '"device=2"' -v /home1/data/haubui/Speech-AI/ASR/deploy:/data \
-e MODEL_DEPLOY_KEY=tlt_encode --name riva-service-maker nvcr.io/nvidia/riva/riva-speech:2.12.0-servicemaker
```

- Convert ckpt ".riva" sang ".rmir"
```bash
riva-build speech_recognition \
  /data/rmir/asr_vn.rmir:tlt_encode \
  /data/riva_model/Conformer-CTC-BPE-VN.riva:tlt_encode \
  --name=conformer-vi-VN-asr-streaming \
  --return_separate_utterances=False \
  --featurizer.use_utterance_norm_params=False \
  --featurizer.precalc_norm_time_steps=0 \
  --featurizer.precalc_norm_params=False \
  --ms_per_timestep=40 \
  --endpointing.start_history=200 \
  --nn.fp16_needs_obey_precision_pass \
  --endpointing.residue_blanks_at_start=-2 \
  --chunk_size=0.16 \
  --left_padding_size=1.92 \
  --right_padding_size=1.92 \
  --decoder_type=greedy \
  --greedy_decoder.asr_model_delay=-1 \
  --language_code=vi-VN \
  --wfst_tokenizer_model=<far_tokenizer_file> \
  --wfst_verbalizer_model=<far_verbalizer_file> \
  --speech_hints_model=<far_speech_hints_file>
```
- Deploy rmir to models
```bash
riva-deploy -f <file.rmir> /data/models
```

### 3. Run Riva
1. Run api riva
- Bởi vì đang chạy model custom, set các biến trong file: [config.sh](./riva_quickstart/config.sh)
```
service_enabled_asr=true
service_enabled_nlp=false
service_enabled_tts=false
service_enabled_nmt=false

riva_model_loc="/home1/data/haubui/Speech-AI/ASR/deploy"
```
-> riva_model_loc: chắc chắn rằng khi chạy trong docker sẽ có folder chứa "models" hay nói cách khác đây là path dùng 
để mount trong docker khi convert model.

- Chạy API Riva
```bash
bash riva_quicstart/riva_start.sh
``` 
NOTE: riva_model_loc="/home1/data/haubui/Speech-AI/ASR/deploy" in riva_quickstart/config.sh

2. Stop api riva
```bash
bash riva_quicstart/riva_stop.sh
```

## IV. Inference
- riva-quickstart cung cấp các mã nguồn chạy infer từ riva
- Khi dùng trong các tác vụ đặc biệt nên custom lại



