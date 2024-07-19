Training ASR with Nemo 

# 1. Prapare Data
- Sử dụng bộ VLSP 2020: https://www.kaggle.com/datasets/tuannguyenvananh/vin-big-data-vlsp-2020-100h
- Run:
```python prapare_data/prepare_data.py```

# 2. Training 
- Git clone source
```git clone https://github.com/NVIDIA/NeMo FIX_ME/path/to/NeMo```

- Create the tokenizer
<!-- ```python FIX_ME/path/to/NeMo/scripts/tokenizers/process_asr_text_tokenizer.py \
                --manifest=prapare_data/train_manifest.json \
                --data_root=prapare_data/vlsp2020_train_set_02 \
                --vocab_size=128 \
                --tokenizer=spe \
                --spe_type=unigram
                ``` -->

- Training Conformer-CTC
<!--```python FIX_ME/path/to/NeMo/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
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
    ++exp_manager.use_datetime_version=False``` -->

# 3. Deploy Riva

## 1. Nemo2Riva
```nemo2riva -out ./riva_model/Conformer-CTC-BPE.riva ./checkpoints/Conformer-CTC-BPE/test/checkpoints/Conformer-CTC-BPE.nemo```

## 2. Riva2RMIR
1. Run docker
```docker run --init -it --rm --gpus '"device=0"' -v /home1/data/haubui/Speech-AI/ASR/deploy:/data -e MODEL_DEPLOY_KEY=tlt_encode --name riva-service-maker nvcr.io/nvidia/riva/riva-speech:2.12.0-servicemaker```

2. Run code 
 <!--```riva-build speech_recognition \
  /data/rmir/asr_vn.rmir:tlt_encode \
  /data/Conformer-CTC-BPE-v13.riva:tlt_encode \
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
```-->

3. Deploy rmir to models
```riva-deploy -f <file.rmir> /data/models```

## 3. Run Riva
1. Run api riva
```bash riva_quicstart/riva_start.sh``` 
NOTE: riva_model_loc="/home1/data/haubui/Speech-AI/ASR/deploy" in riva_quickstart/config.sh

2. Stop api riva
```bash riva_quicstart/riva_stop.sh```