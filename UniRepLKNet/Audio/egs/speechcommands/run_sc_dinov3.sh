#!/bin/bash
# SpeechCommands with the DINOv3 backbone and the custom AudioDINO loss (see audio_loss.py at the repo root).
# Run from this folder: bash run_sc_dinov3.sh

model=unireplknet_dinov3
dataset=speechcommands
bal=none
lr=1e-4
epoch=30
freqm=48
timem=48
mixup=0.6
batch_size=64

dataset_mean=-6.845978
dataset_std=5.5654526
audio_length=128        # must be a multiple of 16
noise=True

metrics=acc
loss=AudioDINO
warmup=False
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.85

# backbone: dinov3_vits16 / dinov3_vitb16 / dinov3_vitl16, ckpt empty = hub default weights
backbone_arch=dinov3_vits16
drop_path_rate=0.1

# AudioDINO loss
gamma_pos=1
gamma_neg=3
margin=0.05
smoothing=0.05
koleo_weight=0.1
gram_weight=1           # 0 disables Gram anchoring and the frozen teacher (halves the backbone memory)

tr_data=./data/datafiles/speechcommand_train_data.json
val_data=./data/datafiles/speechcommand_valid_data.json
eval_data=./data/datafiles/speechcommand_eval_data.json
exp_dir=./exp/test-${dataset}-${backbone_arch}-b$batch_size-lr${lr}-AudioDINO

python ./prep_sc.py

python -W ignore ../../src/run.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${val_data} --data-eval ${eval_data} --exp-dir $exp_dir \
--label-csv ./data/speechcommands_class_labels_indices.csv --n_class 35 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --audio_length ${audio_length} --noise ${noise} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--backbone_arch ${backbone_arch} --drop_path_rate ${drop_path_rate} \
--gamma_pos ${gamma_pos} --gamma_neg ${gamma_neg} --margin ${margin} --smoothing ${smoothing} \
--koleo_weight ${koleo_weight} --gram_weight ${gram_weight}
