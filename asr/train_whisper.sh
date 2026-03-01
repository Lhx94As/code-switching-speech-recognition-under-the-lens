#!/usr/bin/env bash
cmd="/home3/chenchen/research/maison2/egs/VB/slurm.pl --quiet --nodelist=node01 --gpu 1"

source activate whisper


$cmd log/train_whisper_medium_asru.log \
python train_whisper.py \
      --save_every 40000 \
      --lr 0.00001 \
      --module 'all' \
      --model 'openai/whisper-medium' \
      --batch 2 \
      --accumulation 8 \
      --epochs 5 \
      --save_dir 'ckpt/medium_asru_cs/' \
      --train '/home3/hexin/asru_data/data_cs/' \
      --dev '/home3/hexin/espnet/egs2/asru/asr1/data/test/' \
      --print_every 1000