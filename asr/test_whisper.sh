#!/usr/bin/env bash
cmd="/home3/chenchen/research/maison2/egs/VB/slurm.pl --quiet --nodelist=node03 --gpu 1"

source activate whisper


$cmd log/test_asru_whisper_large_v3_aug_0.5.log \
python test_whisper.py \
    --lang "zh" \
    --model "openai/whisper-large-v3" \
    --zeroshot "false" \
    --save_dir "ckpt/asru_large_v3_aug_0.5" \
    --test '/home3/hexin/asru_data/data_test/' \
    # --test '/home3/hexin/asru_data/data_test/' \
    # --test '/home3/hexin/asru_data/data_test/' \