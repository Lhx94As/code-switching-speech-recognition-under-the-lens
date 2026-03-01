#!/usr/bin/env bash
cmd="/home3/chenchen/research/maison2/egs/VB/slurm.pl --quiet --nodelist=node05 --gpu 1"

# source activate whisper


$cmd log/train_whisper_asru.log \
python train_whisper.py --save_every 10000 --lr 0.00001