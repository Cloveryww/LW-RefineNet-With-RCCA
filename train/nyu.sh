#!/bin/sh
PYTHONPATH=$(pwd):$PYTHONPATH python src/train_res152.py \
    --enc 152
