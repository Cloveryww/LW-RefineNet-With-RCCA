#!/bin/sh
PYTHONPATH=$(pwd):$PYTHONPATH python src/train_coco.py \
    --enc 152
