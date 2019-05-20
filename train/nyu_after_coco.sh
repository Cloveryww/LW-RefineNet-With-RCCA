#!/bin/sh
PYTHONPATH=$(pwd):$PYTHONPATH python src/train_after_coco.py \
    --enc 152
