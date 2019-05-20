#!/bin/sh
PYTHONPATH=$(pwd):$PYTHONPATH python src/train.py \
    --enc 101
