#!/bin/bash

for i in {0..9}; do
    LOAD_CHECKPOINT=0 python3 src/train.py
    python3 src/evaluate.py
done
