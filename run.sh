#!/bin/bash

for i in {0..9}; do
    rm -rf checkpoint/
    python3 src/train.py
    python3 src/evaluate.py
done
