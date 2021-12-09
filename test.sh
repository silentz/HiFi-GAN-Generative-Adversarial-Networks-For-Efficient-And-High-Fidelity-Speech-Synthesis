#!/bin/bash

latest_file=$(ls -t1 ./checkpoints | head -n 1)

python trainer.py validate \
    --config config/overfit.yaml \
    --ckpt_path "./checkpoints/$latest_file"

