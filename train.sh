#!/bin/bash

python trainer.py fit \
    --config config/stage1.yaml

latest_file=$(ls -t1 ./checkpoints | head -n 1)

python trainer.py fit \
    --config config/stage2.yaml \
    --ckpt_path "./checkpoints/$latest_file"
