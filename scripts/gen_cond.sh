#!/bin/bash\

# train full model for conditional generation
python gen/train_full.py --output proj_log/gen_full --batchsize 128 \
    --solid_code pretrained/solid.pkl --profile_code  pretrained/profile.pkl --loop_code pretrained/loop.pkl --mode cond --device 0