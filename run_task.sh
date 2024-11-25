#!/bin/bash
#python -m torch.utils.bottleneck train.py task=cartpole_balance
HYDRA_FULL_ERROR=1 python train.py task=cartpole_balance