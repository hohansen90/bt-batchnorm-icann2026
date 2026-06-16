#!/bin/bash
set -e

python main.py --dataset flowers --pretraining bt --mode finetune --seed 0
python main.py --dataset cars --pretraining imgnet_norm --mode lp_bn --seed 42

# Full reproduction of all experiments reported in the paper:
python main.py --dataset all --pretraining all --mode all --all-seeds

# Collect summaries:
python main.py --collect-results