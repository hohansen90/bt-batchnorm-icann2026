# Batch Normalization in Barlow Twins Pretraining

This repository contains the code for the experiments reported in:

**Batch Normalization Shapes Representations in Barlow Twins Pretraining**
Hans-Oliver Hansen and Thomas Martinetz
ICANN 2026

The experiments compare ResNet-50 models pretrained with Barlow Twins and supervised ImageNet pretraining. We evaluate how incorporating BatchNorm running variances into convolutional weights affects downstream transfer learning.

## Repository structure

```text
.
├── main.py              # Command-line entry point for all experiments
├── utils.py             # Dataset loading, model loading, reparameterization, logging
├── run_all.sh           # Script for running all paper experiments
├── requirements.txt     # Python dependencies
└── README.md
```

During execution, the following directories may be created:

```text
data/        # Downloaded datasets, depending on --data-root
features/    # Cached features for linear probing
results/     # Per-run CSV logs
summaries/   # Aggregated result summaries
```

## Installation

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

The pretrained ResNet-50 checkpoints are loaded via `torch.hub`:

* Barlow Twins: `facebookresearch/barlowtwins:main`
* Supervised ImageNet: `pytorch/vision:v0.10.0`

An internet connection is therefore required when the checkpoints are loaded for the first time.

## Datasets

The experiments use the following downstream datasets:

* Oxford Flowers102
* Describable Textures Dataset (DTD)
* Oxford-IIIT Pets
* Stanford Cars

Datasets are downloaded automatically through `torchvision` where available. By default, datasets are stored under:

```bash
../data/data
```

A different location can be specified with:

```bash
--data-root /path/to/data
```

## Running individual experiments

The general command format is:

```bash
python main.py --dataset DATASET --pretraining PRETRAINING --mode MODE --seed SEED
```

Available datasets:

```text
flowers, dtd, pets, cars
```

Available pretraining variants:

```text
bt, bt_norm, imgnet, imgnet_norm
```

Available training modes:

```text
finetune, lp, lp_bn
```

Examples:

```bash
python main.py --dataset flowers --pretraining bt --mode finetune --seed 0
python main.py --dataset cars --pretraining imgnet_norm --mode lp_bn --seed 42
python main.py --dataset pets --pretraining bt_norm --mode lp --seed 1234
```

The default hyperparameters match the paper:

```text
optimizer:      Adam
learning rate:  1e-3
weight decay:   0
scheduler:      none
batch size:     32
epochs:         50
seeds:          0, 42, 1234, 2024, 9999
```

## Training protocols

The code supports three transfer learning protocols:

### Full finetuning

```bash
--mode finetune
```

The full backbone and classification head are optimized. BatchNorm affine parameters and running statistics are updated during training.

### Linear probing

```bash
--mode lp
```

The backbone is kept frozen and evaluated using its stored BatchNorm statistics. Features are extracted once and cached. Only the linear classifier is trained.

### Linear probing with BatchNorm adaptation

```bash
--mode lp_bn
```

The convolutional backbone is frozen, while the linear classifier and BatchNorm affine parameters are optimized. BatchNorm running statistics are updated during training.

## Reparameterized model variants

The variants `bt_norm` and `imgnet_norm` apply the BatchNorm running-variance reparameterization described in the paper.

For each Conv-BN pair, the code applies:

```text
W' = W / sqrt(running_var + eps)
running_mean' = running_mean / sqrt(running_var + eps)
running_var' = 1
```

The BatchNorm affine parameters are left unchanged.

## Reproducing all experiments

To run all experiments reported in the paper:

```bash
bash run_all.sh
```

Alternatively, the complete experiment grid can be started directly with:

```bash
python main.py --dataset all --pretraining all --mode all --all-seeds
```

This runs all combinations of:

```text
4 datasets × 4 pretraining variants × 3 training protocols × 5 seeds
```

After all experiments have finished, aggregate the results with:

```bash
python main.py --collect-results
```

The summary CSV files are written to:

```text
summaries/
```

## Output format

Each run writes a CSV file containing:

```text
train_loss, train_acc, val_loss, val_acc
```

The reported numbers in the paper correspond to the mean and standard deviation over five seeds of the best validation accuracy achieved during training.

## Notes

Linear probing features are cached in the `features/` directory. Remove this directory to force feature extraction to run again.

Result files are written to:

```text
results/<Dataset>/<Pretraining>/
```

For example:

```text
results/Flowers102/bt/finetune_0.csv
results/Cars/imgnet_norm/lp_bn_42.csv
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{hansen2026batchnorm,
  title     = {Batch Normalization Shapes Representations in Barlow Twins Pretraining},
  author    = {Hansen, Hans-Oliver and Martinetz, Thomas},
  booktitle = {International Conference on Artificial Neural Networks},
  year      = {2026}
}
```
