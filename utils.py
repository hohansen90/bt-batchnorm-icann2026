# utils.py

from pathlib import Path

import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets import Flowers102, DTD, OxfordIIITPet, StanfordCars


class Augmentations:
    """Image preprocessing and optional training augmentation used in the paper."""

    def __init__(self, use_horizontal_flip=False):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip() if use_horizontal_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __call__(self, img):
        return self.transform(img)


class ResNet50_Backbone(torch.nn.Module):
    """ResNet-50 backbone returning 2048-dimensional pooled features."""

    def __init__(self, bt=True):
        super().__init__()

        if bt:
            model = torch.hub.load("facebookresearch/barlowtwins:main", "resnet50")
        else:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0",
                "resnet50",
                pretrained=True,
            )

        self.features = torch.nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.flatten(1)


def set_seed(seed, device=None):
    """Set random seeds for reproducibility."""

    torch.manual_seed(seed)

    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def get_dataset(name, data_root):
    """Load one of the downstream datasets used in the paper."""

    data_root = Path(data_root)

    if name == "flowers":
        data_train = Flowers102(
            root=data_root,
            split="train",
            transform=Augmentations(use_horizontal_flip=True),
            download=True,
        )
        data_val = Flowers102(
            root=data_root,
            split="val",
            transform=Augmentations(use_horizontal_flip=False),
            download=True,
        )
        return data_train, data_val, 102, "Flowers102"

    if name == "dtd":
        data_train = DTD(
            root=data_root,
            split="train",
            transform=Augmentations(use_horizontal_flip=True),
            download=True,
        )
        data_val = DTD(
            root=data_root,
            split="val",
            transform=Augmentations(use_horizontal_flip=False),
            download=True,
        )
        return data_train, data_val, 47, "DTD"

    if name == "pets":
        target_transform = lambda y: y - 1

        full_data = OxfordIIITPet(
            root=data_root,
            split="trainval",
            download=True,
            target_transform=target_transform,
        )

        indices = torch.randperm(
            len(full_data),
            generator=torch.Generator().manual_seed(42),
        )

        train_size = int(0.8 * len(full_data))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        data_train = torch.utils.data.Subset(
            OxfordIIITPet(
                root=data_root,
                split="trainval",
                transform=Augmentations(use_horizontal_flip=True),
                target_transform=target_transform,
                download=True,
            ),
            train_indices,
        )

        data_val = torch.utils.data.Subset(
            OxfordIIITPet(
                root=data_root,
                split="trainval",
                transform=Augmentations(use_horizontal_flip=False),
                target_transform=target_transform,
                download=True,
            ),
            val_indices,
        )

        return data_train, data_val, 37, "Pets"

    if name == "cars":
        data_train = StanfordCars(
            root=data_root,
            split="train",
            transform=Augmentations(use_horizontal_flip=True),
            download=True,
        )
        data_val = StanfordCars(
            root=data_root,
            split="test",
            transform=Augmentations(use_horizontal_flip=False),
            download=True,
        )
        return data_train, data_val, 196, "Cars"

    raise ValueError(f"Unknown dataset: {name}")


def get_backbone(pretraining, device):
    """Load a pretrained ResNet-50 backbone and optionally apply reparameterization."""

    if pretraining in ["bt", "bt_norm"]:
        backbone = ResNet50_Backbone(bt=True).to(device)
    elif pretraining in ["imgnet", "imgnet_norm"]:
        backbone = ResNet50_Backbone(bt=False).to(device)
    else:
        raise ValueError(f"Unknown pretraining variant: {pretraining}")

    if pretraining.endswith("_norm"):
        backbone.load_state_dict(network_normalizer(backbone))

    return backbone


def network_normalizer(model):
    """
    Incorporate BatchNorm running variances into the preceding convolutional weights.

    For each Conv-BN pair, this applies

        W' = W / sqrt(running_var + eps)
        running_mean' = running_mean / sqrt(running_var + eps)
        running_var' = 1

    The BatchNorm affine parameters gamma and beta are left unchanged.
    """

    state_dict = model.state_dict()
    conv_bn_pairs = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_bn_pairs.append([name])

        elif isinstance(module, torch.nn.BatchNorm2d):
            if not conv_bn_pairs:
                raise RuntimeError(
                    f"Found BatchNorm layer {name} before any convolution."
                )
            conv_bn_pairs[-1].append(name)

    for pair in conv_bn_pairs:
        if len(pair) != 2:
            raise RuntimeError(
                f"Expected Conv-BN pair, but got: {pair}"
            )

        conv_name, bn_name = pair

        running_var = state_dict[f"{bn_name}.running_var"]
        running_mean = state_dict[f"{bn_name}.running_mean"]

        scale = torch.sqrt(running_var + 1e-10)

        state_dict[f"{conv_name}.weight"] = (
            state_dict[f"{conv_name}.weight"] / scale[:, None, None, None]
        )

        state_dict[f"{bn_name}.running_mean"] = running_mean / scale
        state_dict[f"{bn_name}.running_var"] = torch.ones_like(running_var)

    return state_dict


def freeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad = True


def freeze_backbone_except_batchnorm(model):
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for param in module.parameters():
                param.requires_grad = True


def get_trainable_batchnorm_parameters(model):
    params = []

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            params.extend(list(module.parameters()))

    return params


class Logger:
    """Simple CSV logger for training and validation metrics."""

    def __init__(self, tracklist, path="loss.csv"):
        self.lossdict = {name: [] for name in tracklist}
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.lossdict:
                raise KeyError(f"Unknown logging key: {key}")
            self.lossdict[key].append(value)

    def save(self):
        pd.DataFrame(self.lossdict).to_csv(self.path, index=False)
