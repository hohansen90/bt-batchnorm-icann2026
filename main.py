# main.py

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm, trange

from utils import (
    Logger,
    freeze_all_parameters,
    freeze_backbone_except_batchnorm,
    get_backbone,
    get_dataset,
    get_trainable_batchnorm_parameters,
    set_seed,
    unfreeze_all_parameters,
)


PAPER_SEEDS = [0, 42, 1234, 2024, 9999]
DATASETS = ["flowers", "dtd", "pets", "cars"]
PRETRAINING_VARIANTS = ["bt", "bt_norm", "imgnet", "imgnet_norm"]
MODES = ["finetune", "lp", "lp_bn"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run downstream transfer experiments for Barlow Twins and "
            "supervised ImageNet-pretrained ResNet-50 models."
        )
    )

    parser.add_argument(
        "--dataset",
        choices=DATASETS + ["all"],
        required=True,
        help="Downstream dataset.",
    )

    parser.add_argument(
        "--pretraining",
        choices=PRETRAINING_VARIANTS + ["all"],
        required=True,
        help="Pretraining variant.",
    )

    parser.add_argument(
        "--mode",
        choices=MODES + ["all"],
        required=True,
        help="Training protocol.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Ignored if --all-seeds is used.",
    )

    parser.add_argument(
        "--all-seeds",
        action="store_true",
        help="Run all five seeds used in the paper.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/data",
        help="Root directory for datasets.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory for result files.",
    )

    parser.add_argument(
        "--features-dir",
        type=str,
        default="features",
        help="Directory for cached linear probing features.",
    )

    parser.add_argument(
        "--collect-results",
        action="store_true",
        help="Only collect existing result CSV files into summary CSV files.",
    )

    parser.add_argument(
        "--summary-dir",
        type=str,
        default="summaries",
        help="Directory for summary CSV files.",
    )

    return parser.parse_args()


def evaluate_model(model, classifier, dataloader, criterion, device):
    model.eval()
    classifier.eval()

    val_loss = 0.0
    val_acc = 0.0
    num_val = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            features = model(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * len(labels)
            val_acc += (outputs.argmax(dim=1) == labels).sum().item()
            num_val += len(labels)

    return val_loss / num_val, val_acc / num_val


def evaluate_feature_classifier(classifier, dataloader, criterion, device):
    classifier.eval()

    val_loss = 0.0
    val_acc = 0.0
    num_val = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = classifier(features)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * len(labels)
            val_acc += (outputs.argmax(dim=1) == labels).sum().item()
            num_val += len(labels)

    return val_loss / num_val, val_acc / num_val


def extract_or_load_features(
    model,
    data_train,
    data_val,
    dataset_name,
    pretraining,
    features_dir,
    batch_size,
    device,
):
    features_dir = Path(features_dir) / dataset_name / pretraining
    features_dir.mkdir(parents=True, exist_ok=True)

    train_path = features_dir / "train_features.pt"
    val_path = features_dir / "val_features.pt"

    if train_path.exists() and val_path.exists():
        print(f"Loading cached features from {features_dir}")

        features_train, labels_train = torch.load(train_path, map_location="cpu", weights_only=True)
        features_val, labels_val = torch.load(val_path, map_location="cpu", weights_only=True)

        return features_train, labels_train, features_val, labels_val

    print(f"Extracting features for {dataset_name} / {pretraining}")

    model.eval()

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False)

    features_train = []
    labels_train = []

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Extracting train features"):
            images = images.to(device)
            features = model(images)

            features_train.append(features.cpu())
            labels_train.append(labels.cpu())

    features_train = torch.cat(features_train)
    labels_train = torch.cat(labels_train)

    features_val = []
    labels_val = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Extracting validation features"):
            images = images.to(device)
            features = model(images)

            features_val.append(features.cpu())
            labels_val.append(labels.cpu())

    features_val = torch.cat(features_val)
    labels_val = torch.cat(labels_val)

    torch.save((features_train, labels_train), train_path)
    torch.save((features_val, labels_val), val_path)

    return features_train, labels_train, features_val, labels_val


def linear_probe(
    model,
    classifier,
    data_train,
    data_val,
    dataset_name,
    pretraining,
    savepath,
    features_dir,
    seed,
    lr,
    weight_decay,
    batch_size,
    n_epochs,
    device,
):
    freeze_all_parameters(model)
    model.eval()

    logger = Logger(
        tracklist=["train_loss", "train_acc", "val_loss", "val_acc"],
        path=Path(savepath) / f"lp_{seed}.csv",
    )

    features_train, labels_train, features_val, labels_val = extract_or_load_features(
        model=model,
        data_train=data_train,
        data_val=data_val,
        dataset_name=dataset_name,
        pretraining=pretraining,
        features_dir=features_dir,
        batch_size=batch_size,
        device=device,
    )

    train_dataset = torch.utils.data.TensorDataset(features_train, labels_train)
    val_dataset = torch.utils.data.TensorDataset(features_val, labels_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in trange(n_epochs, desc="LP epochs"):
        classifier.train()

        train_loss = 0.0
        train_acc = 0.0
        num_train = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_acc += (outputs.argmax(dim=1) == labels).sum().item()
            num_train += len(labels)

        val_loss, val_acc = evaluate_feature_classifier(
            classifier=classifier,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        logger(
            train_loss=train_loss / num_train,
            train_acc=train_acc / num_train,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        print(
            f"Epoch {epoch + 1:03d}/{n_epochs} | "
            f"train_acc={train_acc / num_train:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    logger.save()


def linear_probe_bn(
    model,
    classifier,
    data_train,
    data_val,
    savepath,
    seed,
    lr,
    weight_decay,
    batch_size,
    n_epochs,
    device,
):
    freeze_backbone_except_batchnorm(model)

    logger = Logger(
        tracklist=["train_loss", "train_acc", "val_loss", "val_acc"],
        path=Path(savepath) / f"lp_bn_{seed}.csv",
    )

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(
        list(classifier.parameters()) + get_trainable_batchnorm_parameters(model),
        lr=lr, weight_decay=weight_decay
    )

    for epoch in range(n_epochs):
        model.train()
        classifier.train()

        train_loss = 0.0
        train_acc = 0.0
        num_train = 0

        progress = tqdm(
            train_loader,
            desc=f"LP+BN epoch {epoch + 1}/{n_epochs}",
        )

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            features = model(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_acc += (outputs.argmax(dim=1) == labels).sum().item()
            num_train += len(labels)

            progress.set_postfix(
                train_acc=train_acc / num_train,
                train_loss=train_loss / num_train,
            )

        val_loss, val_acc = evaluate_model(
            model=model,
            classifier=classifier,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        logger(
            train_loss=train_loss / num_train,
            train_acc=train_acc / num_train,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        print(
            f"Epoch {epoch + 1:03d}/{n_epochs} | "
            f"train_acc={train_acc / num_train:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    logger.save()


def full_finetuning(
    model,
    classifier,
    data_train,
    data_val,
    savepath,
    seed,
    lr,
    weight_decay,
    batch_size,
    n_epochs,
    device,
):
    unfreeze_all_parameters(model)

    logger = Logger(
        tracklist=["train_loss", "train_acc", "val_loss", "val_acc"],
        path=Path(savepath) / f"finetune_{seed}.csv",
    )

    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(
        list(classifier.parameters()) + list(model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(n_epochs):
        model.train()
        classifier.train()

        train_loss = 0.0
        train_acc = 0.0
        num_train = 0

        progress = tqdm(
            train_loader,
            desc=f"Finetuning epoch {epoch + 1}/{n_epochs}",
        )

        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            features = model(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            train_acc += (outputs.argmax(dim=1) == labels).sum().item()
            num_train += len(labels)

            progress.set_postfix(
                train_acc=train_acc / num_train,
                train_loss=train_loss / num_train,
            )

        val_loss, val_acc = evaluate_model(
            model=model,
            classifier=classifier,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        logger(
            train_loss=train_loss / num_train,
            train_acc=train_acc / num_train,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        print(
            f"Epoch {epoch + 1:03d}/{n_epochs} | "
            f"train_acc={train_acc / num_train:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    logger.save()


def run_experiment(args, dataset, pretraining, mode, seed, device):
    set_seed(seed, device)

    data_train, data_val, n_classes, dataset_name = get_dataset(
        name=dataset,
        data_root=args.data_root,
    )

    model = get_backbone(
        pretraining=pretraining,
        device=device,
    )

    classifier = torch.nn.Linear(2048, n_classes).to(device)

    savepath = Path(args.results_dir) / dataset_name / pretraining
    savepath.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Dataset:      {dataset_name}")
    print(f"Pretraining:  {pretraining}")
    print(f"Mode:         {mode}")
    print(f"Seed:         {seed}")
    print(f"Epochs:       {args.epochs}")
    print(f"Batch size:   {args.batch_size}")
    print(f"LR:           {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print("=" * 80)

    if mode == "finetune":
        full_finetuning(
            model=model,
            classifier=classifier,
            data_train=data_train,
            data_val=data_val,
            savepath=savepath,
            seed=seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            device=device,
        )

    elif mode == "lp":
        linear_probe(
            model=model,
            classifier=classifier,
            data_train=data_train,
            data_val=data_val,
            dataset_name=dataset_name,
            pretraining=pretraining,
            savepath=savepath,
            features_dir=args.features_dir,
            seed=seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            device=device,
        )

    elif mode == "lp_bn":
        linear_probe_bn(
            model=model,
            classifier=classifier,
            data_train=data_train,
            data_val=data_val,
            savepath=savepath,
            seed=seed,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            device=device,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


def collect_results(results_dir, summary_dir):
    results_dir = Path(results_dir)
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = ["Flowers102", "DTD", "Pets", "Cars"]

    for dataset_name in dataset_names:
        resultdict = {
            "model": [],
            "finetune": [],
            "ft_pm": [],
            "lp": [],
            "lp_pm": [],
            "lp_bn": [],
            "lp_bn_pm": [],
        }

        for pretraining in PRETRAINING_VARIANTS:
            result_dir = results_dir / dataset_name / pretraining

            finetune_values = []
            lp_values = []
            lp_bn_values = []

            if not result_dir.exists():
                print(f"Missing result directory: {result_dir}")
                continue

            for csv_file in sorted(result_dir.glob("*.csv")):
                df = pd.read_csv(csv_file)
                best_val_acc = df["val_acc"].max()

                if csv_file.name.startswith("finetune_"):
                    finetune_values.append(best_val_acc)
                elif csv_file.name.startswith("lp_bn_"):
                    lp_bn_values.append(best_val_acc)
                elif csv_file.name.startswith("lp_"):
                    lp_values.append(best_val_acc)

            resultdict["model"].append(pretraining)

            resultdict["finetune"].append(
                torch.tensor(finetune_values).mean().item()
                if finetune_values else float("nan")
            )
            resultdict["ft_pm"].append(
                torch.tensor(finetune_values).std().item()
                if len(finetune_values) > 1 else float("nan")
            )

            resultdict["lp"].append(
                torch.tensor(lp_values).mean().item()
                if lp_values else float("nan")
            )
            resultdict["lp_pm"].append(
                torch.tensor(lp_values).std().item()
                if len(lp_values) > 1 else float("nan")
            )

            resultdict["lp_bn"].append(
                torch.tensor(lp_bn_values).mean().item()
                if lp_bn_values else float("nan")
            )
            resultdict["lp_bn_pm"].append(
                torch.tensor(lp_bn_values).std().item()
                if len(lp_bn_values) > 1 else float("nan")
            )

        summary = pd.DataFrame(resultdict)
        output_path = summary_dir / f"summary_{dataset_name}.csv"
        summary.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")


def expand_arg(value, all_values):
    if value == "all":
        return all_values
    return [value]


def main():
    args = parse_args()

    if args.collect_results:
        collect_results(
            results_dir=args.results_dir,
            summary_dir=args.summary_dir,
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    datasets = expand_arg(args.dataset, DATASETS)
    pretrainings = expand_arg(args.pretraining, PRETRAINING_VARIANTS)
    modes = expand_arg(args.mode, MODES)
    seeds = PAPER_SEEDS if args.all_seeds else [args.seed]

    for dataset in datasets:
        for pretraining in pretrainings:
            for mode in modes:
                for seed in seeds:
                    run_experiment(
                        args=args,
                        dataset=dataset,
                        pretraining=pretraining,
                        mode=mode,
                        seed=seed,
                        device=device,
                    )


if __name__ == "__main__":
    main()