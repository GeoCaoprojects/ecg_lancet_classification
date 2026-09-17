"""
Command-line example:

python attia_ecg_load_and_test_v1.py \
    --device cuda:1 \
    --kfold-data-path ~/projects/04cv/code/attia_kfold_scheme.pt \
    --checkpoint-dir ~/projects/04cv/code/attia_checkpoints \
    --result-path ~/projects/04cv/code/attia_post_training_test_results.pt \
    --excel-result-path ~/projects/04cv/code/attia_post_training_test_results.xlsx \
    --positive-label 1 \
    --batch-size 100 \
    --num-workers 0
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from attia_ecg_training_v1 import (
    AttiaECGNet,
    ECGTrainer,
    TrainConfig,
    encode_binary_labels,
    set_seed,
)

def report_cuda_memory(device):
    if device.type != "cuda":
        return

    device_index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )

    free_bytes, total_bytes = torch.cuda.mem_get_info(
        device_index
    )

    print(
        f"GPU: {torch.cuda.get_device_name(device_index)}"
    )
    print(
        f"Free VRAM: {free_bytes / 1024**3:.2f} GB / "
        f"{total_bytes / 1024**3:.2f} GB"
    )


def validate_cuda_device(
    device,
    minimum_free_gb=1.0,
):
    if device.type != "cuda":
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was requested, but CUDA is not available."
        )

    device_index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )

    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device}, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are visible."
        )

    free_bytes, _ = torch.cuda.mem_get_info(
        device_index
    )

    free_gb = free_bytes / 1024**3

    if free_gb < minimum_free_gb:
        report_cuda_memory(device)

        raise RuntimeError(
            f"GPU {device_index} has only "
            f"{free_gb:.2f} GB free VRAM. "
            f"At least {minimum_free_gb:.2f} GB is required."
        )


def load_model_checkpoint(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model_config = checkpoint.get(
        "model_config",
        {
            "dropout": 0.2,
            "num_classes": 2,
        },
    )

    model = AttiaECGNet(
        dropout=model_config["dropout"],
        num_classes=model_config["num_classes"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
    )

    model.to(DEVICE)
    model.eval()

    return model, checkpoint


def build_label_mapping(fold):
    all_labels = (
        list(fold["training"]["diagnostic_class_id"])
        + list(fold["validation"]["diagnostic_class_id"])
        + list(fold["testing"]["diagnostic_class_id"])
    )

    _, label_mapping = encode_binary_labels(
        all_labels,
        positive_label=POSITIVE_LABEL,
    )

    return label_mapping


def test_one_fold(fold):
    fold_id = int(fold["fold"])

    checkpoint_path = (
        Path(CHECKPOINT_DIR)
        / f"fold_{fold_id}_best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model, checkpoint = load_model_checkpoint(
        checkpoint_path
    )

    label_mapping = build_label_mapping(fold)
    testing = fold["testing"]

    testing_labels = np.asarray(
        [
            label_mapping[label]
            for label in testing["diagnostic_class_id"]
        ],
        dtype=np.int64,
    )

    trainer = ECGTrainer(
        model=model,
        config=TrainConfig(
            batch_size=BATCH_SIZE,
            max_epochs=1,
            num_workers=NUM_WORKERS,
        ),
        device=DEVICE,
    )

    trainer.best_threshold = float(
        checkpoint["best_threshold"]
    )

    metrics, probability, prediction = trainer.evaluate(
        testing,
        testing_labels,
        "testing",
    )

    results = []

    threshold = float(
        checkpoint["best_threshold"]
    )

    for ecg_id, true_label, prob, pred in zip(
        testing["ecg_id"],
        testing_labels,
        probability,
        prediction,
    ):
        results.append(
            {
                "ecg_id": ecg_id,
                "true_label": int(true_label),
                "probability": float(prob),
                "threshold": threshold,
                "prediction": int(pred),
            }
        )

    return {
        "fold": fold_id,
        "checkpoint": str(checkpoint_path),
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_auc": float(checkpoint["best_val_auc"]),
        "threshold": threshold,
        "testing": metrics,
        "testing_results": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Load trained Attia ECG models from five fold checkpoints "
            "and independently test them."
        )
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="PyTorch device, for example cuda:1 or cuda:2.",
    )

    parser.add_argument(
        "--kfold-data-path",
        type=str,
        default="~/projects/04cv/code/attia_kfold_scheme.pt",
        help="Path to the saved k-fold data scheme.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="~/projects/04cv/code/attia_checkpoints",
        help="Directory containing fold_X_best.pt checkpoints.",
    )

    parser.add_argument(
        "--result-path",
        type=str,
        default="~/projects/04cv/code/attia_post_training_test_results.pt",
        help="Path for the post-training testing .pt result.",
    )

    parser.add_argument(
        "--excel-result-path",
        type=str,
        default="~/projects/04cv/code/attia_post_training_test_results.xlsx",
        help="Path for the post-training testing Excel result.",
    )

    parser.add_argument(
        "--positive-label",
        type=int,
        default=1,
        help="diagnostic_class_id encoded as positive class.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Testing batch size.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count for testing.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0.")

    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0.")

    global DEVICE
    global KFOLD_DATA_PATH
    global CHECKPOINT_DIR
    global RESULT_PATH
    global EXCEL_RESULT_PATH
    global POSITIVE_LABEL
    global BATCH_SIZE
    global NUM_WORKERS

    DEVICE = torch.device(args.device)

    KFOLD_DATA_PATH = str(
        Path(args.kfold_data_path)
        .expanduser()
        .resolve()
    )

    CHECKPOINT_DIR = str(
        Path(args.checkpoint_dir)
        .expanduser()
        .resolve()
    )

    RESULT_PATH = str(
        Path(args.result_path)
        .expanduser()
        .resolve()
    )

    EXCEL_RESULT_PATH = str(
        Path(args.excel_result_path)
        .expanduser()
        .resolve()
    )

    POSITIVE_LABEL = args.positive_label
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    if DEVICE.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU was requested, but CUDA is not available."
            )

        if DEVICE.index is None:
            raise ValueError(
                "Please specify an explicit CUDA device, "
                "for example --device cuda:1."
            )

        if DEVICE.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {DEVICE}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )

        print(f"Using device: {DEVICE}")
        print(
            f"GPU: {torch.cuda.get_device_name(DEVICE.index)}"
        )
    else:
        print(f"Using device: {DEVICE}")

    print("\n" + "=" * 80)
    print("POST-TRAINING TEST CONFIGURATION")
    print("=" * 80)
    print(f"DEVICE             = {DEVICE}")
    print(f"KFOLD_DATA_PATH    = {KFOLD_DATA_PATH}")
    print(f"CHECKPOINT_DIR     = {CHECKPOINT_DIR}")
    print(f"RESULT_PATH        = {RESULT_PATH}")
    print(f"EXCEL_RESULT_PATH  = {EXCEL_RESULT_PATH}")
    print(f"POSITIVE_LABEL     = {POSITIVE_LABEL}")
    print(f"BATCH_SIZE         = {BATCH_SIZE}")
    print(f"NUM_WORKERS        = {NUM_WORKERS}")

    set_seed(args.seed)

    if DEVICE.type == "cuda":
        validate_cuda_device(
            DEVICE,
            minimum_free_gb=1.0,
        )
        report_cuda_memory(DEVICE)

    kfold_path = Path(
        KFOLD_DATA_PATH
    )

    if not kfold_path.exists():
        raise FileNotFoundError(
            f"K-fold file not found: {kfold_path}"
        )

    kfold_scheme = torch.load(
        kfold_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(kfold_scheme, list):
        raise ValueError(
            "The loaded k-fold scheme is not a list."
        )

    print(
        f"\nLoaded {len(kfold_scheme)} folds from:"
        f"\n{kfold_path}"
    )

    fold_results = []

    for fold in kfold_scheme:
        result = test_one_fold(fold)
        fold_results.append(result)

        testing = result["testing"]

        print("\n" + "#" * 80)
        print(f"FOLD {result['fold']}")
        print("#" * 80)
        print(
            f"Best epoch: {result['best_epoch']}"
        )
        print(
            f"Best validation AUC: "
            f"{result['best_val_auc']:.6f}"
        )
        print(
            f"Threshold: "
            f"{result['threshold']:.6f}"
        )
        print(
            f"Testing AUC: "
            f"{testing['auc']:.6f}"
        )
        print(
            f"Testing accuracy: "
            f"{testing['accuracy']:.6f}"
        )
        print(
            f"Testing sensitivity: "
            f"{testing['sensitivity']:.6f}"
        )
        print(
            f"Testing specificity: "
            f"{testing['specificity']:.6f}"
        )
        print(
            f"Testing F1: "
            f"{testing['f1']:.6f}"
        )

    output = {
        "fold_results": fold_results,
    }

    result_path = Path(
        RESULT_PATH
    )
    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        output,
        result_path,
    )

    summary_df = pd.DataFrame(
        [
            {
                "fold": result["fold"],
                "best_epoch": result["best_epoch"],
                "best_val_auc": result["best_val_auc"],
                "threshold": result["threshold"],
                **result["testing"],
            }
            for result in fold_results
        ]
    )

    excel_path = Path(
        EXCEL_RESULT_PATH
    )
    excel_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        excel_writer = pd.ExcelWriter(
            excel_path,
            engine="openpyxl",
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Excel export requires openpyxl. "
            "Install it with: python -m pip install openpyxl"
        ) from exc

    with excel_writer as writer:
        summary_df.to_excel(
            writer,
            sheet_name="Overall_Testing",
            index=False,
        )

        for result in fold_results:
            pd.DataFrame(
                result["testing_results"]
            ).to_excel(
                writer,
                sheet_name=f"Fold_{result['fold']}",
                index=False,
            )

    print(
        f"\nSaved PT results to: {result_path}"
    )
    print(
        f"Saved Excel results to: {excel_path}"
    )
    print(
        "\nPost-training test completed."
    )


if __name__ == "__main__":
    main()
