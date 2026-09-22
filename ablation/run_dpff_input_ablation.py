"""Run amplitude-only and I/Q-only input ablations without touching prior outputs.

The script reuses the existing fixed train/test split in
model_output/radar_data_processed.pkl.  It preserves the 40-bin TransCIR
CNN--Transformer topology and varies only the DPFF input subset.
"""

import argparse
import json
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


FEATURE_SETS = {
    "amplitude_only": [2],
    "iq_only": [0, 1],
}


class TransCIR(nn.Module):
    """40-bin CNN--Transformer classifier used by the original 40-to-40 run."""

    def __init__(self, input_channels: int, num_classes: int, d_model: int = 64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
        self.pos_encoder = nn.Parameter(torch.randn(1, 40, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * 40, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # [B, features, 40]
        x = self.enc1(x)
        x = self.enc2(x)
        x = x.permute(0, 2, 1)  # [B, 40, 64]
        x = self.transformer(x + self.pos_encoder)
        x = x.permute(0, 2, 1)
        return self.classifier(x)


def encode_labels(y_train: np.ndarray, y_test: np.ndarray):
    labels = sorted({tuple(np.round(row).astype(int)) for row in y_train})
    label_to_id = {label: index for index, label in enumerate(labels)}
    y_train_ids = np.array(
        [label_to_id[tuple(np.round(row).astype(int))] for row in y_train], dtype=np.int64
    )
    y_test_ids = np.array(
        [label_to_id[tuple(np.round(row).astype(int))] for row in y_test], dtype=np.int64
    )
    return labels, y_train_ids, y_test_ids


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_and_evaluate(
    name: str,
    feature_indices: list[int],
    x_train: np.ndarray,
    y_train_ids: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[tuple[int, ...]],
    device: torch.device,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    seed: int,
):
    set_seed(seed)
    x_train_tensor = torch.from_numpy(x_train[:, :, feature_indices]).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_ids).long().to(device)
    x_test_tensor = torch.from_numpy(x_test[:, :, feature_indices]).float().to(device)

    model = TransCIR(len(feature_indices), len(labels)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )
    criterion = nn.CrossEntropyLoss()

    print(f"\n=== {name}: features {feature_indices}, device {device} ===")
    start_time = time.time()
    num_samples = len(x_train_tensor)
    history = []

    model.train()
    for epoch in range(1, epochs + 1):
        indices = torch.randperm(num_samples, device=device)
        total_loss = 0.0
        correct = 0

        for start in range(0, num_samples, batch_size):
            batch_indices = indices[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(x_train_tensor[batch_indices])
            loss = criterion(logits, y_train_tensor[batch_indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_indices)
            correct += (logits.argmax(dim=1) == y_train_tensor[batch_indices]).sum().item()

        average_loss = total_loss / num_samples
        train_accuracy = 100.0 * correct / num_samples
        scheduler.step(average_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": average_loss,
                "train_accuracy": train_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:03d}/{epochs} | loss {average_loss:.4f} | "
                f"train acc {train_accuracy:.2f}% | "
                f"lr {optimizer.param_groups[0]['lr']:.6f} | {elapsed:.0f}s"
            )

    model.eval()
    prediction_ids = []
    with torch.no_grad():
        for start in range(0, len(x_test_tensor), 1024):
            prediction_ids.append(model(x_test_tensor[start : start + 1024]).argmax(dim=1).cpu().numpy())
    prediction_ids = np.concatenate(prediction_ids)
    prediction_vectors = np.asarray([labels[index] for index in prediction_ids], dtype=np.int64)
    exact_matches = np.all(np.round(y_test).astype(np.int64) == prediction_vectors, axis=1)
    exact_match_accuracy = float(exact_matches.mean() * 100.0)

    torch.save(model.state_dict(), output_dir / f"{name}_model.pth")
    with open(output_dir / f"{name}_inference_results.pkl", "wb") as handle:
        pickle.dump(
            {
                "y_test": y_test,
                "y_pred": prediction_vectors,
                "y_pred_ids": prediction_ids,
                "exact_matches": exact_matches,
            },
            handle,
        )
    with open(output_dir / f"{name}_training_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    return {
        "variant": name,
        "feature_indices": feature_indices,
        "input_channels": len(feature_indices),
        "exact_match_accuracy": exact_match_accuracy,
        "correct_frames": int(exact_matches.sum()),
        "total_frames": int(len(exact_matches)),
        "elapsed_seconds": time.time() - start_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", choices=FEATURE_SETS, default=list(FEATURE_SETS))
    parser.add_argument("--data-path", type=Path, default=Path("model_output/radar_data_processed.pkl"))
    parser.add_argument("--output-root", type=Path, default=Path("model_output/dpff_input_ablation"))
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Processed data not found: {args.data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is unavailable; this ablation script is configured for GPU execution.")

    with open(args.data_path, "rb") as handle:
        data = pickle.load(handle)
    x_train, y_train = data["X_train"], data["y_train"]
    x_test, y_test = data["X_test"], data["y_test"]
    if x_train.shape[1:] != (40, 4) or x_test.shape[1:] != (40, 4):
        raise ValueError(f"Expected [N, 40, 4] DPFF tensors, got {x_train.shape} and {x_test.shape}")

    labels, y_train_ids, _ = encode_labels(y_train, y_test)
    run_dir = args.output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "d_model": 64,
        "encoder_layers": 2,
        "attention_heads": 4,
        "feedforward_dimension": 128,
        "transformer_dropout": 0.1,
        "classifier_dropout": 0.2,
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    results = []
    for variant in args.variants:
        results.append(
            train_and_evaluate(
                variant,
                FEATURE_SETS[variant],
                x_train,
                y_train_ids,
                x_test,
                y_test,
                labels,
                device,
                run_dir,
                args.epochs,
                args.batch_size,
                args.seed,
            )
        )
    with open(run_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("\n=== Exact-match summary ===")
    for result in results:
        print(f"{result['variant']}: {result['exact_match_accuracy']:.2f}%")
    print(f"All new artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
