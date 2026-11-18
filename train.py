import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import load_dataset, split_samples
from model import BehavioralCloningCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN to regress two driving-control targets."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data") / "Canny_Preprocess.mat",
        help="MATLAB file containing X and y arrays.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="", help="For example: cpu, cuda, or cuda:0.")
    return parser.parse_args()


def save_plots(
    output_dir: Path,
    training_losses: List[float],
    validation_losses: List[float],
    validation_mae: List[Tuple[float, float]],
) -> None:
    epochs = range(1, len(training_losses) + 1)

    figure, axis = plt.subplots()
    axis.plot(epochs, training_losses, label="Training MSE")
    axis.plot(epochs, validation_losses, label="Validation MSE")
    axis.set(xlabel="Epoch", ylabel="MSE (standardized targets)", title="Training loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss.png")
    plt.close(figure)

    mae_values = np.asarray(validation_mae)
    figure, axis = plt.subplots()
    axis.plot(epochs, mae_values[:, 0], label="Control target 1")
    axis.plot(epochs, mae_values[:, 1], label="Control target 2")
    axis.set(xlabel="Epoch", ylabel="Mean absolute error", title="Validation error")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "validation-mae.png")
    plt.close(figure)


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs and batch size must be positive; learning rate must be > 0.")

    torch.manual_seed(args.seed)
    images, targets = load_dataset(args.data)
    train_images, train_targets, val_images, val_targets = split_samples(
        images, targets, args.validation_fraction
    )

    target_mean = train_targets.mean(axis=0)
    target_std = train_targets.std(axis=0)
    target_std[target_std == 0.0] = 1.0
    normalized_train_targets = (train_targets - target_mean) / target_std
    normalized_val_targets = (val_targets - target_mean) / target_std

    training_data = TensorDataset(
        torch.from_numpy(train_images),
        torch.from_numpy(normalized_train_targets.astype(np.float32)),
    )
    validation_data = TensorDataset(
        torch.from_numpy(val_images),
        torch.from_numpy(normalized_val_targets.astype(np.float32)),
    )
    train_loader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BehavioralCloningCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = torch.nn.MSELoss()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_losses: List[float] = []
    validation_losses: List[float] = []
    validation_mae: List[Tuple[float, float]] = []
    best_validation_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_squared_error = 0.0
        train_sample_count = 0
        for batch_images, batch_targets in train_loader:
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_images)
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            train_squared_error += loss.item() * len(batch_images)
            train_sample_count += len(batch_images)
        training_loss = train_squared_error / train_sample_count

        model.eval()
        validation_squared_error = 0.0
        validation_sample_count = 0
        absolute_error = np.zeros(2, dtype=np.float64)
        with torch.no_grad():
            for batch_images, batch_targets in validation_loader:
                batch_images = batch_images.to(device)
                batch_targets = batch_targets.to(device)
                predictions = model(batch_images)
                loss = loss_function(predictions, batch_targets)
                validation_squared_error += loss.item() * len(batch_images)
                validation_sample_count += len(batch_images)

                predictions_original = (
                    predictions.cpu().numpy() * target_std + target_mean
                )
                targets_original = batch_targets.cpu().numpy() * target_std + target_mean
                absolute_error += np.abs(predictions_original - targets_original).sum(axis=0)

        validation_loss = validation_squared_error / validation_sample_count
        mae = tuple((absolute_error / validation_sample_count).tolist())
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        validation_mae.append(mae)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs}: "
            f"train_mse={training_loss:.5f} "
            f"val_mse={validation_loss:.5f} "
            f"val_mae=({mae[0]:.4f}, {mae[1]:.4f})"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_mean": target_mean.tolist(),
                    "target_std": target_std.tolist(),
                    "input_shape": list(images.shape[1:]),
                },
                output_dir / "best_model.pt",
            )

    save_plots(output_dir, training_losses, validation_losses, validation_mae)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "samples": len(images),
                "training_samples": len(train_images),
                "validation_samples": len(val_images),
                "best_validation_mse": best_validation_loss,
                "final_validation_mae": list(validation_mae[-1]),
                "target_mean": target_mean.tolist(),
                "target_std": target_std.tolist(),
            },
            file,
            indent=2,
        )
    print(f"Saved model, metrics, and plots to {output_dir.resolve()}")


if __name__ == "__main__":
    train(parse_args())
