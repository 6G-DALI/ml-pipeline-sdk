"""Train a small torch network and report per-epoch metrics.

Resumable: the checkpoint is a torch ``state_dict`` plus the number of epochs done. With a base version the
weights are loaded and training continues, epoch numbers carry on from the base.
"""

import numpy as np
import pandas as pd
from zenml import step

from dali_mlops import (
    Checkpoint, ExportedModel, SDKError, TrainingOutput, dali_pipeline, dump_torch_state, load_torch_state,
)
from dali_mlops.onnx import export_torch_onnx


@step
def fit(
    df: pd.DataFrame, base: Checkpoint | None, target: str, epochs: int, learning_rate: float,
) -> tuple[ExportedModel, dict, list, Checkpoint]:
    import torch

    features = torch.tensor(df.drop(columns=[target]).to_numpy(dtype=np.float32))
    labels = torch.tensor(df[target].to_numpy(dtype=np.float32)).unsqueeze(1)
    torch.manual_seed(0)
    network = torch.nn.Sequential(torch.nn.Linear(features.shape[1], 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    done = 0
    if base is not None:
        state = load_torch_state(base)
        if state.get("feature_count") != features.shape[1]:
            raise SDKError(
                f"The base model was trained on {state.get('feature_count')} features but the new dataset "
                f"provides {features.shape[1]}. Choose a dataset with the same schema or start from scratch."
            )
        network.load_state_dict(state["weights"])
        done = int(state["epochs_done"])
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    history = []
    for epoch in range(done, done + epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(network(features), labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mae = (network(features) - labels).abs().mean()
        history.append({"epoch": epoch, "loss": float(loss), "mae": float(mae)})
    exported = export_torch_onnx(network, features[:1], input_name="features")
    checkpoint = dump_torch_state(
        {"weights": network.state_dict(), "epochs_done": done + epochs, "feature_count": int(features.shape[1])},
    )
    return exported, {key: value for key, value in history[-1].items() if key != "epoch"}, history, checkpoint


@dali_pipeline(dataset="csv", resumable=True, requirements=["torch>=2.3", "onnx>=1.17,<2"])
def epoch_mlp_pipeline(
    df: pd.DataFrame, base: Checkpoint | None = None, target: str = "target", epochs: int = 10,
    learning_rate: float = 0.01,
):
    model, metrics, history, checkpoint = fit(df, base, target, epochs, learning_rate)
    return TrainingOutput(model, metrics, history, checkpoint)
