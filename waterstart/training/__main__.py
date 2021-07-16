import argparse
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from pyro.distributions.transforms.affine_autoregressive import (  # type: ignore
    affine_autoregressive,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # type: ignore

from ..inference.low_level_engine import LowLevelInferenceEngine, NetworkModules
from ..inference.model import CNN, Emitter, GatedTransition
from .loss import LossEvaluator, LossOutput, NeuralBaseline
from .train_data import TrainDataManager, TrainingData

SAVE_FOLDER_NAME_FORMAT = "%Y%m%d_%H%S%f"


@dataclass
class TraingingArgs(argparse.Namespace):
    train_data_file: str = ""
    iterations: int = 0
    learning_rate: float = 3e-4
    baseline_learning_rate: float = 1e-3
    weight_decay: float = 0.1
    max_gradient_norm: float = 10.0
    detect_anomaly: bool = False
    use_cuda: bool = False
    batch_size: int = 1
    n_samples: int = 10
    window_size: int = 100
    cnn_out_features: int = 256
    max_trades: int = 10
    hidden_state_size: int = 128
    gated_trans_hidden_size: int = 200
    nn_baseline_hidden_size: int = 200
    emitter_hidden_size: int = 200
    n_iafs: int = 2
    iafs_hidden_sizes: list[int] = field(default_factory=lambda: [200])
    leverage: float = 20.0
    initial_balance: float = 1000.0
    tensorboard_log_dir: Optional[str] = None
    checkpoint_interval: int = 100
    checkpoints_dir: str = "./checkpoints"
    load_checkpoint: bool = False


def load_training_data(path: str) -> TrainingData:
    data = np.load(path, allow_pickle=True)  # type: ignore

    # TODO: make names uniform
    # BUG: convert to tensor
    return TrainingData(
        market_data=torch.from_numpy(data["market_data_arr"]),  # type: ignore
        midpoint_prices=torch.from_numpy(data["sym_prices"]),  # type: ignore
        spreads=torch.from_numpy(data["spreads"]),  # type: ignore
        base_to_dep_rates=torch.from_numpy(data["margin_rates"]),  # type: ignore
        quote_to_dep_rates=torch.from_numpy(data["quote_to_dep_rates"]),  # type: ignore
        market_data_blueprint=data["market_data_blueprint"].item(),  # type: ignore
        traded_sym_blueprint_map=data[  # type: ignore
            "traded_sym_blueprint_map"
        ].item(),
        traded_symbols=data["traded_symbols"].tolist(),  # type: ignore
    )


def write_summary(
    writer: SummaryWriter,
    balance: torch.Tensor,
    loss: torch.Tensor,
    grad_norm: torch.Tensor,
    n_iter: int,
):
    step_log_rates = balance[..., 1:].log() - balance[..., :-1].log()
    positive_log_rates = torch.sum(step_log_rates > 0).item()
    negative_log_rates = torch.sum(step_log_rates < 0).item()
    tot_log_rates = step_log_rates.numel()
    writer.add_scalar(  # type: ignore
        "single step/positive fraction", positive_log_rates / tot_log_rates, n_iter
    )
    writer.add_scalar(  # type: ignore
        "single step/negative fraction", negative_log_rates / tot_log_rates, n_iter
    )

    avg_step_gain = step_log_rates.mean(-2).exp_().mean().item()
    writer.add_scalar("single step/average gain", avg_step_gain, n_iter)  # type: ignore

    whole_seq_log_rates = balance[..., -1].log() - balance[..., 0].log()
    positive_log_rates = torch.sum(whole_seq_log_rates > 0).item()
    negative_log_rates = torch.sum(whole_seq_log_rates < 0).item()
    tot_log_rates = whole_seq_log_rates.numel()
    writer.add_scalar(  # type: ignore
        "whole sequence/positive fraction", positive_log_rates / tot_log_rates, n_iter
    )
    writer.add_scalar(  # type: ignore
        "whole sequence/negative fraction", negative_log_rates / tot_log_rates, n_iter
    )

    avg_whole_seq_gain = whole_seq_log_rates.mean(-1).exp_().mean().item()
    writer.add_scalar(  # type: ignore
        "whole sequence/average gain", avg_whole_seq_gain, n_iter
    )

    loss_loc: float = loss.mean().item()  # type: ignore
    loss_scale: float = loss.std(-2).mean().item()  # type: ignore
    writer.add_scalars(  # type: ignore
        "loss with stdev bounds",
        {
            "lower bound": loss_loc - loss_scale,
            "value": loss_loc,
            "upper bound": loss_loc + loss_scale,
        },
        n_iter,
    )

    writer.add_scalar("gradient norm", grad_norm.item(), n_iter)  # type: ignore


def build_modules(
    args: TraingingArgs, training_data: TrainingData
) -> tuple[NetworkModules, NeuralBaseline]:
    cnn = CNN(
        args.batch_size,
        args.window_size,
        training_data.n_market_features,
        args.cnn_out_features,
        training_data.n_traded_sym,
        args.max_trades,
    )
    gated_trans = GatedTransition(
        args.cnn_out_features, args.hidden_state_size, args.gated_trans_hidden_size
    )
    emitter = Emitter(
        args.hidden_state_size, training_data.n_traded_sym, args.emitter_hidden_size
    )

    iafs = [
        affine_autoregressive(args.hidden_state_size, args.iafs_hidden_sizes)
        for _ in range(args.n_iafs)
    ]

    return NetworkModules(
        cnn,
        gated_trans,
        iafs,
        emitter,
        training_data.market_data_arr_mapper,
        training_data.traded_sym_arr_mapper,
        training_data.traded_symbols,
        args.leverage,
    ), NeuralBaseline(
        args.cnn_out_features,
        args.hidden_state_size,
        args.nn_baseline_hidden_size,
        training_data.n_traded_sym,
    )


def get_modules(
    args: TraingingArgs, training_data: TrainingData
) -> tuple[NetworkModules, NeuralBaseline]:
    if not args.load_checkpoint:
        return build_modules(args, training_data)

    checkpoints_dir = Path(args.checkpoint_dir)
    latest_checkpoint_dir: Optional[Path] = None
    latest_dt: datetime.datetime = datetime.datetime.min

    for dir in checkpoints_dir.iterdir():
        if not dir.is_dir():
            continue

        try:
            dt = datetime.datetime.strptime(str(dir), SAVE_FOLDER_NAME_FORMAT)
        except ValueError:
            raise RuntimeError()

        if dt > latest_dt:
            latest_dt = dt
            latest_checkpoint_dir = dir

    if latest_checkpoint_dir is None:
        raise RuntimeError()

    net_modules: NetworkModules = torch.load(  # type: ignore
        latest_checkpoint_dir / "net_modules.pt"
    )
    nn_baseline: NeuralBaseline = torch.load(  # type: ignore
        latest_checkpoint_dir / "nn_baseline.pt"
    )

    return net_modules, nn_baseline


def save_modules(net_modules: NetworkModules, nn_baseline: NeuralBaseline) -> None:
    now = datetime.datetime.now()
    save_dir = args.checkpoints_dir / Path(now.strftime(SAVE_FOLDER_NAME_FORMAT))
    save_dir.mkdir(parents=True)
    torch.save(net_modules, save_dir / "net_modules.pt")  # type: ignore
    torch.save(nn_baseline, save_dir / "nn_baseline.pt")  # type: ignore


def main(args: TraingingArgs):
    training_data = load_training_data(args.train_data_file)
    net_modules, nn_baseline = get_modules(args, training_data)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    device = torch.device("cuda" if args.use_cuda else "cpu")
    net_modules.to(device)  # type: ignore
    nn_baseline.to(device)  # type: ignore
    loss_eval = LossEvaluator(LowLevelInferenceEngine(net_modules), nn_baseline)

    parameters_groups: list[dict[Any, Any]] = [
        {"params": list(net_modules.parameters())},
        {"params": list(nn_baseline.parameters()), "lr": args.baseline_learning_rate},
    ]
    parameters = [param for group in parameters_groups for param in group["params"]]
    optimizer = torch.optim.Adam(
        parameters_groups, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    writer = SummaryWriter(args.tensorboard_log_dir)

    train_data_manager = TrainDataManager(
        training_data,
        args.batch_size,
        args.n_samples,
        args.window_size,
        args.max_trades,
        args.hidden_state_size,
        args.initial_balance,
        device,
    )

    n_iter: int
    for n_iter in tqdm(range(args.iterations)):  # type: ignore
        model_input = train_data_manager.load()
        loss_out: LossOutput = loss_eval(model_input)  # type: ignore

        # TODO: mean?
        loss_out.loss.sum().backward()  # type: ignore

        grad_norm = nn.utils.clip_grad_norm_(
            parameters, max_norm=args.max_gradient_norm, error_if_nonfinite=True
        )
        optimizer.step()

        write_summary(
            writer, loss_out.account_state.balance, loss_out.loss, grad_norm, n_iter
        )

        train_data_manager.save(loss_out.account_state, loss_out.hidden_state)

        if (n_iter + 1) % args.checkpoint_interval == 0:
            save_modules(net_modules, nn_baseline)


# TODO: use a config file...
parser = argparse.ArgumentParser()
parser.add_argument("train_data_file", type=str)
parser.add_argument("iterations", type=int)
parser.add_argument("-lr", "--learning-rate", type=float)
parser.add_argument("--baseline-learning-rate", type=float)
parser.add_argument("--weight-decay", type=float)
parser.add_argument("--max-gradient-norm", type=float)
parser.add_argument("--detect-anomaly", action="store_true")
parser.add_argument("--use-cuda", action="store_true")

parser.add_argument("--batch-size", type=int)
parser.add_argument("--n-samples", type=int)
parser.add_argument("--window-size", type=int)
parser.add_argument("--cnn-out-features", type=int)
parser.add_argument("--max-trades", type=int)
parser.add_argument("--hidden-state-size", type=int)
parser.add_argument("--gated-trans-hidden-size", type=int)
parser.add_argument("--emitter-hidden-size", type=int)
parser.add_argument("--nn-baseline-hidden-size", type=int)
parser.add_argument("--n-iafs", type=int)
parser.add_argument("--iafs-hidden-sizes", type=int, nargs="+")
parser.add_argument("--leverage", type=float)
parser.add_argument("--initial-balance", type=float)

parser.add_argument("--tensorboard-log-dir", type=str)
parser.add_argument("--checkpoint-interval", type=int)
parser.add_argument("--checkpoints-dir", type=str)
parser.add_argument("--load-checkpoint", action="store_true")

nsp = TraingingArgs()
args = parser.parse_args(namespace=nsp)
main(args)
