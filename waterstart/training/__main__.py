import argparse
import datetime
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm  # type: ignore

from ..inference import (
    CNN,
    Emitter,
    GatedTransition,
    LowLevelInferenceEngine,
    NetworkModules,
)
from .loss import Critic, LossOutput
from .loss.sequential import SequentialLossEvaluator
from .traindata import TrainingData, TrainingState, load_training_data
from .traindata.sequential import SequentialTrainDataManager

SAVE_FOLDER_NAME_FORMAT = "%Y%m%d_%H%S%f"


@dataclass
class TrainingArgs(argparse.Namespace):
    train_data_file: str = ""
    iterations: int = 0
    # learning_rate: float = 3e-4
    learning_rate: float = 1e-3
    gamma: float = 0.99
    # max_gradient_norm: float = 0.5
    max_gradient_norm: float = float("inf")
    detect_anomaly: bool = False
    use_cuda: bool = False
    seq_len: int = 50
    batch_size: int = 64
    window_size: int = 50
    # window_size: int = 30
    cnn_out_features: int = 256
    max_trades: int = 10
    hidden_state_size: int = 128
    gated_trans_hidden_size: int = 200
    critic_hidden_size: int = 200
    emitter_hidden_size: int = 200
    leverage: float = 1.0
    initial_balance: float = 100_000.0
    tensorboard_log_dir: Optional[str] = None
    checkpoint_interval: int = 1000
    checkpoints_dir: Optional[str] = None
    load_checkpoint: bool = False


def write_summary(
    writer: SummaryWriter,
    balance: torch.Tensor,
    loss: torch.Tensor,
    grad_norm: torch.Tensor,
    n_iter: int,
):
    single_step_returns = balance[1:] / balance[:-1] - 1
    writer.add_scalar(  # type: ignore
        "single step/avg return", single_step_returns.mean().item(), n_iter
    )

    positive_returns = torch.sum(single_step_returns > 0).item()
    negative_returns = torch.sum(single_step_returns < 0).item()
    tot_returns = single_step_returns.numel()
    writer.add_scalar(  # type: ignore
        "single step/positive fraction", positive_returns / tot_returns, n_iter
    )
    writer.add_scalar(  # type: ignore
        "single step/negative fraction", negative_returns / tot_returns, n_iter
    )

    max_returns = balance.max(0).values / balance[0] - 1
    writer.add_scalar(  # type: ignore
        "whole sequence/avg max return", max_returns.mean().item(), n_iter
    )

    min_returns = balance.min(0).values / balance[0] - 1
    writer.add_scalar(  # type: ignore
        "whole sequence/avg min return", min_returns.mean().item(), n_iter
    )

    close_returns = balance[-1] / balance[0] - 1
    writer.add_scalar(  # type: ignore
        "whole sequence/avg close return", close_returns.mean().item(), n_iter
    )

    loss_loc: float = loss.mean().item()  # type: ignore
    loss_scale: float = loss.std().item()  # type: ignore
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
    args: TrainingArgs, training_data: TrainingData
) -> tuple[NetworkModules, Critic]:
    cnn = CNN(args.window_size, training_data.n_market_features, args.cnn_out_features)
    gated_trans = GatedTransition(
        args.cnn_out_features,
        args.hidden_state_size,
        args.gated_trans_hidden_size,
    )
    emitter = Emitter(
        # args.hidden_state_size,
        args.cnn_out_features,
        training_data.n_traded_sym,
        args.emitter_hidden_size,
    )

    return NetworkModules(
        cnn,
        gated_trans,
        emitter,
        training_data.market_data_arr_mapper,
        training_data.traded_sym_arr_mapper,
        training_data.traded_symbols,
        args.max_trades,
        args.leverage,
    ), Critic(
        # args.hidden_state_size,
        args.cnn_out_features,
        args.critic_hidden_size,
        training_data.n_traded_sym,
    )


def load_state_dict(search_dir: str) -> Optional[dict[str, Any]]:
    checkpoints_dir = Path(search_dir)
    latest_checkpoint_dir: Optional[Path] = None
    latest_dt: datetime.datetime = datetime.datetime.min

    for dir in checkpoints_dir.iterdir():
        if not dir.is_dir():
            continue

        try:
            dt = datetime.datetime.strptime(dir.name, SAVE_FOLDER_NAME_FORMAT)
        except ValueError:
            raise RuntimeError()

        if dt > latest_dt:
            latest_dt = dt
            latest_checkpoint_dir = dir

    if latest_checkpoint_dir is None:
        return None

    state_dict = torch.load(latest_checkpoint_dir / "checkpoint.pt")  # type: ignore

    assert isinstance(state_dict, dict)
    assert all(isinstance(key, str) for key in state_dict)  # type: ignore

    return state_dict  # type: ignore


def get_state(
    args: TrainingArgs, training_data: TrainingData
) -> tuple[NetworkModules, Critic, Optional[TrainingState]]:

    net_modules, critic = build_modules(args, training_data)

    if (
        (checkpoints_dir := args.checkpoints_dir) is None
        or not args.load_checkpoint
        or (state_dict := load_state_dict(checkpoints_dir)) is None
    ):
        return net_modules, critic, None

    net_modules.load_state_dict(state_dict["net_modules"])
    critic.load_state_dict(state_dict["critic"])
    training_state = state_dict["training_state"]
    assert isinstance(training_state, TrainingState)
    return net_modules, critic, training_state


def remove_content(save_dir: Path) -> None:
    for dir in save_dir.iterdir():
        if not dir.is_dir():
            continue

        try:
            datetime.datetime.strptime(dir.name, SAVE_FOLDER_NAME_FORMAT)
        except ValueError:
            continue

        rmtree(dir)


def save_state(
    checkpoints_dir: Path,
    net_modules: NetworkModules,
    critic: Critic,
    training_state: TrainingState,
    remove_prev: bool = True,
) -> None:
    if remove_prev:
        remove_content(checkpoints_dir)

    now = datetime.datetime.now()
    save_dir = checkpoints_dir / now.strftime(SAVE_FOLDER_NAME_FORMAT)
    save_dir.mkdir(parents=True)

    torch.save(  # type: ignore
        {
            "net_modules": net_modules.state_dict(),
            "critic": critic.state_dict(),
            "training_state": training_state,
        },
        save_dir / "checkpoint.pt",
    )


def main(args: TrainingArgs):
    training_data = load_training_data(args.train_data_file)
    net_modules, critic, training_state = get_state(args, training_data)

    torch.autograd.anomaly_mode.set_detect_anomaly(args.detect_anomaly)

    loss_eval = SequentialLossEvaluator(
        LowLevelInferenceEngine(net_modules), critic, args.seq_len, args.gamma
    )
    # loss_eval: SequentialLossEvaluator = jit.script(loss_eval)  # type: ignore

    device = torch.device("cuda" if args.use_cuda else "cpu")
    loss_eval.to(device)  # type: ignore

    parameters = list(loss_eval.parameters())
    optimizer = torch.optim.Adam(
        parameters, lr=args.learning_rate, eps=1e-5, weight_decay=0.001
    )
    writer = SummaryWriter(args.tensorboard_log_dir)

    train_data_manager = SequentialTrainDataManager(
        training_data,
        training_state,
        args.batch_size,
        args.seq_len,
        args.window_size,
        args.max_trades,
        args.hidden_state_size,
        args.initial_balance,
        device,
    )

    max_gradient_norm = args.max_gradient_norm
    checkpoint_interval = args.checkpoint_interval
    checkpoints_dir = None if (dir := args.checkpoints_dir) is None else Path(dir)

    n_iter: int
    for n_iter in tqdm(range(args.iterations)):  # type: ignore
        model_input = train_data_manager.load_data()
        loss_out: LossOutput = loss_eval.evaluate(model_input)  # type: ignore

        optimizer.zero_grad()
        loss_out.surrogate_loss.mean().backward()  # type: ignore

        grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            parameters, max_norm=max_gradient_norm, error_if_nonfinite=True
        )
        optimizer.step()

        write_summary(
            writer, loss_out.account_state.balance, loss_out.loss, grad_norm, n_iter
        )

        # train_data_manager.store_data(loss_out.account_state, loss_out.hidden_state)

        if checkpoints_dir is not None and (n_iter + 1) % checkpoint_interval == 0:
            save_state(
                checkpoints_dir,
                net_modules,
                critic,
                train_data_manager.save_state(),
            )


# TODO: use a config file...
parser = argparse.ArgumentParser()
parser.add_argument("train_data_file", type=str)
parser.add_argument("iterations", type=int)
parser.add_argument("-lr", "--learning-rate", type=float)
parser.add_argument("--baseline-learning-rate", type=float)
parser.add_argument("--gamma", type=float)
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
# parser.add_argument("--n-iafs", type=int)
# parser.add_argument("--iafs-hidden-sizes", type=int, nargs="+")
parser.add_argument("--leverage", type=float)
parser.add_argument("--initial-balance", type=float)

parser.add_argument("--tensorboard-log-dir", type=str)
parser.add_argument("--checkpoint-interval", type=int)
parser.add_argument("--checkpoints-dir", type=str)
parser.add_argument("--load-checkpoint", action="store_true")

nsp = TrainingArgs()
args = parser.parse_args(namespace=nsp)
main(args)
