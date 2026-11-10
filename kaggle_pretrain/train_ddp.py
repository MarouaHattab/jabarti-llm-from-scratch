"""Two-GPU Kaggle pretraining with resumable Hugging Face checkpoints.

Launch with:
    torchrun --standalone --nproc_per_node=2 kaggle_pretrain/train_ddp.py
"""

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from huggingface_hub import HfApi, hf_hub_download
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from jabarti_llm.config import ModelConfig, TrainingConfig, unique_run_name
from jabarti_llm.config import GenerationConfig
from jabarti_llm.data import PackedDataset, collate_fn
from jabarti_llm.generation import generate
from jabarti_llm.model import GPT
from jabarti_llm.training.schedule import cosine_with_warmup
from jabarti_llm.training.trainer import document_cross_entropy
from jabarti_llm.training.tracking import Tracker
from jabarti_llm.tokenizer import Tokenizer


DEFAULT_DATA_DIR = PROJECT_ROOT / "scripts/build_tokenizer/output"
DEFAULT_HF_REPO = "marouaHattab/jabarti-pretrain"
DEFAULT_CHECKPOINT_DIR = (
    Path("/kaggle/working/checkpoints")
    if Path("/kaggle/working").exists()
    else PROJECT_ROOT / "checkpoints"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="training device; DDP Kaggle runs should use cuda",
    )
    parser.add_argument(
        "--resume",
        help="local checkpoint path or hf://repo_id/path/to/checkpoint.pt",
    )
    parser.add_argument("--no-tracking", action="store_true")
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="disable Hugging Face uploads for local smoke tests",
    )
    return parser.parse_args()


def is_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed():
    if not is_distributed():
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():
        raise RuntimeError("DDP Kaggle training requires CUDA, but CUDA is unavailable.")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_device(local_rank, requested):
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if requested != "cpu":
        raise RuntimeError(f"Requested device {requested!r} is unavailable.")
    return torch.device("cpu")


def make_loader(dataset, config, sampler):
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, pad_id=0),
        pin_memory=torch.cuda.is_available(),
    )


def infinite_batches(loader, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def build_optimizer(model, config):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.dim() >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=config.betas,
    )


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def resolve_resume(value, repo_id):
    if not value:
        return None
    if value.startswith("hf://"):
        remote = value[5:]
        parts = remote.split("/")
        if len(parts) < 3:
            raise ValueError("hf:// resume paths must be hf://owner/repo/path/to/file.pt")
        remote_repo = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        return hf_hub_download(repo_id=remote_repo, filename=filename, repo_type="model")
    if value.startswith("hf:"):
        return hf_hub_download(repo_id=repo_id, filename=value[3:], repo_type="model")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return str(path)


def save_checkpoint(path, model, optimizer, scheduler, model_config, step):
    payload = {
        "model": cpu_state(model.state_dict()),
        "model_config": asdict(model_config),
        "optimizer": cpu_state(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def upload_checkpoint(api, path, repo_id, step):
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=f"checkpoints/{path.name}",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload pretraining checkpoint at step {step}",
    )
    manifest = path.parent / "latest.json"
    manifest.write_text(
        json.dumps(
            {"step": step, "checkpoint": f"checkpoints/{path.name}"},
            indent=2,
        ),
        encoding="utf-8",
    )
    api.upload_file(
        path_or_fileobj=str(manifest),
        path_in_repo="checkpoints/latest.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update latest checkpoint pointer at step {step}",
    )


@torch.no_grad()
def evaluate(model, loader, device, batches):
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_batches = torch.zeros(1, device=device)
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        _, loss, _ = model(input_ids, targets=targets)
        total_loss += loss
        total_batches += 1
    if dist.is_initialized():
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_batches, op=dist.ReduceOp.SUM)
    model.train()
    return (total_loss / total_batches.clamp(min=1)).item()


def sample(model, tokenizer, config, device, step):
    model.eval()
    for prompt in config.sample_prompts[:2]:
        text = generate(
            model.module if isinstance(model, DistributedDataParallel) else model,
            tokenizer,
            prompt,
            config=GenerationConfig(max_new_tokens=60, seed=config.seed),
            device=device,
        )
        print(f"  step {step:>6} sample {prompt!r} -> {text!r}")
    model.train()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    device = get_device(local_rank, args.device)

    try:
        torch.manual_seed(args.shuffle_seed + rank)
        model_config = ModelConfig.jabarti()
        training_config = TrainingConfig(
            max_steps=args.steps,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            learning_rate=args.lr,
            warmup_steps=args.warmup,
            weight_decay=args.weight_decay,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            print_every=args.print_every,
            sample_every=args.sample_every,
            save_every=args.save_every,
            checkpoint_dir=str(args.checkpoint_dir),
            run_name=unique_run_name("kaggle-ddp") if is_main else "kaggle-ddp",
        )

        train_dataset = PackedDataset.from_meta(args.data_dir / "train.bin", model_config)
        eval_dataset = PackedDataset.from_meta(args.data_dir / "eval.bin", model_config)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.shuffle_seed,
        )
        eval_sampler = DistributedSampler(
            eval_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        train_loader = make_loader(train_dataset, training_config, train_sampler)
        eval_loader = make_loader(eval_dataset, training_config, eval_sampler)

        model = GPT(model_config).to(device)
        optimizer = build_optimizer(model, training_config)
        scheduler = cosine_with_warmup(optimizer, training_config)
        resume_path = resolve_resume(args.resume, args.hf_repo)
        start_step = 0
        if resume_path:
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_step = int(checkpoint["step"])

        if is_distributed():
            model = DistributedDataParallel(model, device_ids=[local_rank])

        tokenizer = Tokenizer.from_file() if is_main else None
        token = os.environ.get(args.hf_token_env)
        if is_main and not args.no_upload and not token:
            raise RuntimeError(
                f"Missing Hugging Face token in environment variable {args.hf_token_env}."
            )

        tracker = None
        if is_main and not args.no_tracking:
            tracker = Tracker(training_config=training_config, model_config=model_config)
        api = None
        if is_main and not args.no_upload:
            api = HfApi(token=token)
            api.create_repo(repo_id=args.hf_repo, repo_type="model", exist_ok=True)

        batches = infinite_batches(train_loader, train_sampler)
        progress = tqdm(
            total=training_config.max_steps,
            initial=start_step,
            disable=not is_main,
            desc=training_config.run_name,
            unit="step",
        )
        started = time.time()
        for step in range(start_step + 1, training_config.max_steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for micro_step in range(training_config.accumulation_steps):
                batch = next(batches)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                sync_context = (
                    model.no_sync()
                    if isinstance(model, DistributedDataParallel)
                    and micro_step < training_config.accumulation_steps - 1
                    else contextlib.nullcontext()
                )
                with sync_context:
                    logits, _, _ = model(input_ids)
                    loss = document_cross_entropy(
                        logits, targets, model_config.pad_id
                    ) / training_config.accumulation_steps
                    loss.backward()
                total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
            optimizer.step()
            scheduler.step()

            if is_main:
                learning_rate = scheduler.get_last_lr()[0]
                progress.update(1)
                progress.set_postfix(loss=f"{total_loss:.4f}", lr=f"{learning_rate:.2e}")
                if step % training_config.print_every == 0:
                    tqdm.write(f"  step {step:>6} loss {total_loss:.4f} lr {learning_rate:.2e}")
                if tracker is not None:
                    tracker.log(
                        {"train_loss": total_loss, "learning_rate": learning_rate},
                        step=step,
                    )

            if training_config.eval_every and step % training_config.eval_every == 0:
                eval_loss = evaluate(model, eval_loader, device, training_config.eval_batches)
                if is_main:
                    tqdm.write(f"  step {step:>6} eval_loss {eval_loss:.4f}")
                    if tracker is not None:
                        tracker.log({"eval_loss": eval_loss}, step=step)

            if step % training_config.save_every == 0:
                barrier()
                if is_main:
                    path = args.checkpoint_dir / f"pretrain_step{step}.pt"
                    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
                    save_checkpoint(path, raw_model, optimizer, scheduler, model_config, step)
                    if not args.no_upload:
                        upload_checkpoint(api, path, args.hf_repo, step)
                barrier()

            if is_main and training_config.sample_every and step % training_config.sample_every == 0:
                sample(model, tokenizer, training_config, device, step)

        progress.close()
        if is_main:
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            final_path = args.checkpoint_dir / "pretrain_final.pt"
            save_checkpoint(final_path, raw_model, optimizer, scheduler, model_config, training_config.max_steps)
            if not args.no_upload:
                upload_checkpoint(api, final_path, args.hf_repo, training_config.max_steps)
            print(f"done: {training_config.max_steps} steps in {time.time() - started:.1f}s")
        barrier()
        if tracker is not None:
            tracker.finish()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
