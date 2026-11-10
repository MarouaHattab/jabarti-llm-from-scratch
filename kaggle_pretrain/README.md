# Kaggle two-GPU pretraining

This folder runs the full Jabarti model with PyTorch DistributedDataParallel on
Kaggle's two T4 GPUs. Checkpoints are uploaded during training to
[marouaHattab/jabarti-pretrain](https://huggingface.co/marouaHattab/jabarti-pretrain),
so a stopped Kaggle session does not lose the latest saved step.

## Kaggle setup

1. Create a Kaggle Notebook and enable **Internet**.
2. Select **GPU T4 x2** in Notebook settings.
3. Add a Kaggle Secret named `HF_TOKEN` containing a Hugging Face token with
   write permission to `marouaHattab/jabarti-pretrain`.
4. Clone and install:

```python
!git clone https://github.com/MarouaHattab/jabarti-llm-from-scratch.git
%cd jabarti-llm-from-scratch
!pip install -e .
```

Expose the Kaggle Secret to the runner:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Verify both GPUs:

```python
import torch
print(torch.cuda.device_count())
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

The output should show `2` and two T4 devices. The runner must be launched
with `torchrun`; running `python` directly uses only one process/GPU.

## Dataset

Attach a Kaggle Dataset containing `train.bin`, `eval.bin`, and `meta.json`,
then copy it into the repository's expected path:

```python
!mkdir -p scripts/build_tokenizer/output
!cp /kaggle/input/YOUR_DATASET/train.bin scripts/build_tokenizer/output/
!cp /kaggle/input/YOUR_DATASET/eval.bin scripts/build_tokenizer/output/
!cp /kaggle/input/YOUR_DATASET/meta.json scripts/build_tokenizer/output/
!ls -lh scripts/build_tokenizer/output
```

## Start training

```python
!torchrun --standalone --nproc_per_node=2 kaggle_pretrain/train_ddp.py \
  --steps 5000 \
  --batch-size 4 \
  --accumulation-steps 4 \
  --lr 3e-4 \
  --warmup 500 \
  --weight-decay 0.1 \
  --eval-every 500 \
  --eval-batches 20 \
  --print-every 100 \
  --sample-every 1000 \
  --save-every 500 \
  --shuffle-seed 42 \
  --checkpoint-dir /kaggle/working/checkpoints \
  --hf-repo marouaHattab/jabarti-pretrain
```

The effective batch size is:

```text
batch-size × accumulation-steps × GPU count
4 × 4 × 2 = 32
```

Only rank zero creates Trackio runs, saves files, and uploads them. The other
rank participates in training and evaluation but never writes duplicate
artifacts.

For a local smoke test without Hugging Face credentials, add
`--no-upload --no-tracking`. Do not use those flags for the real Kaggle run.

## Resume after Kaggle stops

Resume from the manifest's checkpoint path:

```python
!torchrun --standalone --nproc_per_node=2 kaggle_pretrain/train_ddp.py \
  --resume hf://marouaHattab/jabarti-pretrain/checkpoints/pretrain_step500.pt \
  --steps 10000 \
  --batch-size 4 \
  --accumulation-steps 4 \
  --warmup 500 \
  --save-every 500 \
  --checkpoint-dir /kaggle/working/checkpoints \
  --hf-repo marouaHattab/jabarti-pretrain
```

Replace `pretrain_step500.pt` with the latest uploaded checkpoint. The
checkpoint includes model weights, optimizer state, scheduler state, model
configuration, and training step.

## Verify uploaded checkpoints

```python
from huggingface_hub import HfApi

api = HfApi()
files = api.list_repo_files("marouaHattab/jabarti-pretrain", repo_type="model")
print("\n".join(files))
```

Kaggle's `/kaggle/working` storage is temporary. Treat the Hugging Face
repository as the durable copy and use `--save-every 500` or a smaller value
if the session is unstable.
