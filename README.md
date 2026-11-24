<img src="assets/icon.svg" width="64" height="64" alt="">

# mlt

Train language models across two or more Apple Silicon laptops connected by Thunderbolt.
Built on [MLX](https://github.com/ml-explore/mlx); model architectures and loading come from
[mlx-lm](https://github.com/ml-explore/mlx-lm).

| Stage (`stage:`) | Starts from | Data | Methods | Loss on |
|---|---|---|---|---|
| `pretraining` | `scratch` (mlt's GPT, any size) | raw text | `full` | every token |
| `continued_pretraining` | a checkpoint (HF repo or local dir) | raw text | `full`, `lora`, `qlora` | every token |
| `sft` (supervised fine-tuning) | a checkpoint | chat conversations | `full`, `lora`, `qlora` | assistant replies |

Continued pretraining and SFT share one pipeline; they differ only in the data format and the
loss mask. A model pretrained with mlt is saved in Hugging Face Llama format, so it can be the
`from:` of a continued-pretraining or SFT run just like a downloaded model.

## How the laptops cooperate

Each laptop holds a full copy of the model and processes a share of every batch (data
parallelism). After each optimizer step's micro-batches, one bucketed all-reduce sums the
gradients over the Thunderbolt link. The loss is a token sum, and gradients are divided by the
global token count, so the result is identical to one machine training on the whole batch,
however unevenly the batch is split. That lets a slower or smaller laptop take a smaller share
(`share:`) and smaller micro-batches (`micro_batch_size:` per host).

Thunderbolt 4 laptops talk over TCP on the Thunderbolt Bridge (MLX's `ring` backend). LoRA
gradients are a few MB per step; full fine-tuning sends the whole gradient (2 bytes/param with
`comm_dtype: bf16`), so use gradient accumulation (a large `global_batch_size` relative to
`micro_batch_size`) to amortise it.

Since every laptop holds the full model, the smallest laptop sets the model size. Rough limits
for a 16 GB laptop (about 11 GB usable by the GPU):

| Method | Bytes per parameter | Fits |
|---|---|---|
| full, `precision: bf16` | ~12 | ~0.6B (Qwen3-0.6B) |
| lora (bf16 base) | ~2 + adapters | Qwen3-1.7B comfortably, 4B tight |
| qlora (4-bit base) | ~0.56 + adapters | Qwen3-8B |
| pretraining, `precision: mixed` | ~16 | a few hundred M |

`mlt train <config> --dry-run` prints a per-laptop estimate.

## Setup

On every laptop:

1. Clone this repo **at the same absolute path** and run `uv sync`.
2. Turn on *System Settings → General → Sharing → Remote Login*.
3. Make passwordless SSH work from the coordinator (the laptop you launch from) to every
   other laptop: `ssh-copy-id <user>@<host>`.

Then connect the laptops with Thunderbolt cables (two laptops: one cable; three or more:
a ring, each laptop cabled to two neighbours) and, from the coordinator:

```sh
uv run mlt cluster setup --hosts air-24,air-16 --auto-setup   # this laptop first; asks for sudo on each
uv run mlt doctor configs/examples/sft-qwen3-8b-qlora.yaml     # versions, code, GPU memory on every laptop
uv run mlt cluster bench configs/examples/sft-qwen3-8b-qlora.yaml   # optional: measure speeds for share: auto
```

`cluster setup` wraps `mlx.distributed_config`, which assigns Thunderbolt Bridge addresses
and writes `.mlt/cluster/hostfile.json`. mlt never changes macOS's GPU memory limit;
`doctor` only reports it.

## Train

```sh
uv run mlt train configs/examples/pretrain-gpt-small.yaml
uv run mlt train configs/examples/sft-qwen3-8b-qlora.yaml --dry-run   # prepare + memory estimate only
uv run mlt train configs/examples/sft-qwen3-8b-qlora.yaml --resume    # continue from the last checkpoint
```

`mlt train` on the coordinator:

1. downloads the model (to `.mlt/models/`), resolves or trains the tokenizer, and tokenizes
   the data into cached shards under `.mlt/data/` (Hugging Face datasets are streamed);
2. copies the code and exactly those artifacts to the other laptops with `rsync` over the
   Thunderbolt link, and runs `uv sync --frozen` there;
3. starts one process per laptop with `mlx.launch`.

Without a `cluster:` section (or with one host) it simply trains locally.

Rank 0 (the coordinator) writes checkpoints to `runs/<name>/checkpoints/` and the result to
`runs/<name>/final/`: a Hugging Face-format model for `full`, or adapters (mlx-lm format) for
`lora`/`qlora`. The data order depends only on the seed, so `--resume` works with a different
set of laptops too (including just one). Resuming refuses to continue if settings that change
the training trajectory were edited.

## After training

```sh
uv run mlt chat runs/qwen3-8b-sft-no-robots/final
uv run mlt export runs/qwen3-8b-sft-no-robots/final --out exports/qwen3-8b-sft      # merge LoRA into bf16 base
uv run mlt export exports/qwen3-8b-sft --out exports/qwen3-8b-sft-4bit --quantize 4
```

Exports load with `mlx_lm.load` and (unquantized) with `transformers`.

## Configs

See `configs/examples/`. The main sections:

- `model.from`: `scratch`, a Hugging Face repo id, or a local model dir. For `scratch`, also
  `arch` (`preset: tiny|small|base|medium` or explicit `n_layers`, `d_model`, `n_heads`,
  `n_kv_heads`, `d_ff`, `context_length`, ...) and `tokenizer` (`reuse: <hf id>`,
  `path: <dir>`, or `train: {vocab_size, ...}` for a new byte-level BPE).
- `method`: `type` plus LoRA `rank`, `alpha`, `dropout`, `targets`, `layers`; QLoRA `bits`,
  `group_size`.
- `data`: `train` / `val` sources (`jsonl`, `txt`, `parquet`, `hf`), `val_fraction`,
  `seq_len`, `train_on` (sft). Chat records are `{"messages": [{role, content}, ...]}` or
  `{prompt, completion}`.
- `train`: `global_batch_size` (sequences per step across all laptops), `micro_batch_size`,
  `steps` or `epochs`, LR schedule, `precision` (`fp32`, `bf16`, `mixed`), `grad_checkpoint`,
  eval/save cadence.
- `cluster`: `hosts` (this laptop first, optional per-host `micro_batch_size`), `share`
  (`auto`, `equal` or fractions), `comm_dtype`.

## Adding a model family

mlx-lm supplies the architecture. mlt needs a small adapter per family: which `model_type`
values it covers, default LoRA targets and any chat-template quirks. Add
`src/mlt/models/families/<name>.py` (see `qwen.py`), list it in
`registry._FAMILY_MODULES`, and add a test. Qwen2, Qwen2.5 and Qwen3 dense models and
Llama-format models (including everything pretrained with mlt) are supported now.

## Development

```sh
uv run pytest        # in-memory tests; MLT_NETWORK_TESTS=1 also runs ones that download tokenizers
uv run ruff check src tests
```
