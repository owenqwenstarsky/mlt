"""Run configuration schema.

A run is described by three independent choices:
  * stage:  pretraining | continued_pretraining | sft
  * model.from: `scratch` (our GPT) or a checkpoint (HF repo id or local dir)
  * method.type: full | lora | qlora
The stage fixes which combinations are legal and fills in defaults.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .presets import PRESETS, default_ffn_dim

Stage = Literal["pretraining", "continued_pretraining", "sft"]
MethodType = Literal["full", "lora", "qlora"]
Precision = Literal["fp32", "bf16", "mixed"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------- data


class DataSource(_Strict):
    """Where records come from.

    jsonl/parquet: one record per row. txt: one document per file (path may be a
    file, a directory or a glob). hf: a Hugging Face dataset, always streamed.
    """

    source: Literal["jsonl", "txt", "parquet", "hf"]
    path: str
    split: str = "train"
    name: str | None = None  # HF dataset config name
    field: str | None = None  # defaults to "text" (text stages) or "messages" (sft)
    max_samples: int | None = Field(default=None, gt=0)


class DataConfig(_Strict):
    train: DataSource
    val: DataSource | None = None
    # Hold out this fraction of the training data for validation when `val` is not set.
    val_fraction: float = Field(default=0.0, ge=0.0, lt=0.5)
    seq_len: int = Field(default=1024, gt=1)
    # sft only: compute loss on assistant replies only, or on every token.
    train_on: Literal["assistant", "all"] = "assistant"


# -------------------------------------------------------------------------- model


class TokenizerTrainConfig(_Strict):
    vocab_size: int = Field(default=32000, ge=300)
    min_frequency: int = Field(default=2, ge=1)
    corpus: DataSource | None = None  # defaults to data.train
    max_docs: int | None = Field(default=None, gt=0)


class TokenizerConfig(_Strict):
    """Exactly one of: reuse an existing HF tokenizer, load one from a path, or train BPE."""

    reuse: str | None = None
    path: str | None = None
    train: TokenizerTrainConfig | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> TokenizerConfig:
        n = sum(x is not None for x in (self.reuse, self.path, self.train))
        if n != 1:
            raise ValueError("tokenizer: set exactly one of `reuse`, `path` or `train`")
        return self


class ArchConfig(_Strict):
    preset: str | None = None
    n_layers: int | None = Field(default=None, gt=0)
    d_model: int | None = Field(default=None, gt=0)
    n_heads: int | None = Field(default=None, gt=0)
    n_kv_heads: int | None = Field(default=None, gt=0)
    d_ff: int | None = Field(default=None, gt=0)
    context_length: int | None = Field(default=None, gt=0)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02

    def resolved(self) -> ArchConfig:
        """Fill unset fields from the preset and derived defaults."""
        values = self.model_dump()
        if self.preset is not None:
            if self.preset not in PRESETS:
                raise ValueError(
                    f"unknown preset {self.preset!r}; choose one of {sorted(PRESETS)}"
                )
            for k, v in PRESETS[self.preset].items():
                if values.get(k) is None:
                    values[k] = v
        missing = [k for k in ("n_layers", "d_model", "n_heads") if values.get(k) is None]
        if missing:
            raise ValueError(f"arch: set `preset` or all of {missing}")
        if values["n_kv_heads"] is None:
            values["n_kv_heads"] = values["n_heads"]
        if values["d_ff"] is None:
            values["d_ff"] = default_ffn_dim(values["d_model"])
        if values["context_length"] is None:
            values["context_length"] = 1024
        if values["d_model"] % values["n_heads"]:
            raise ValueError("arch: d_model must be divisible by n_heads")
        if values["n_heads"] % values["n_kv_heads"]:
            raise ValueError("arch: n_heads must be divisible by n_kv_heads")
        return ArchConfig(**values)


class ModelConfig(_Strict):
    from_: str = Field(alias="from")
    arch: ArchConfig | None = None
    tokenizer: TokenizerConfig | None = None
    revision: str | None = None
    trust_remote_code: bool = False

    @property
    def is_scratch(self) -> bool:
        return self.from_ == "scratch"


# ------------------------------------------------------------------------- method


class MethodConfig(_Strict):
    type: MethodType = "full"
    rank: int = Field(default=16, gt=0)
    alpha: float = Field(default=32.0, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    # "default" uses the model family's target list; otherwise module suffixes like
    # "self_attn.q_proj".
    targets: Literal["default"] | list[str] = "default"
    # Apply LoRA to the last N transformer blocks only (None = all blocks).
    layers: int | None = Field(default=None, gt=0)
    # qlora only
    bits: Literal[2, 3, 4, 5, 6, 8] = 4
    group_size: Literal[32, 64, 128] = 64

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


# -------------------------------------------------------------------------- train


class TrainConfig(_Strict):
    global_batch_size: int = Field(default=32, gt=0)  # sequences per optimizer step
    micro_batch_size: int = Field(default=1, gt=0)  # default per-laptop micro batch
    steps: int | None = Field(default=None, gt=0)
    epochs: float | None = Field(default=None, gt=0)
    lr: float = Field(default=3e-4, gt=0)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    warmup_steps: int = Field(default=0, ge=0)
    schedule: Literal["cosine", "linear", "constant"] = "cosine"
    weight_decay: float | None = Field(default=None, ge=0.0)
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float | None = Field(default=1.0, gt=0)
    precision: Precision | None = None
    grad_checkpoint: bool = False
    compile: bool = True
    seed: int = 0
    log_every: int = Field(default=10, gt=0)
    eval_every: int | None = Field(default=None, gt=0)
    eval_batches: int = Field(default=20, gt=0)
    save_every: int | None = Field(default=None, gt=0)
    keep_checkpoints: int = Field(default=3, gt=0)
    sample_prompts: list[str] = Field(default_factory=list)
    sample_max_tokens: int = Field(default=64, gt=0)

    @model_validator(mode="after")
    def _steps_or_epochs(self) -> TrainConfig:
        if (self.steps is None) == (self.epochs is None):
            raise ValueError("train: set exactly one of `steps` or `epochs`")
        return self


# ------------------------------------------------------------------------ cluster


class HostConfig(_Strict):
    host: str
    micro_batch_size: int | None = Field(default=None, gt=0)


class ClusterConfig(_Strict):
    hosts: list[HostConfig] = Field(min_length=1)
    # "auto" uses `mlt cluster bench` results (equal split if none), "equal", or
    # explicit fractions that sum to 1.
    share: Literal["auto", "equal"] | list[float] = "auto"
    comm_dtype: Literal["bf16", "fp32"] = "bf16"
    hostfile: str = ".mlt/cluster/hostfile.json"

    @model_validator(mode="after")
    def _check(self) -> ClusterConfig:
        names = [h.host for h in self.hosts]
        if len(set(names)) != len(names):
            raise ValueError("cluster.hosts: duplicate host names")
        if isinstance(self.share, list):
            if len(self.share) != len(self.hosts):
                raise ValueError("cluster.share: need one fraction per host")
            if any(s <= 0 for s in self.share):
                raise ValueError("cluster.share: fractions must be > 0")
            if abs(sum(self.share) - 1.0) > 1e-6:
                raise ValueError("cluster.share: fractions must sum to 1")
        return self


# ---------------------------------------------------------------------------- run


class RunConfig(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    stage: Stage
    model: ModelConfig
    method: MethodConfig = Field(default_factory=MethodConfig)
    data: DataConfig
    train: TrainConfig
    cluster: ClusterConfig | None = None
    output_dir: str = "runs"

    @property
    def data_kind(self) -> Literal["text", "chat"]:
        return "chat" if self.stage == "sft" else "text"

    @model_validator(mode="after")
    def _stage_rules(self) -> RunConfig:
        m, stage = self.model, self.stage
        if stage == "pretraining":
            if not m.is_scratch:
                raise ValueError(
                    "pretraining starts from `scratch`; to keep training an existing "
                    "checkpoint use stage `continued_pretraining`"
                )
            if self.method.type != "full":
                raise ValueError("pretraining trains every weight: method.type must be `full`")
            if m.arch is None or m.tokenizer is None:
                raise ValueError("pretraining needs `model.arch` and `model.tokenizer`")
            self.model.arch = m.arch.resolved()
            if self.data.seq_len > self.model.arch.context_length:
                raise ValueError(
                    f"data.seq_len ({self.data.seq_len}) exceeds arch.context_length "
                    f"({self.model.arch.context_length})"
                )
        else:
            if m.is_scratch:
                raise ValueError(
                    f"{stage} starts from a checkpoint (HF repo or local dir), not `scratch`"
                )
            if m.arch is not None or m.tokenizer is not None:
                raise ValueError(
                    f"{stage}: `arch` and `tokenizer` come from the checkpoint; remove them"
                )
        if stage != "sft" and self.data.train_on != "assistant":
            raise ValueError("data.train_on only applies to stage `sft`")

        # Stage/method dependent defaults.
        t = self.train
        if t.precision is None:
            t.precision = "mixed" if stage == "pretraining" else "bf16"
        if t.weight_decay is None:
            t.weight_decay = 0.1 if stage == "pretraining" else 0.0
        if self.method.type != "full" and t.precision == "mixed":
            raise ValueError(
                "precision `mixed` keeps fp32 master weights for full training; LoRA "
                "adapters are always fp32, so use `bf16` or `fp32` for the frozen base"
            )
        for src in (self.data.train, self.data.val):
            if src is not None and src.field is None:
                src.field = "messages" if self.data_kind == "chat" else "text"
        if self.data.val is not None and self.data.val_fraction > 0:
            raise ValueError("data: set either `val` or `val_fraction`, not both")
        if self.cluster is not None:
            world = len(self.cluster.hosts)
            if t.global_batch_size < world:
                raise ValueError("train.global_batch_size must be >= number of hosts")
        return self

    def micro_batch_sizes(self) -> list[int]:
        if self.cluster is None:
            return [self.train.micro_batch_size]
        return [h.micro_batch_size or self.train.micro_batch_size for h in self.cluster.hosts]
