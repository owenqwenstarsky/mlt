"""mlt: pretraining, continued pretraining and SFT across Apple Silicon laptops."""

import os

# transformers is only used for tokenizers here; silence its "PyTorch not found" advisory.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
