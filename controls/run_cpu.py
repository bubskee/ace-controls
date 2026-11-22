"""Run one of the repository's scripts unchanged, on CPU.

torch 2.14 on Apple MPS raises NotImplementedError for dropout inside scaled_dot_product_attention,
which nn.TransformerEncoderLayer uses in training mode. The repository's get_device() prefers MPS,
so this wrapper reports MPS as unavailable and then runs the script as __main__.

Usage: python run_cpu.py SCRIPT [args...]   (run from the directory the script expects)
"""
import functools
import runpy
import sys

import torch

torch.backends.mps.is_available = functools.lru_cache(lambda: False)  # torch._dynamo expects __wrapped__
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
