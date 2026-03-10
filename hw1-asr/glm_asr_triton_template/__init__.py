"""
Template Baseline - Triton Student Assignment
Performance: TBD (Torch baseline with Triton kernels available)

Key Characteristics:
- Pure Torch tensor operations
- Triton kernels for core ops (student TODOs)
"""

import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from . import layers

layers.Linear.BACKEND = "triton"
layers.MLP.FUSED = True
layers.EncoderMLP.FUSED = True

from . import model
from . import rope
from . import conv
from . import weight_loader
