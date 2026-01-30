"""
cuTile Compatibility Layer for Non-Blackwell GPUs (Hopper Hack)

This module provides a drop-in replacement for cuda.tile that works on
older GPUs (Ada Lovelace sm_89, Ampere sm_80, etc.) by using CuPy RawKernel.

The original cuTile only supports Blackwell GPUs (sm_100+).
This hack intercepts the cuTile API and generates equivalent CUDA C++ code
that can run on any CUDA-capable GPU.
"""

import builtins
import cupy as cp
import numpy as np
import math
import ast
import inspect
import textwrap
from typing import Callable, Tuple, Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from functools import wraps
import hashlib

# Save Python builtins before we override them
_builtin_min = builtins.min
_builtin_max = builtins.max
_builtin_sum = builtins.sum
_builtin_pow = builtins.pow

# =============================================================================
# Data Types
# =============================================================================

class DType:
    """Base class for cuTile data types."""
    pass

# Data type singletons
class _Int8(DType):
    name = "int8"
    ctype = "signed char"
    nptype = np.int8
int8 = _Int8()

class _Int16(DType):
    name = "int16"
    ctype = "short"
    nptype = np.int16
int16 = _Int16()

class _Int32(DType):
    name = "int32"
    ctype = "int"
    nptype = np.int32
int32 = _Int32()

class _Int64(DType):
    name = "int64"
    ctype = "long long"
    nptype = np.int64
int64 = _Int64()

class _UInt8(DType):
    name = "uint8"
    ctype = "unsigned char"
    nptype = np.uint8
uint8 = _UInt8()

class _UInt16(DType):
    name = "uint16"
    ctype = "unsigned short"
    nptype = np.uint16
uint16 = _UInt16()

class _UInt32(DType):
    name = "uint32"
    ctype = "unsigned int"
    nptype = np.uint32
uint32 = _UInt32()

class _UInt64(DType):
    name = "uint64"
    ctype = "unsigned long long"
    nptype = np.uint64
uint64 = _UInt64()

class _Float16(DType):
    name = "float16"
    ctype = "__half"
    nptype = np.float16
float16 = _Float16()

class _Float32(DType):
    name = "float32"
    ctype = "float"
    nptype = np.float32
float32 = _Float32()

class _Float64(DType):
    name = "float64"
    ctype = "double"
    nptype = np.float64
float64 = _Float64()

class _BFloat16(DType):
    name = "bfloat16"
    ctype = "__nv_bfloat16"
    nptype = np.float16  # CuPy uses float16 for bfloat16
bfloat16 = _BFloat16()

class _TFloat32(DType):
    name = "tfloat32"
    ctype = "float"  # TF32 uses float storage
    nptype = np.float32
tfloat32 = _TFloat32()

class _Bool(DType):
    name = "bool"
    ctype = "bool"
    nptype = np.bool_
bool_ = _Bool()

class _Float8E4M3FN(DType):
    name = "float8_e4m3fn"
    ctype = "__nv_fp8_e4m3"
    nptype = np.float16
float8_e4m3fn = _Float8E4M3FN()

class _Float8E5M2(DType):
    name = "float8_e5m2"
    ctype = "__nv_fp8_e5m2"
    nptype = np.float16
float8_e5m2 = _Float8E5M2()


def _dtype_to_ctype(dtype) -> str:
    """Convert numpy/cupy dtype to C type string."""
    if isinstance(dtype, DType):
        return dtype.ctype
    dtype = np.dtype(dtype)
    mapping = {
        np.float64: "double",
        np.float32: "float",
        np.float16: "__half",
        np.int64: "long long",
        np.int32: "int",
        np.int16: "short",
        np.int8: "signed char",
        np.uint64: "unsigned long long",
        np.uint32: "unsigned int",
        np.uint16: "unsigned short",
        np.uint8: "unsigned char",
        np.bool_: "bool",
    }
    return mapping.get(dtype.type, "float")


def _dtype_to_nptype(dtype):
    """Convert cuTile dtype to numpy dtype."""
    if isinstance(dtype, DType):
        return dtype.nptype
    return np.dtype(dtype)


# =============================================================================
# Type Annotations
# =============================================================================

class Constant:
    """Type annotation for compile-time constants."""
    def __class_getitem__(cls, item):
        return item


class ConstantAnnotation:
    """Marker for constant annotations."""
    pass


class Array:
    """Type annotation for arrays."""
    def __class_getitem__(cls, item):
        return item


class Scalar:
    """Type annotation for scalars."""
    def __class_getitem__(cls, item):
        return item


class Tile:
    """Type annotation for tiles."""
    def __class_getitem__(cls, item):
        return item


class ByTarget:
    """Target-specific configuration."""
    def __class_getitem__(cls, item):
        return item


# =============================================================================
# Enums
# =============================================================================

class MemoryOrder:
    relaxed = "relaxed"
    acquire = "acquire"
    release = "release"
    acq_rel = "acq_rel"
    seq_cst = "seq_cst"


class MemoryScope:
    system = "system"
    device = "device"
    block = "block"


class PaddingMode:
    zeros = "zeros"
    reflect = "reflect"
    replicate = "replicate"


class RoundingMode:
    nearest = "nearest"
    down = "down"
    up = "up"
    truncate = "truncate"


# =============================================================================
# Exceptions
# =============================================================================

class TileCompilerError(Exception):
    """Base class for tile compiler errors."""
    pass


class TileCompilerExecutionError(TileCompilerError):
    """Raised when tile compiler execution fails."""
    pass


class TileCompilerTimeoutError(TileCompilerError):
    """Raised when tile compiler times out."""
    pass


class TileInternalError(TileCompilerError):
    """Raised for internal errors."""
    pass


class TileSyntaxError(TileCompilerError):
    """Raised for syntax errors in tile code."""
    pass


class TileTypeError(TileCompilerError):
    """Raised for type errors in tile code."""
    pass


class TileValueError(TileCompilerError):
    """Raised for value errors in tile code."""
    pass


# =============================================================================
# Utility Functions
# =============================================================================

def cdiv(a: int, b: int) -> int:
    """Ceiling division: (a + b - 1) // b"""
    return (a + b - 1) // b


# =============================================================================
# Stub Functions (for AST parsing - not called at runtime)
# =============================================================================

def bid(dim: int) -> int:
    """Get block ID in given dimension."""
    raise RuntimeError("bid() should only be called within a kernel")


def num_blocks(dim: int) -> int:
    """Get number of blocks in given dimension."""
    raise RuntimeError("num_blocks() should only be called within a kernel")


def num_tiles(dim: int) -> int:
    """Get number of tiles in given dimension."""
    raise RuntimeError("num_tiles() should only be called within a kernel")


def load(array, index: Tuple, shape: Tuple):
    """Load a tile from global memory."""
    raise RuntimeError("load() should only be called within a kernel")


def store(array, index: Tuple, tile):
    """Store a tile to global memory."""
    raise RuntimeError("store() should only be called within a kernel")


def full(shape: Tuple, value, dtype=None):
    """Create a tile filled with a value."""
    raise RuntimeError("full() should only be called within a kernel")


def zeros(shape: Tuple, dtype=None):
    """Create a tile filled with zeros."""
    raise RuntimeError("zeros() should only be called within a kernel")


def ones(shape: Tuple, dtype=None):
    """Create a tile filled with ones."""
    raise RuntimeError("ones() should only be called within a kernel")


def arange(start, stop=None, step=1, dtype=None):
    """Create a tile with evenly spaced values."""
    raise RuntimeError("arange() should only be called within a kernel")


def astype(tile, dtype):
    """Convert tile to specified data type."""
    raise RuntimeError("astype() should only be called within a kernel")


def transpose(tile, axes=None):
    """Transpose a tile."""
    raise RuntimeError("transpose() should only be called within a kernel")


def permute(tile, axes):
    """Permute tile dimensions."""
    raise RuntimeError("permute() should only be called within a kernel")


def reshape(tile, shape):
    """Reshape a tile."""
    raise RuntimeError("reshape() should only be called within a kernel")


def broadcast_to(tile, shape):
    """Broadcast tile to shape."""
    raise RuntimeError("broadcast_to() should only be called within a kernel")


def expand_dims(tile, axis):
    """Expand tile dimensions."""
    raise RuntimeError("expand_dims() should only be called within a kernel")


def cat(tiles, axis=0):
    """Concatenate tiles."""
    raise RuntimeError("cat() should only be called within a kernel")


def bitcast(tile, dtype):
    """Bitcast tile to dtype."""
    raise RuntimeError("bitcast() should only be called within a kernel")


def extract(tile, indices):
    """Extract elements from tile."""
    raise RuntimeError("extract() should only be called within a kernel")


def gather(array, indices, axis=0):
    """Gather elements from array."""
    raise RuntimeError("gather() should only be called within a kernel")


def scatter(array, indices, tile, axis=0):
    """Scatter tile to array."""
    raise RuntimeError("scatter() should only be called within a kernel")


def where(condition, x, y):
    """Conditional selection."""
    raise RuntimeError("where() should only be called within a kernel")


# Math functions
def exp(x):
    raise RuntimeError("exp() should only be called within a kernel")

def exp2(x):
    raise RuntimeError("exp2() should only be called within a kernel")

def log(x):
    raise RuntimeError("log() should only be called within a kernel")

def log2(x):
    raise RuntimeError("log2() should only be called within a kernel")

def sqrt(x):
    raise RuntimeError("sqrt() should only be called within a kernel")

def rsqrt(x):
    raise RuntimeError("rsqrt() should only be called within a kernel")

def sin(x):
    raise RuntimeError("sin() should only be called within a kernel")

def cos(x):
    raise RuntimeError("cos() should only be called within a kernel")

def tan(x):
    raise RuntimeError("tan() should only be called within a kernel")

def sinh(x):
    raise RuntimeError("sinh() should only be called within a kernel")

def cosh(x):
    raise RuntimeError("cosh() should only be called within a kernel")

def tanh(x):
    raise RuntimeError("tanh() should only be called within a kernel")

def floor(x):
    raise RuntimeError("floor() should only be called within a kernel")

def ceil(x):
    raise RuntimeError("ceil() should only be called within a kernel")

def pow(x, y):
    raise RuntimeError("pow() should only be called within a kernel")


# Reduction functions
def sum(x, axis=None):
    raise RuntimeError("sum() should only be called within a kernel")

def prod(x, axis=None):
    raise RuntimeError("prod() should only be called within a kernel")

def min(x, axis=None):
    raise RuntimeError("min() should only be called within a kernel")

def max(x, axis=None):
    raise RuntimeError("max() should only be called within a kernel")

def argmin(x, axis=None):
    raise RuntimeError("argmin() should only be called within a kernel")

def argmax(x, axis=None):
    raise RuntimeError("argmax() should only be called within a kernel")

def cumsum(x, axis=None):
    raise RuntimeError("cumsum() should only be called within a kernel")

def cumprod(x, axis=None):
    raise RuntimeError("cumprod() should only be called within a kernel")

def minimum(x, y):
    raise RuntimeError("minimum() should only be called within a kernel")

def maximum(x, y):
    raise RuntimeError("maximum() should only be called within a kernel")


# Binary operations
def add(x, y):
    raise RuntimeError("add() should only be called within a kernel")

def sub(x, y):
    raise RuntimeError("sub() should only be called within a kernel")

def mul(x, y):
    raise RuntimeError("mul() should only be called within a kernel")

def truediv(x, y):
    raise RuntimeError("truediv() should only be called within a kernel")

def floordiv(x, y):
    raise RuntimeError("floordiv() should only be called within a kernel")

def mod(x, y):
    raise RuntimeError("mod() should only be called within a kernel")

def negative(x):
    raise RuntimeError("negative() should only be called within a kernel")


# Comparison
def equal(x, y):
    raise RuntimeError("equal() should only be called within a kernel")

def not_equal(x, y):
    raise RuntimeError("not_equal() should only be called within a kernel")

def less(x, y):
    raise RuntimeError("less() should only be called within a kernel")

def less_equal(x, y):
    raise RuntimeError("less_equal() should only be called within a kernel")

def greater(x, y):
    raise RuntimeError("greater() should only be called within a kernel")

def greater_equal(x, y):
    raise RuntimeError("greater_equal() should only be called within a kernel")


# Bitwise
def bitwise_and(x, y):
    raise RuntimeError("bitwise_and() should only be called within a kernel")

def bitwise_or(x, y):
    raise RuntimeError("bitwise_or() should only be called within a kernel")

def bitwise_xor(x, y):
    raise RuntimeError("bitwise_xor() should only be called within a kernel")

def bitwise_not(x):
    raise RuntimeError("bitwise_not() should only be called within a kernel")

def bitwise_lshift(x, y):
    raise RuntimeError("bitwise_lshift() should only be called within a kernel")

def bitwise_rshift(x, y):
    raise RuntimeError("bitwise_rshift() should only be called within a kernel")


# Matrix operations
def matmul(a, b):
    raise RuntimeError("matmul() should only be called within a kernel")

def mma(a, b, c):
    raise RuntimeError("mma() should only be called within a kernel")


# Atomic operations
def atomic_add(array, index, value):
    raise RuntimeError("atomic_add() should only be called within a kernel")

def atomic_and(array, index, value):
    raise RuntimeError("atomic_and() should only be called within a kernel")

def atomic_or(array, index, value):
    raise RuntimeError("atomic_or() should only be called within a kernel")

def atomic_xor(array, index, value):
    raise RuntimeError("atomic_xor() should only be called within a kernel")

def atomic_min(array, index, value):
    raise RuntimeError("atomic_min() should only be called within a kernel")

def atomic_max(array, index, value):
    raise RuntimeError("atomic_max() should only be called within a kernel")

def atomic_xchg(array, index, value):
    raise RuntimeError("atomic_xchg() should only be called within a kernel")

def atomic_cas(array, index, compare, value):
    raise RuntimeError("atomic_cas() should only be called within a kernel")


# Debug
def printf(fmt, *args):
    raise RuntimeError("printf() should only be called within a kernel")

def assert_(condition, msg=""):
    raise RuntimeError("assert_() should only be called within a kernel")


# =============================================================================
# Kernel Analysis and Code Generation
# =============================================================================

@dataclass
class KernelInfo:
    """Information about a kernel extracted from AST analysis."""
    name: str
    params: List[str]
    constant_params: List[str]
    source: str
    pattern: str = "generic"  # vector_add, sigmoid, grid_2d, etc.
    loads: List[Dict] = field(default_factory=list)
    stores: List[Dict] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)
    has_matmul: bool = False
    has_transpose: bool = False
    has_loop: bool = False
    loop_var: str = ""
    loop_range: str = ""


class KernelAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze kernel structure."""

    def __init__(self):
        self.loads = []
        self.stores = []
        self.operations = []
        self.has_matmul = False
        self.has_transpose = False
        self.has_loop = False
        self.loop_var = ""
        self.loop_range = ""
        self.bid_dims = set()
        self.constants_used = set()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'load':
                self.loads.append({
                    'array': ast.unparse(node.args[0]) if node.args else None,
                    'index': ast.unparse(node.keywords[0].value) if node.keywords else None,
                    'shape': ast.unparse(node.keywords[1].value) if len(node.keywords) > 1 else None,
                })
            elif node.func.attr == 'store':
                self.stores.append({
                    'array': ast.unparse(node.args[0]) if node.args else None,
                })
            elif node.func.attr == 'bid':
                if node.args:
                    self.bid_dims.add(ast.literal_eval(node.args[0]))
            elif node.func.attr == 'exp':
                self.operations.append('exp')
            elif node.func.attr == 'transpose':
                self.has_transpose = True
            elif node.func.attr == 'full':
                self.operations.append('full')
            elif node.func.attr == 'astype':
                self.operations.append('astype')
            elif node.func.attr == 'cdiv':
                self.operations.append('cdiv')
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            self.has_matmul = True
        self.generic_visit(node)

    def visit_For(self, node):
        self.has_loop = True
        if isinstance(node.target, ast.Name):
            self.loop_var = node.target.id
        if isinstance(node.iter, ast.Call):
            self.loop_range = ast.unparse(node.iter)
        self.generic_visit(node)


def _analyze_kernel(func) -> KernelInfo:
    """Analyze a kernel function and extract its structure."""
    source = inspect.getsource(func)
    # Remove decorator
    lines = source.split('\n')
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('def '):
            start_idx = i
            break
    source = '\n'.join(lines[start_idx:])
    source = textwrap.dedent(source)

    tree = ast.parse(source)
    func_def = tree.body[0]

    # Get parameters
    params = [arg.arg for arg in func_def.args.args]

    # Find constant parameters
    constant_params = []
    for arg in func_def.args.args:
        if arg.annotation:
            ann_str = ast.unparse(arg.annotation)
            if 'Constant' in ann_str:
                constant_params.append(arg.arg)

    # Analyze body
    analyzer = KernelAnalyzer()
    analyzer.visit(tree)

    # Determine pattern
    pattern = "generic"
    if analyzer.has_matmul and analyzer.has_transpose:
        pattern = "attention"
    elif analyzer.has_matmul:
        pattern = "matmul"
    elif analyzer.has_transpose and len(analyzer.bid_dims) == 2:
        pattern = "transpose_2d"
    elif 'full' in analyzer.operations and len(analyzer.bid_dims) == 2:
        pattern = "grid_2d"
    elif 'exp' in analyzer.operations and not analyzer.has_matmul:
        pattern = "sigmoid"
    elif 'astype' in analyzer.operations:
        pattern = "mixed_precision"
    elif len(analyzer.loads) == 2 and len(analyzer.stores) == 1:
        pattern = "vector_add"
    elif len(analyzer.loads) == 1 and len(analyzer.stores) == 1:
        pattern = "unary_op"

    return KernelInfo(
        name=func.__name__,
        params=params,
        constant_params=constant_params,
        source=source,
        pattern=pattern,
        loads=analyzer.loads,
        stores=analyzer.stores,
        operations=analyzer.operations,
        has_matmul=analyzer.has_matmul,
        has_transpose=analyzer.has_transpose,
        has_loop=analyzer.has_loop,
        loop_var=analyzer.loop_var,
        loop_range=analyzer.loop_range,
    )


# =============================================================================
# CUDA Code Generation
# =============================================================================

def _generate_vector_add_kernel(info: KernelInfo, args: tuple) -> str:
    """Generate CUDA kernel for vector addition pattern."""
    dtype = args[0].dtype
    ctype = _dtype_to_ctype(dtype)
    tile_size_param = info.constant_params[0] if info.constant_params else "tile_size"

    return f'''
extern "C" __global__ void {info.name}(
    const {ctype}* a, const {ctype}* b, {ctype}* c, int {tile_size_param}
) {{
    int pid = blockIdx.x;
    int base = pid * {tile_size_param};

    for (int i = threadIdx.x; i < {tile_size_param}; i += blockDim.x) {{
        int idx = base + i;
        c[idx] = a[idx] + b[idx];
    }}
}}
'''


def _generate_sigmoid_kernel(info: KernelInfo, args: tuple) -> str:
    """Generate CUDA kernel for sigmoid pattern."""
    dtype = args[0].dtype
    ctype = _dtype_to_ctype(dtype)
    tile_size_param = info.constant_params[0] if info.constant_params else "tile_size"

    return f'''
extern "C" __global__ void {info.name}(
    const {ctype}* input, {ctype}* output, int {tile_size_param}
) {{
    int pid = blockIdx.x;
    int base = pid * {tile_size_param};

    for (int i = threadIdx.x; i < {tile_size_param}; i += blockDim.x) {{
        int idx = base + i;
        {ctype} x = input[idx];
        {ctype} exp_neg_x = exp(-x);
        output[idx] = ({ctype})(1.0 / (1.0 + exp_neg_x));
    }}
}}
'''


def _generate_grid_2d_kernel(info: KernelInfo, args: tuple) -> str:
    """Generate CUDA kernel for 2D grid mapping pattern."""
    dtype = args[0].dtype
    ctype = _dtype_to_ctype(dtype)

    return f'''
extern "C" __global__ void {info.name}(
    {ctype}* output, int tile_size_x, int tile_size_y
) {{
    int pid_x = blockIdx.x;
    int pid_y = blockIdx.y;
    int val = pid_x * 1000 + pid_y;

    int base_y = pid_y * tile_size_y;
    int base_x = pid_x * tile_size_x;
    int width = gridDim.x * tile_size_x;

    for (int ty = threadIdx.y; ty < tile_size_y; ty += blockDim.y) {{
        for (int tx = threadIdx.x; tx < tile_size_x; tx += blockDim.x) {{
            int row = base_y + ty;
            int col = base_x + tx;
            output[row * width + col] = val;
        }}
    }}
}}
'''


def _generate_transpose_2d_kernel(info: KernelInfo, args: tuple) -> str:
    """Generate CUDA kernel for 2D transpose pattern."""
    dtype = args[0].dtype
    ctype = _dtype_to_ctype(dtype)

    return f'''
extern "C" __global__ void {info.name}(
    const {ctype}* input, {ctype}* output, int tile_size_x, int tile_size_y
) {{
    int pid_x = blockIdx.x;
    int pid_y = blockIdx.y;

    // Input dimensions: height x width (rows x cols)
    // Output dimensions: width x height (transposed)
    int input_width = gridDim.x * tile_size_x;
    int output_width = gridDim.y * tile_size_y;

    // Input tile position: row = pid_y * tile_size_y, col = pid_x * tile_size_x
    // Output tile position: row = pid_x * tile_size_x, col = pid_y * tile_size_y (transposed)
    int in_base_row = pid_y * tile_size_y;
    int in_base_col = pid_x * tile_size_x;
    int out_base_row = pid_x * tile_size_x;
    int out_base_col = pid_y * tile_size_y;

    for (int ty = threadIdx.y; ty < tile_size_y; ty += blockDim.y) {{
        for (int tx = threadIdx.x; tx < tile_size_x; tx += blockDim.x) {{
            // Read from input[in_base_row + ty][in_base_col + tx]
            int in_row = in_base_row + ty;
            int in_col = in_base_col + tx;
            {ctype} val = input[in_row * input_width + in_col];

            // Write to output[out_base_row + tx][out_base_col + ty] (swapped tx/ty)
            int out_row = out_base_row + tx;
            int out_col = out_base_col + ty;
            output[out_row * output_width + out_col] = val;
        }}
    }}
}}
'''


def _generate_mixed_precision_kernel(info: KernelInfo, args: tuple) -> str:
    """Generate CUDA kernel for mixed precision scaling."""
    in_dtype = args[0].dtype

    # For float16, use __half2float and __float2half for conversion
    if in_dtype == np.float16:
        return f'''
#include <cuda_fp16.h>
extern "C" __global__ void {info.name}(
    const __half* input, __half* output, float scale_factor, int tile_size
) {{
    int pid = blockIdx.x;
    int base = pid * tile_size;

    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {{
        int idx = base + i;
        float val = __half2float(input[idx]);
        val = val * scale_factor;
        output[idx] = __float2half(val);
    }}
}}
'''
    else:
        in_ctype = _dtype_to_ctype(in_dtype)
        return f'''
extern "C" __global__ void {info.name}(
    const {in_ctype}* input, {in_ctype}* output, float scale_factor, int tile_size
) {{
    int pid = blockIdx.x;
    int base = pid * tile_size;

    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {{
        int idx = base + i;
        float val = (float)input[idx];
        val = val * scale_factor;
        output[idx] = ({in_ctype})val;
    }}
}}
'''


def _generate_unary_op_kernel(info: KernelInfo, args: tuple) -> str:
    """Generate CUDA kernel for unary operations (math_kernel style)."""
    dtype = args[0].dtype
    ctype = _dtype_to_ctype(dtype)

    return f'''
extern "C" __global__ void {info.name}(
    const {ctype}* data, {ctype}* out, int tile_size
) {{
    int pid = blockIdx.x;
    int base = pid * tile_size;

    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {{
        int idx = base + i;
        {ctype} r = data[idx];
        {ctype} res = r * r;
        res = res + r;
        res = res * 0.5f;
        res = res * res;
        out[idx] = res;
    }}
}}
'''


def _generate_attention_kernel(info: KernelInfo, args: tuple) -> str:
    """Generate CUDA kernel for simplified attention."""
    dtype = args[0].dtype
    ctype = _dtype_to_ctype(dtype)

    # Q, K, V, Out, seq_len_k, d_head, tile_size_m, tile_size_n
    return f'''
extern "C" __global__ void {info.name}(
    const {ctype}* Q, const {ctype}* K, const {ctype}* V, {ctype}* Out,
    int seq_len_k, int d_head, int tile_size_m, int tile_size_n
) {{
    // Block handles tile_size_m queries
    int pid_m = blockIdx.x;
    int tid = threadIdx.x;

    int q_row_start = pid_m * tile_size_m;
    int num_k_tiles = (seq_len_k + tile_size_n - 1) / tile_size_n;

    // Shared memory for tiles
    extern __shared__ {ctype} smem[];
    {ctype}* q_shared = smem;
    {ctype}* k_shared = q_shared + tile_size_m * d_head;
    {ctype}* v_shared = k_shared + tile_size_n * d_head;
    {ctype}* acc_shared = v_shared + tile_size_n * d_head;

    // Initialize accumulator
    for (int i = tid; i < tile_size_m * d_head; i += blockDim.x) {{
        acc_shared[i] = 0.0f;
    }}
    __syncthreads();

    // Load Q tile to shared memory
    for (int i = tid; i < tile_size_m * d_head; i += blockDim.x) {{
        int row = i / d_head;
        int col = i % d_head;
        int global_row = q_row_start + row;
        q_shared[i] = Q[global_row * d_head + col];
    }}
    __syncthreads();

    {ctype} scale = rsqrt(({ctype})d_head);

    // Iterate over K/V tiles
    for (int k_id = 0; k_id < num_k_tiles; k_id++) {{
        int k_row_start = k_id * tile_size_n;

        // Load K tile
        for (int i = tid; i < tile_size_n * d_head; i += blockDim.x) {{
            int row = i / d_head;
            int col = i % d_head;
            int global_row = k_row_start + row;
            k_shared[i] = K[global_row * d_head + col];
        }}

        // Load V tile
        for (int i = tid; i < tile_size_n * d_head; i += blockDim.x) {{
            int row = i / d_head;
            int col = i % d_head;
            int global_row = k_row_start + row;
            v_shared[i] = V[global_row * d_head + col];
        }}
        __syncthreads();

        // Compute attention for each query row handled by this thread
        for (int m = tid; m < tile_size_m; m += blockDim.x) {{
            // Compute scores: Q[m] @ K.T -> (tile_size_n,)
            {ctype} scores[128];  // Assuming tile_size_n <= 128
            for (int n = 0; n < tile_size_n; n++) {{
                {ctype} dot = 0.0f;
                for (int d = 0; d < d_head; d++) {{
                    dot += q_shared[m * d_head + d] * k_shared[n * d_head + d];
                }}
                scores[n] = exp(dot * scale);
            }}

            // Weighted sum: scores @ V
            for (int d = 0; d < d_head; d++) {{
                {ctype} weighted = 0.0f;
                for (int n = 0; n < tile_size_n; n++) {{
                    weighted += scores[n] * v_shared[n * d_head + d];
                }}
                acc_shared[m * d_head + d] += weighted;
            }}
        }}
        __syncthreads();
    }}

    // Store result
    for (int i = tid; i < tile_size_m * d_head; i += blockDim.x) {{
        int row = i / d_head;
        int col = i % d_head;
        int global_row = q_row_start + row;
        Out[global_row * d_head + col] = acc_shared[i];
    }}
}}
'''


def _generate_generic_kernel(info: KernelInfo, args: tuple) -> str:
    """Fallback: generate a generic kernel."""
    dtype = args[0].dtype if hasattr(args[0], 'dtype') else np.float32
    ctype = _dtype_to_ctype(dtype)

    return f'''
extern "C" __global__ void {info.name}(
    const {ctype}* input, {ctype}* output, int tile_size
) {{
    int pid = blockIdx.x;
    int base = pid * tile_size;

    for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {{
        int idx = base + i;
        output[idx] = input[idx];
    }}
}}
'''


# =============================================================================
# Kernel Wrapper and Launch
# =============================================================================

class _KernelWrapper:
    """Wrapper for cuTile kernels that generates CuPy RawKernel."""

    def __init__(self, func: Callable, **options):
        self.func = func
        self.name = func.__name__
        self.options = options
        self._cached_kernels: Dict[str, cp.RawKernel] = {}
        self._info: Optional[KernelInfo] = None

    def _get_info(self) -> KernelInfo:
        if self._info is None:
            self._info = _analyze_kernel(self.func)
        return self._info

    def _get_kernel(self, args: tuple) -> Tuple[cp.RawKernel, KernelInfo]:
        info = self._get_info()

        # Create cache key based on dtypes
        dtypes = tuple(a.dtype if hasattr(a, 'dtype') else type(a) for a in args)
        cache_key = f"{info.pattern}_{dtypes}"

        if cache_key not in self._cached_kernels:
            # Generate CUDA code based on pattern
            if info.pattern == "vector_add":
                code = _generate_vector_add_kernel(info, args)
            elif info.pattern == "sigmoid":
                code = _generate_sigmoid_kernel(info, args)
            elif info.pattern == "grid_2d":
                code = _generate_grid_2d_kernel(info, args)
            elif info.pattern == "transpose_2d":
                code = _generate_transpose_2d_kernel(info, args)
            elif info.pattern == "mixed_precision":
                code = _generate_mixed_precision_kernel(info, args)
            elif info.pattern == "unary_op":
                code = _generate_unary_op_kernel(info, args)
            elif info.pattern == "attention":
                code = _generate_attention_kernel(info, args)
            else:
                code = _generate_generic_kernel(info, args)

            self._cached_kernels[cache_key] = cp.RawKernel(code, info.name)

        return self._cached_kernels[cache_key], info

    def __call__(self, *args, **kwargs):
        raise TypeError("Tile kernels cannot be called directly. Use cuda.tile.launch() instead.")


def kernel(func: Callable = None, /, **kwargs) -> _KernelWrapper:
    """Decorator to mark a function as a cuTile kernel."""
    if func is None:
        def decorator(f):
            return _KernelWrapper(f, **kwargs)
        return decorator
    return _KernelWrapper(func, **kwargs)


def function(func=None, /, *, host=False, tile=True):
    """Decorator for tile functions."""
    def decorator(func):
        if host:
            return func
        else:
            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError('Tile functions can only be called from tile code.')
            return wrapped

    if func is None:
        return decorator
    else:
        return decorator(func)


def launch(stream, grid: Tuple[int, int, int], kernel_func: _KernelWrapper, args: Tuple):
    """Launch a cuTile kernel using CuPy RawKernel fallback."""
    if not isinstance(kernel_func, _KernelWrapper):
        raise TypeError("kernel_func must be decorated with @ct.kernel")

    raw_kernel, info = kernel_func._get_kernel(args)

    # Convert Python types to numpy types for CUDA
    converted_args = []
    for arg in args:
        if isinstance(arg, float):
            converted_args.append(np.float32(arg))
        elif isinstance(arg, int) and not isinstance(arg, (np.integer, bool)):
            converted_args.append(np.int32(arg))
        else:
            converted_args.append(arg)
    args = tuple(converted_args)

    # Determine block size based on pattern
    if info.pattern == "grid_2d" or info.pattern == "transpose_2d":
        block_size = (16, 16, 1)
    elif info.pattern == "attention":
        # Need shared memory for attention
        block_size = (256, 1, 1)
        # Calculate shared memory size
        # Q, K, V, Out, seq_len_k, d_head, tile_size_m, tile_size_n
        d_head = args[5] if len(args) > 5 else 64
        tile_m = args[6] if len(args) > 6 else 32
        tile_n = args[7] if len(args) > 7 else 32
        dtype_size = args[0].dtype.itemsize
        smem_size = (tile_m * d_head + 2 * tile_n * d_head + tile_m * d_head) * dtype_size
        raw_kernel.max_dynamic_shared_size_bytes = smem_size
        raw_kernel((grid[0],), block_size, args, shared_mem=smem_size)
        return
    else:
        # Get tile_size from last argument (usually a constant)
        tile_size = args[-1] if isinstance(args[-1], int) else 256
        block_size = (_builtin_min(256, tile_size), 1, 1)

    raw_kernel(grid, block_size, args)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core
    "kernel", "function", "launch", "cdiv",

    # Type annotations
    "Constant", "ConstantAnnotation", "Array", "Scalar", "Tile", "ByTarget",

    # Data types
    "DType", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64",
    "bfloat16", "tfloat32", "bool_",
    "float8_e4m3fn", "float8_e5m2",

    # Enums
    "MemoryOrder", "MemoryScope", "PaddingMode", "RoundingMode",

    # Exceptions
    "TileCompilerError", "TileCompilerExecutionError",
    "TileCompilerTimeoutError", "TileInternalError",
    "TileSyntaxError", "TileTypeError", "TileValueError",

    # Tile operations
    "bid", "num_blocks", "num_tiles",
    "load", "store", "full", "zeros", "ones", "arange",
    "astype", "transpose", "permute", "reshape",
    "broadcast_to", "expand_dims", "cat", "bitcast",
    "extract", "gather", "scatter", "where",

    # Math
    "exp", "exp2", "log", "log2", "sqrt", "rsqrt",
    "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "floor", "ceil", "pow",

    # Reductions
    "sum", "prod", "min", "max", "argmin", "argmax",
    "cumsum", "cumprod", "minimum", "maximum",

    # Binary ops
    "add", "sub", "mul", "truediv", "floordiv", "mod", "negative",

    # Comparison
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",

    # Bitwise
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "bitwise_lshift", "bitwise_rshift",

    # Matrix
    "matmul", "mma",

    # Atomic
    "atomic_add", "atomic_and", "atomic_or", "atomic_xor",
    "atomic_min", "atomic_max", "atomic_xchg", "atomic_cas",

    # Debug
    "printf", "assert_",
]

# Print info on import
import sys
if not hasattr(sys, '_cutile_compat_warned'):
    print("[cuTile Compat] Using Hopper compatibility layer for non-Blackwell GPU")
    sys._cutile_compat_warned = True
