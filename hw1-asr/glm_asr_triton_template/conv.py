"""
Triton 1D Convolution Implementation
End-to-end implementation using Triton kernels
Uses im2col approach to convert convolution to matrix multiplication.
"""

from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


# ============================================================================
# Triton Kernels for Conv1d via Matrix Multiplication
# ============================================================================

@triton.jit
def conv1d_matmul_kernel(
    col_ptr,
    weight_ptr,
    output_ptr,
    out_channels,
    col_size,
    out_length,
    stride_col0,
    stride_col1,
    stride_col2,
    stride_w0,
    stride_w1,
    stride_out0,
    stride_out1,
    stride_out2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Conv1d via matrix multiplication after im2col transformation.
    Grid: (batch,)
    """
    pid_b = tl.program_id(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    w = tl.load(
        weight_ptr + offs_m[:, None] * stride_w0 + offs_k[None, :] * stride_w1,
        mask=(offs_m[:, None] < out_channels) & (offs_k[None, :] < col_size),
        other=0.0,
    )
    col = tl.load(
        col_ptr
        + pid_b * stride_col0
        + offs_k[:, None] * stride_col1
        + offs_n[None, :] * stride_col2,
        mask=(offs_k[:, None] < col_size) & (offs_n[None, :] < out_length),
        other=0.0,
    )

    out = tl.dot(w, col)
    tl.store(
        output_ptr
        + pid_b * stride_out0
        + offs_m[:, None] * stride_out1
        + offs_n[None, :] * stride_out2,
        out,
        mask=(offs_m[:, None] < out_channels) & (offs_n[None, :] < out_length),
    )


# ============================================================================
# Conv1d Classes
# ============================================================================

def im2col_1d(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """
    Convert input to column format for efficient convolution.

    Args:
        x: Input (batch, in_channels, length) - already padded
        kernel_size: Size of convolution kernel
        stride: Stride for convolution

    Returns:
        Column matrix (batch, in_channels * kernel_size, out_length)
    """
    batch, in_channels, length = x.shape
    out_length = (length - kernel_size) // stride + 1

    stride_b, stride_c, stride_l = x.stride()
    shape = (batch, in_channels, kernel_size, out_length)
    strides = (stride_b, stride_c, stride_l, stride_l * stride)

    col = torch.as_strided(x, size=shape, stride=strides)
    col = col.reshape(batch, in_channels * kernel_size, out_length).contiguous()
    return col


class Conv1d:
    """
    1D Convolution using im2col + Triton matrix multiplication.
    """

    MAX_TILE_DIM = 256

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.has_bias = bias

        self.col_size = in_channels * kernel_size
        self.col_size_padded = next_power_of_two(self.col_size)
        self.out_channels_padded = next_power_of_two(out_channels)

        self.use_triton = (
            self.col_size_padded <= self.MAX_TILE_DIM
            and self.out_channels_padded <= self.MAX_TILE_DIM
        )

        k = 1.0 / (in_channels * kernel_size)
        weight = torch.empty(
            (out_channels, in_channels, kernel_size), dtype=torch.float32
        ).uniform_(-np.sqrt(k), np.sqrt(k))

        self.weight = weight.reshape(out_channels, self.col_size)

        if self.use_triton and (
            self.col_size_padded != self.col_size
            or self.out_channels_padded != out_channels
        ):
            self.weight_padded = torch.zeros(
                (self.out_channels_padded, self.col_size_padded), dtype=torch.float32
            )
            self.weight_padded[:out_channels, : self.col_size] = self.weight
        else:
            self.weight_padded = self.weight

        if bias:
            self.bias = torch.zeros(out_channels, dtype=torch.float32)
        else:
            self.bias = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using im2col + matrix multiply.
        """
        batch, in_channels, length = x.shape

        out_length = (length + 2 * self.padding - self.kernel_size) // self.stride + 1

        if self.padding > 0:
            x_padded = torch.nn.functional.pad(
                x,
                (self.padding, self.padding),
                mode="constant",
                value=0.0,
            )
        else:
            x_padded = x

        x_padded = x_padded.to(torch.float32).contiguous()

        col = im2col_1d(x_padded, self.kernel_size, self.stride)

        out_length_padded = next_power_of_two(out_length)

        can_use_triton = self.use_triton and out_length_padded <= self.MAX_TILE_DIM and x.is_cuda

        if can_use_triton:
            if self.col_size_padded != self.col_size or out_length_padded != out_length:
                col_padded = torch.zeros(
                    (batch, self.col_size_padded, out_length_padded),
                    dtype=torch.float32,
                    device=x.device,
                )
                col_padded[:, : self.col_size, : out_length] = col
                col = col_padded
            else:
                col = col.to(x.device)

            if self.weight_padded.device != x.device:
                self.weight_padded = self.weight_padded.to(x.device)

            output_padded = torch.empty(
                (batch, self.out_channels_padded, out_length_padded),
                dtype=torch.float32,
                device=x.device,
            )

            grid = (batch,)
            conv1d_matmul_kernel[grid](
                col,
                self.weight_padded,
                output_padded,
                self.out_channels_padded,
                self.col_size_padded,
                out_length_padded,
                col.stride(0),
                col.stride(1),
                col.stride(2),
                self.weight_padded.stride(0),
                self.weight_padded.stride(1),
                output_padded.stride(0),
                output_padded.stride(1),
                output_padded.stride(2),
                BLOCK_M=self.out_channels_padded,
                BLOCK_N=out_length_padded,
                BLOCK_K=self.col_size_padded,
            )

            output = output_padded[:, : self.out_channels, : out_length].contiguous()
        else:
            if self.weight.device != x.device:
                self.weight = self.weight.to(x.device)
            output = torch.einsum("oc,bcl->bol", self.weight, col)

        if self.has_bias and self.bias is not None:
            if self.bias.device != output.device:
                self.bias = self.bias.to(output.device)
            output = output + self.bias[None, :, None]

        return output


class Conv1dSubsampler:
    """
    Audio feature subsampling using two Conv1d layers.
    Reduces time dimension by factor of 4 (stride 2 x 2).
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: tuple = (3, 3),
    ):
        self.conv1 = Conv1d(
            in_channels,
            mid_channels,
            kernel_size=kernel_sizes[0],
            stride=2,
            padding=kernel_sizes[0] // 2,
        )
        self.conv2 = Conv1d(
            mid_channels,
            out_channels,
            kernel_size=kernel_sizes[1],
            stride=2,
            padding=kernel_sizes[1] // 2,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GELU activation between layers.
        """
        x = self.conv1(x)
        x = gelu(x)
        x = self.conv2(x)
        x = gelu(x)
        return x


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation using tanh approximation."""
    return 0.5 * x * (1.0 + torch.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
    ))


if __name__ == "__main__":
    print("Testing Triton Conv1d...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nSmall Conv1d (Triton kernel):")
    batch_size = 2
    in_channels = 16
    length = 32
    out_channels = 32
    kernel_size = 3

    conv = Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1)
    x = torch.randn(batch_size, in_channels, length, device=device)
    print(f"  Using Triton: {conv.use_triton}")
    y = conv(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")

    print("\nLarger Conv1d (may use Torch fallback):")
    in_channels_large = 80
    out_channels_large = 256

    conv_large = Conv1d(in_channels_large, out_channels_large, kernel_size, stride=1, padding=1)
    x_large = torch.randn(batch_size, in_channels_large, length, device=device)
    print(f"  Using Triton: {conv_large.use_triton}")
    y_large = conv_large(x_large)
    print(f"  Input shape: {x_large.shape}")
    print(f"  Output shape: {y_large.shape}")

    print("\nStrided Conv1d (stride=2):")
    conv_strided = Conv1d(in_channels, out_channels, kernel_size, stride=2, padding=1)
    y_strided = conv_strided(x)
    expected_len = (length + 2 - kernel_size) // 2 + 1
    print(f"  Output shape: {y_strided.shape}")
    print(f"  Expected length: {expected_len}")

    print("\nOutput statistics:")
    print(f"  Mean: {float(y.mean()):.4f}")
    print(f"  Std:  {float(y.std()):.4f}")

    print("\nTriton Conv1d working!")
