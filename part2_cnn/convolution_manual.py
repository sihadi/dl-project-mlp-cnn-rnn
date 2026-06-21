"""Small educational script to demonstrate 2D cross-correlation."""

from __future__ import annotations

import numpy as np


def cross_correlation_2d(input_array, kernel, stride: int = 1, padding: int = 0):
    input_array = np.asarray(input_array, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    if input_array.ndim != 2 or kernel.ndim != 2:
        raise ValueError('input_array and kernel must be 2D')

    padded = np.pad(input_array, ((padding, padding), (padding, padding)), mode='constant')
    kernel_height, kernel_width = kernel.shape
    out_height = (padded.shape[0] - kernel_height) // stride + 1
    out_width = (padded.shape[1] - kernel_width) // stride + 1
    output = np.zeros((out_height, out_width), dtype=np.float32)

    for row in range(out_height):
        for col in range(out_width):
            window = padded[row * stride: row * stride + kernel_height, col * stride: col * stride + kernel_width]
            output[row, col] = float(np.sum(window * kernel))
    return output


def output_size(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
    return (input_size + 2 * padding - kernel_size) // stride + 1


if __name__ == '__main__':
    x = np.array([[1, 2, 0], [0, 1, 3], [1, 2, 2]], dtype=np.float32)
    k = np.array([[1, 0], [0, -1]], dtype=np.float32)
    print(cross_correlation_2d(x, k))
