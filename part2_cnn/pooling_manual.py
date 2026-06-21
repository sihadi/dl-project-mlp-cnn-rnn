"""Small educational script to demonstrate pooling."""

from __future__ import annotations

import numpy as np


def _pool2d(input_array, pool_size: int = 2, stride: int = 2, reducer='max'):
    input_array = np.asarray(input_array, dtype=np.float32)
    if input_array.ndim != 2:
        raise ValueError('input_array must be 2D')
    out_height = (input_array.shape[0] - pool_size) // stride + 1
    out_width = (input_array.shape[1] - pool_size) // stride + 1
    output = np.zeros((out_height, out_width), dtype=np.float32)
    for row in range(out_height):
        for col in range(out_width):
            window = input_array[row * stride: row * stride + pool_size, col * stride: col * stride + pool_size]
            if reducer == 'max':
                output[row, col] = float(np.max(window))
            else:
                output[row, col] = float(np.mean(window))
    return output


def max_pool2d(input_array, pool_size: int = 2, stride: int = 2):
    return _pool2d(input_array, pool_size=pool_size, stride=stride, reducer='max')


def avg_pool2d(input_array, pool_size: int = 2, stride: int = 2):
    return _pool2d(input_array, pool_size=pool_size, stride=stride, reducer='avg')


if __name__ == '__main__':
    x = np.array([[1, 2, 0, 4], [0, 1, 3, 2], [1, 2, 2, 1], [0, 1, 0, 3]], dtype=np.float32)
    print(max_pool2d(x))
    print(avg_pool2d(x))
