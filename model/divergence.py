
import tensorflow as tf
import numpy as np

from model.nn import int_shape, simple_conv_2d, simple_conv_3d

_simple_conv_2d = simple_conv_2d


def spatial_divergence_2d(field):
    """
    Compute the spatial divergence for a 2D field.
    Assumes input shape [sequence_length, height, width, channels].
    Removes sequence dimension, applies finite difference filters in x and y,
    and returns the absolute divergence with edge cells removed.
    """
    shape = int_shape(field)  # [seq_len, h, w, c]
    field = tf.reshape(field, [shape[0] * shape[1], shape[2], shape[3], shape[4]])

    weight_x = tf.constant(np.array([
        [[[-0.5]], [[0.0]], [[0.5]]]
    ], dtype=np.float32))  # shape [3, 1, 1, 1]

    weight_y = tf.constant(np.array([
        [[[-0.5]], [[0.0]], [[0.5]]]
    ], dtype=np.float32)).transpose([1, 0, 2, 3])  # shape [1, 3, 1, 1]

    dx = simple_conv_2d(field, weight_x)
    dy = simple_conv_2d(field, weight_y)

    divergence = tf.abs(dx[:, 1:-1, 1:-1, :] + dy[:, 1:-1, 1:-1, :])

    return divergence
