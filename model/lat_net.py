
"""Builds the ring network.

Summary of available functions:

  # Compute pics of the simulation runnig.
  
  # Create a graph to train on.
"""


import tensorflow as tf
import numpy as np
from absl import flags

from .nn import *
from .divergence import spatial_divergence_2d
from input.lbm_inputs import * 

tf.compat.v1.disable_eager_execution()
FLAGS = flags.FLAGS

# Define flags safely
def define_flag_safe(flag_fn, name, default, help_text):
    try:
        flag_fn(name, default, help_text)
    except flags.DuplicateFlagError:
        pass

# System

define_flag_safe(flags.DEFINE_string, 'base_dir', './checkpoints', 'Base directory for saving model checkpoints')
define_flag_safe(flags.DEFINE_string, 'system', 'fluid_flow', "system to compress")
define_flag_safe(flags.DEFINE_integer, 'lattice_size', 9, "lattice size")
define_flag_safe(flags.DEFINE_integer, 'boundary_size', 4, "boundary size")
define_flag_safe(flags.DEFINE_string, 'dimensions', '512x512', "dimensions")

# Model
define_flag_safe(flags.DEFINE_integer, 'nr_residual', 2, "residual blocks")
define_flag_safe(flags.DEFINE_integer, 'nr_downsamples', 4, "downsamples")
define_flag_safe(flags.DEFINE_string, 'nonlinearity', "relu", "nonlinearity type")
define_flag_safe(flags.DEFINE_float, 'keep_p', 1.0, "dropout keep prob")
define_flag_safe(flags.DEFINE_bool, 'gated', False, "gated res blocks")
define_flag_safe(flags.DEFINE_integer, 'filter_size', 16, "initial filter size")
define_flag_safe(flags.DEFINE_bool, 'restore', True, 'Whether to restore from latest checkpoint')
define_flag_safe(flags.DEFINE_integer, 'nr_gpus', 1, 'number of gpus for training')


define_flag_safe(flags.DEFINE_bool, 'lstm', False, "use LSTM")
define_flag_safe(flags.DEFINE_integer, 'nr_residual_compression', 3, "res compression")
define_flag_safe(flags.DEFINE_integer, 'filter_size_compression', 128, "compression filters")

# Training
define_flag_safe(flags.DEFINE_integer, 'unroll_length', 5, "unroll length")
define_flag_safe(flags.DEFINE_integer, 'init_unroll_length', 0, "init unroll len")
define_flag_safe(flags.DEFINE_integer, 'batch_size', 4, "batch size")
define_flag_safe(flags.DEFINE_bool, 'train', True, "train or test")
define_flag_safe(flags.DEFINE_float, 'lambda_divergence', 0.2, "weight of divergence or gradient difference error")
define_flag_safe(flags.DEFINE_float, 'reconstruction_lr', 0.001, "learning rate for reconstruction")
define_flag_safe(flags.DEFINE_integer, 'max_steps', 1000000, "maximum number of training steps")

# Templates
encode_state_template = tf.compat.v1.make_template('encode_state_template', lambda x: encoding(x, name='state'))
encode_boundary_template = tf.compat.v1.make_template('encode_boundary_template', lambda x: encoding(x, name='boundary', boundary=True))
compress_template = tf.compat.v1.make_template('compress_template', lambda x: compression(x))
decoding_template = tf.compat.v1.make_template('decoding_template', lambda x, extract_type=None, extract_pos=64: decoding(x, extract_type, extract_pos))
unroll_template = tf.compat.v1.make_template('unroll_template', lambda state, boundary, z=None: unroll(state, boundary, z))
continual_unroll_template = tf.compat.v1.make_template('continual_unroll_template', lambda state, boundary, z=None, extract_type=None, extract_pos=None: continual_unroll(state, boundary, z, extract_type, extract_pos))

def inputs(empty=False, name="inputs", shape=None, batch_size=None, single_step=False):
    if shape is None:
        shape = list(map(int, FLAGS.dimensions.split('x')))
    if batch_size is None:
        batch_size = FLAGS.batch_size
    if empty:
        if single_step:
            state = tf.compat.v1.placeholder(tf.float32, [batch_size] + shape + [FLAGS.lattice_size], name=name + "_state")
            boundary = tf.compat.v1.placeholder(tf.float32, [batch_size] + shape + [FLAGS.boundary_size], name=name + "_boundary")
        else:
            state = tf.compat.v1.placeholder(tf.float32, [batch_size, FLAGS.unroll_length] + shape + [FLAGS.lattice_size], name=name + "_state")
            boundary = tf.compat.v1.placeholder(tf.float32, [batch_size, 1] + shape + [FLAGS.boundary_size], name=name + "_boundary")
        return state, boundary
    else:
        dataset = lbm_inputs(batch_size, FLAGS.unroll_length, shape, lattice_size=FLAGS.lattice_size, train=FLAGS.train)
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        return iterator.get_next()

def encoding(inputs, name='', boundary=False):
    x_i = inputs
    nonlinearity = set_nonlinearity(FLAGS.nonlinearity)
    padding = "same"
    for i in range(FLAGS.nr_downsamples):
        filter_size = FLAGS.filter_size * (2 ** i)
        x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity,
                        keep_p=FLAGS.keep_p, stride=2, gated=FLAGS.gated,
                        padding=padding, name=f"{name}_down_{i}_res0", begin_nonlinearity=False)
        for j in range(FLAGS.nr_residual - 1):
            x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity,
                            keep_p=FLAGS.keep_p, stride=1, gated=FLAGS.gated,
                            padding=padding, name=f"{name}_down_{i}_res{j + 1}")
    final_filter_size = FLAGS.filter_size_compression * (2 if boundary else 1)
    x_i = res_block(x_i, filter_size=final_filter_size, nonlinearity=nonlinearity,
                    keep_p=FLAGS.keep_p, stride=1, gated=FLAGS.gated,
                    padding=padding, name=name + "_final")
    return x_i

def compression(inputs):
    x_i = inputs
    nonlinearity = set_nonlinearity(FLAGS.nonlinearity)
    for i in range(FLAGS.nr_residual_compression):
        x_i = res_block(x_i, filter_size=FLAGS.filter_size_compression, nonlinearity=nonlinearity,
                        keep_p=FLAGS.keep_p, stride=1, gated=FLAGS.gated,
                        padding="same", name=f"compression_res{i}")
    return x_i

def decoding(inputs, extract_type=None, extract_pos=64):
    x_i = inputs
    nonlinearity = set_nonlinearity(FLAGS.nonlinearity)
    padding = "same"
    if extract_type:
        width = (FLAGS.nr_downsamples - 1) * FLAGS.nr_residual * 2
        extract_pos = extract_pos or (width + 1)
        x_i = trim_tensor(x_i, extract_pos, width, extract_type)
    for i in range(FLAGS.nr_downsamples - 1):
        filter_size = FLAGS.filter_size * (2 ** (FLAGS.nr_downsamples - i - 2))
        x_i = transpose_conv_layer(x_i, 4, 2, filter_size, padding, f"up_conv_{i}")
        for j in range(FLAGS.nr_residual):
            x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity,
                            keep_p=FLAGS.keep_p, stride=1, gated=FLAGS.gated,
                            padding=padding, name=f"up_{i}_res{j + 1}")
            if extract_type:
                width -= 2
                x_i = trim_tensor(x_i, width + 2, width, extract_type)
    x_i = transpose_conv_layer(x_i, 4, 2, FLAGS.lattice_size, padding, "final_up")
    return tf.nn.tanh(x_i)

def unroll(state, boundary, z=None):
    total_unroll = FLAGS.init_unroll_length + FLAGS.unroll_length
    if FLAGS.lstm:
        raise NotImplementedError("LSTM unroll is not implemented.")
    x_out = []
    y = encode_state_template(state[:, 0])
    small_boundary = encode_boundary_template(boundary[:, 0])
    mul, add = tf.split(small_boundary, 2, axis=-1)
    y = mul * y + add
    for i in range(FLAGS.unroll_length):
        x_ = decoding_template(y)
        x_out.append(x_)
        if FLAGS.unroll_length > 1:
            y = compress_template(y)
            y = mul * y + add
    x_out = tf.stack(x_out, axis=0)
    perm = np.concatenate([[1, 0], np.arange(2, len(x_out.shape))], axis=0)
    return tf.transpose(x_out, perm=perm)

def continual_unroll(state, boundary, z=None, extract_type=None, extract_pos=None):
    if FLAGS.lstm:
        raise NotImplementedError("LSTM continual unroll is not implemented.")
    y = encode_state_template(state)
    small_boundary = encode_boundary_template(boundary)
    mul, add = tf.split(small_boundary, 2, axis=-1)
    y_boundary = mul * y + add
    x = decoding_template(y_boundary, extract_type=extract_type, extract_pos=extract_pos)
    y_next = compress_template(y_boundary)
    return y, mul, add, x, y_next
