import tensorflow as tf
import numpy as np
from absl import flags


tf.compat.v1.disable_eager_execution()
FLAGS = flags.FLAGS

def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    """
    Implements manual Adam optimizer updates.
    Args:
        params: list of model variables
        cost_or_grads: either loss tensor or list of gradients
        lr: learning rate
        mom1: momentum term beta1
        mom2: momentum term beta2
    Returns:
        A tf.group of update operations
    """
    updates = []

    if not isinstance(cost_or_grads, list):
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads

    t = tf.compat.v1.get_variable(
        name='adam_t',
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False
    )

    for p, g in zip(params, grads):
        param_name = p.name.split(":")[0]

        mg = tf.compat.v1.get_variable(
            name=param_name + '_adam_mg',
            shape=p.get_shape().as_list(),
            dtype=tf.float32,
            initializer=tf.zeros_initializer(),
            trainable=False
        )

        if mom1 > 0:
            v = tf.compat.v1.get_variable(
                name=param_name + '_adam_v',
                shape=p.get_shape().as_list(),
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=False
            )
            v_t = mom1 * v + (1.0 - mom1) * g
            v_hat = v_t / (1.0 - tf.pow(mom1, t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g

        mg_t = mom2 * mg + (1.0 - mom2) * tf.square(g)
        mg_hat = mg_t / (1.0 - tf.pow(mom2, t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t

        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))

    updates.append(t.assign_add(1.0))
    return tf.group(*updates)
