
"""functions used to construct different losses
"""

import tensorflow as tf
import numpy as np
from .divergence import *


def loss_mse(true, generated):
    """Mean-squared error loss between true and generated fields."""
    return tf.nn.l2_loss(true - generated)

def loss_divergence(true_field, generated_field):
    """
    Enforces divergence-free velocity field by comparing divergence magnitudes.
    Assumes input shape: [batch, sequence, height, width, channels].
    """
    true_div = spatial_divergence_2d(true_field)
    pred_div = spatial_divergence_2d(generated_field)
    return tf.abs(tf.nn.l2_loss(true_div) - tf.nn.l2_loss(pred_div))

def loss_gradient_difference(true, generated):
    """
    Gradient difference loss based on local spatial derivatives.
    Input shape: [batch, sequence, height, width, channels]
    """
    true_dx = tf.abs(true[:, :, :, 1:, :] - true[:, :, :, :-1, :])
    pred_dx = tf.abs(generated[:, :, :, 1:, :] - generated[:, :, :, :-1, :])
    loss_x = tf.nn.l2_loss(true_dx - pred_dx)

    true_dy = tf.abs(true[:, :, 1:, :, :] - true[:, :, :-1, :, :])
    pred_dy = tf.abs(generated[:, :, 1:, :, :] - generated[:, :, :-1, :, :])
    loss_y = tf.nn.l2_loss(true_dy - pred_dy)

    return loss_x + loss_y

def loss_gan_true(true_label, generated_label):
    d_true = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=true_label, labels=tf.ones_like(true_label)))
    d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=generated_label, labels=tf.zeros_like(generated_label)))
    d_loss = d_true + d_fake

    tf.summary.scalar('discriminator_true_loss', d_true)
    tf.summary.scalar('discriminator_fake_loss', d_fake)
    tf.summary.scalar('discriminator_total_loss', d_loss)
    return d_loss

def loss_gan_generated(generated_label):
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=generated_label, labels=tf.ones_like(generated_label)))
    tf.summary.scalar('generator_loss', g_loss)
    return g_loss

def loss_ssim(true, pred, max_val=1.0):
    ssim = tf.image.ssim(true, pred, max_val=max_val)
    return 1.0 - tf.reduce_mean(ssim)

def total_loss(true, pred, alpha=0.8, beta=0.2):
    mse = loss_mse(true, pred)
    ssim_component = loss_ssim(true, pred, max_val=1.0)
    return alpha * mse + beta * ssim_component
