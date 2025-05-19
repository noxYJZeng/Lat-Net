import os
import sys
import math
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.rc('font', family='DejaVu Sans', weight='normal', size=16)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from absl import flags
from absl import app
from tqdm import tqdm

sys.path.append('../')

from model.lat_net import continual_unroll_template, inputs
from model.lattice import add_lattice, lattice_to_vel, vel_to_norm
from input.lbm_inputs import lbm_inputs

tf.compat.v1.disable_eager_execution()
FLAGS = flags.FLAGS

output_folder = './results/prediction'
os.makedirs(output_folder, exist_ok=True)

base_dir = './checkpoints'

def make_checkpoint_path(base_dir, FLAGS):
    dim_str = "dimensions." + str(FLAGS.dimensions)
    lat_str = "lattice_size." + str(FLAGS.lattice_size)
    sys_str = "system." + str(FLAGS.system)
    restore_dir = os.path.join(base_dir, dim_str, lat_str, sys_str)
    return restore_dir

def evaluate():
    spatial_shape = [512, 512]
    time_sample = [0, 100, 200]

    with tf.Graph().as_default():
        state, boundary = inputs(empty=True, shape=spatial_shape, single_step=True)

        y_1, small_boundary_mul, small_boundary_add, x_2, y_2 = continual_unroll_template(state, boundary)

        x_2_add = add_lattice(x_2)
        state_add = add_lattice(state)
        velocity_generated = lattice_to_vel(x_2_add)
        velocity_norm_generated = vel_to_norm(velocity_generated)
        velocity_true = lattice_to_vel(state_add)
        velocity_norm_true = vel_to_norm(velocity_true)

        variables_to_restore = tf.compat.v1.global_variables()
        saver = tf.compat.v1.train.Saver(variables_to_restore)
        sess = tf.compat.v1.Session()

        RESTORE_DIR = make_checkpoint_path(base_dir, FLAGS)
        ckpt = tf.train.get_checkpoint_state(RESTORE_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring checkpoint from:", ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint found in {}. Exiting.".format(RESTORE_DIR))
            return

        test_ds = lbm_inputs(
            batch_size=4,
            seq_length=1,
            shape=spatial_shape,
            lattice_size=FLAGS.lattice_size,
            train=False
        )
        iterator = tf.compat.v1.data.make_one_shot_iterator(test_ds)
        state_feed, boundary_feed = iterator.get_next()

        state_feed_np, boundary_feed_np = sess.run([state_feed, boundary_feed])
        state_feed_np = np.squeeze(state_feed_np, axis=1)
        boundary_feed_np = np.squeeze(boundary_feed_np, axis=1)

        y_1_g, small_boundary_mul_g, small_boundary_add_g = sess.run(
            [y_1, small_boundary_mul, small_boundary_add],
            feed_dict={state: state_feed_np, boundary: boundary_feed_np}
        )

        d2d = (len(spatial_shape) == 2)
        label_move = 0.99
        title_move = 0.94
        ratio = 1.0
        font_size = 16
        vmax = 0.18

        plt.figure(figsize=(4 * len(time_sample), ratio * 4 * 3))
        gs1 = gridspec.GridSpec(len(time_sample), 3)
        gs1.update(wspace=0.025, hspace=0.025)
        index = 0

        for step in tqdm(range(time_sample[-1] + 1)):
            state_fd_np, boundary_fd_np = sess.run([state_feed, boundary_feed])
            state_fd_np = np.squeeze(state_fd_np, axis=1)
            boundary_fd_np = np.squeeze(boundary_fd_np, axis=1)

            fd = {
                state: state_fd_np,
                boundary: boundary_fd_np,
                y_1: y_1_g,
                small_boundary_mul: small_boundary_mul_g,
                small_boundary_add: small_boundary_add_g
            }

            v_n_g, v_n_t, y2_out = sess.run(
                [velocity_norm_generated, velocity_norm_true, y_2],
                feed_dict=fd
            )

            alpha = 0.1
            y_1_g = (1 - alpha) * y_1_g + alpha * y2_out

            if step in time_sample:
                if not d2d:
                    v_n_g = v_n_g[:, 0]
                    v_n_t = v_n_t[:, 0]
                v_n_g_img = v_n_g[0, :, :, 0]
                v_n_t_img = v_n_t[0, :, :, 0]

                ax1 = plt.subplot(gs1[3 * index + 0])
                ax1.imshow(v_n_g_img, cmap='jet', vmin=0.0, vmax=vmax)
                ax1.set_ylabel("Step " + str(step))
                ax1.axis('off')

                ax2 = plt.subplot(gs1[3 * index + 1])
                ax2.imshow(v_n_t_img, cmap='jet', vmin=0.0, vmax=vmax)
                ax2.axis('off')
                if index == 0:
                    ax2.set_title("True", y=label_move)

                ax3 = plt.subplot(gs1[3 * index + 2])
                ax3.imshow(np.abs(v_n_g_img - v_n_t_img), cmap='jet', vmin=0.0, vmax=vmax)
                ax3.axis('off')
                if index == 0:
                    ax3.set_title("Difference", y=label_move)

                index += 1

        shape_str = "x".join(str(s) for s in spatial_shape)
        plt.suptitle(shape_str + " Simulation", fontsize="x-large", y=title_move)
        output_path = os.path.join(output_folder, shape_str + "_flow_image.png")
        plt.savefig(output_path)
        print("Evaluation done, image saved to:", output_path)

def main(_):
    evaluate()

if __name__ == '__main__':
    tf.compat.v1.app.run()
