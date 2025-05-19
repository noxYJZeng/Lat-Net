import os
import sys
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import types

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lat_net import unroll_template, inputs
from model.loss import loss_mse, loss_gradient_difference
from model.optimizer import adam_updates


sys.path.append('../')
tf.compat.v1.disable_eager_execution()

FLAGS = tf.compat.v1.app.flags.FLAGS


FLAGS(sys.argv, known_only=True)

params_for_path = {
    "dimensions": FLAGS.dimensions,
    "lattice_size": FLAGS.lattice_size,
    "system": FLAGS.system
}
params_obj = types.SimpleNamespace(**params_for_path)

def make_checkpoint_path(base_dir, params):
    dim_str = "dimensions." + str(params.dimensions)
    lat_str = "lattice_size." + str(params.lattice_size)
    sys_str = "system." + str(params.system)
    restore_dir = os.path.join(base_dir, dim_str, lat_str, sys_str)
    return restore_dir

TRAIN_DIR = make_checkpoint_path(FLAGS.base_dir, params_obj)
if not tf.compat.v1.gfile.Exists(TRAIN_DIR):
    tf.compat.v1.gfile.MakeDirs(TRAIN_DIR)
print("Checkpoint path:", TRAIN_DIR)

def train():
    with tf.compat.v1.Graph().as_default():
        print("Training on system:", FLAGS.system)
        print("Dimensions:", FLAGS.dimensions, "Lattice Size:", FLAGS.lattice_size)

        loss_history = []
        grads = []
        loss_gen = []

        with tf.device('/cpu:0'):
            global_step = tf.compat.v1.get_variable('global_step', [], initializer=tf.compat.v1.constant_initializer(0), trainable=False)

        device_str = "/gpu:0" if tf.test.is_gpu_available() else "/cpu:0"

        for i in range(FLAGS.nr_gpus):
            print("Unrolling on GPU:", i)
            state, boundary = inputs()
            with tf.device(device_str):
                x_2_o = unroll_template(state, boundary)

                if i == 0:
                    with tf.device('/cpu:0'):
                        if len(x_2_o.get_shape()) == 5:
                            tf.compat.v1.summary.image('generated_d_' + str(i), x_2_o[:, 0, :, :, 0:1])
                            tf.compat.v1.summary.image('generated_d_' + str(i), x_2_o[:, 0, :, :, 2:5])
                            tf.compat.v1.summary.image('true_d_' + str(i), state[:, 0, :, :, 0:1])
                            tf.compat.v1.summary.image('true_d_' + str(i), state[:, 0, :, :, 2:5])
                    all_params = tf.compat.v1.trainable_variables()

                error_mse = loss_mse(state, x_2_o)
                error_gradient = loss_gradient_difference(state, x_2_o)
                error = error_mse + FLAGS.lambda_divergence * error_gradient
                loss_gen.append(error)
                grads.append(tf.compat.v1.gradients(error, all_params))

        ema = tf.compat.v1.train.ExponentialMovingAverage(decay=0.9995)
        maintain_averages_op = tf.group(ema.apply(all_params))

        with tf.device('/cpu:0'):
            for i in range(1, FLAGS.nr_gpus):
                loss_gen[0] += loss_gen[i]
                for j in range(len(grads[0])):
                    grads[0][j] += grads[i][j]

            # === Gradient Clipping ===
            clipped_grads = [
                tf.clip_by_value(g, -1.0, 1.0) if g is not None else None
                for g in grads[0]
            ]

            optimizer = tf.group(
                adam_updates(all_params, clipped_grads, lr=FLAGS.reconstruction_lr, mom1=0.95, mom2=0.9995),
                maintain_averages_op,
                global_step.assign_add(1)
            )

        total_loss = loss_gen[0]
        tf.compat.v1.summary.scalar('total_loss', total_loss)

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=1)
        summary_op = tf.compat.v1.summary.merge_all()
        init = tf.compat.v1.global_variables_initializer()

        sess = tf.compat.v1.Session()
        sess.run(init)

        saver_restore = tf.compat.v1.train.Saver()
        ckpt = tf.compat.v1.train.get_checkpoint_state(TRAIN_DIR)
        if ckpt and FLAGS.restore:
            print("Restoring from", TRAIN_DIR)
            try:
                saver_restore.restore(sess, ckpt.model_checkpoint_path)
            except Exception as e:
                print("Failed to restore checkpoint. Using fresh init.", e)

        tf.compat.v1.train.start_queue_runners(sess=sess)
        summary_writer = tf.compat.v1.summary.FileWriter(TRAIN_DIR, sess.graph)

        t = time.time()
        run_steps = FLAGS.max_steps - int(sess.run(global_step))
        print("Training for {} steps".format(run_steps))

        batch_loss_sum = 0.0
        try:
            for step in range(run_steps):
                current_step = int(sess.run(global_step))
                try:
                    _, loss_value = sess.run([optimizer, total_loss])

                    if not np.isfinite(loss_value):
                        print(f"[Warning] Non-finite loss at step {current_step}: {loss_value}")
                        with open("bad_batches.log", "a") as f:
                            f.write(f"Step {current_step}, loss: {loss_value}\n")
                        continue

                    batch_loss_sum += loss_value
                    loss_history.append((current_step, loss_value))

                except tf.errors.InvalidArgumentError as e:
                    print(f"[InvalidArgumentError] Step {current_step}: {e}")
                    with open("bad_batches.log", "a") as f:
                        f.write(f"Step {current_step}, InvalidArgumentError: {str(e)}\n")
                    continue

                except Exception as e:
                    print(f"[Unexpected Error] Step {current_step}: {e}")
                    with open("bad_batches.log", "a") as f:
                        f.write(f"Step {current_step}, Unexpected Error: {str(e)}\n")
                    continue

                if current_step % 100 == 0:
                    avg_loss = batch_loss_sum / 100.0
                    print("Step {}: Avg Loss = {:.6f}, Time = {:.3f}s".format(
                        current_step, avg_loss / FLAGS.batch_size, time.time() - t))
                    t = time.time()
                    batch_loss_sum = 0.0

                if (current_step + 1) % 1000 == 0:
                    time.sleep(1)
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, current_step)
                    checkpoint_path = os.path.join(TRAIN_DIR, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    print("Saved checkpoint to", TRAIN_DIR)

        except KeyboardInterrupt:
            print("Training interrupted manually.")
        finally:
            sess.close()


def main(argv=None):
    if not tf.compat.v1.gfile.Exists(TRAIN_DIR):
        tf.compat.v1.gfile.MakeDirs(TRAIN_DIR)
    if tf.compat.v1.gfile.Exists(TRAIN_DIR) and not FLAGS.restore:
        tf.compat.v1.gfile.DeleteRecursively(TRAIN_DIR)
    train()

if __name__ == '__main__':
    tf.compat.v1.app.run()