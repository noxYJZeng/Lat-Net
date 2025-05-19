import os
import numpy as np
import tensorflow as tf
from glob import glob as glb
from absl import flags

FLAGS = flags.FLAGS

LBM_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'npz')
MIN_QUEUE_EXAMPLES = 200
NUM_PREPROCESS_THREADS = 2

def convert_to_lattice_np(macroscopic_flow, lattice_size=9):
    if lattice_size != 9:
        raise ValueError("Only D2Q9 (lattice_size=9) is currently supported.")

    weights = np.array([4/9., 1/9., 1/9., 1/9., 1/9., 1/36., 1/36., 1/36., 1/36.], dtype=np.float32)
    c = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [-1, 0],
                  [0, -1],
                  [1, 1],
                  [-1, 1],
                  [-1, -1],
                  [1, -1]], dtype=np.float32)

    H, W, _ = macroscopic_flow.shape
    f_eq = np.zeros((H, W, lattice_size), dtype=np.float32)

    u_x = macroscopic_flow[..., 0:1]
    u_y = macroscopic_flow[..., 1:2]
    rho = macroscopic_flow[..., 2:3]

    rho = np.clip(rho, 0, 100)
    u_x = np.clip(u_x, -1, 1)
    u_y = np.clip(u_y, -1, 1)
    u_sq = np.square(u_x) + np.square(u_y)

    for i in range(lattice_size):
        cu = c[i, 0] * u_x + c[i, 1] * u_y
        term = 1 + 3 * cu + 4.5 * np.square(cu) - 1.5 * u_sq
        term = np.clip(term, -1e4, 1e4)
        f_eq[..., i] = np.squeeze(weights[i] * rho * term, axis=-1)

    return np.clip(f_eq, -1e4, 1e4)

def read_data_lbm(seq_length, shape, lattice_size=9):
    file_list = sorted(glb(os.path.join(LBM_DATA_DIR, '*.npz')))
    if not file_list:
        raise ValueError(f"No .npz files found in directory: {LBM_DATA_DIR}")

    dataset = tf.data.Dataset.from_tensor_slices(file_list)

    def _parse_npz(filename):
        def _load_npz(path):
            path_str = path.numpy().decode('utf-8')
            try:
                data = np.load(path_str, allow_pickle=True)
            except Exception as e:
                raise RuntimeError(f"Error loading {path_str}: {e}")

            velocity = data['velocity']
            if velocity.ndim == 4:
                velocity = velocity[0]
            if velocity.ndim == 3 and velocity.shape[0] == 2:
                velocity = np.transpose(velocity, (1, 2, 0))

            density = data['density']
            if density.ndim == 3:
                density = density[0]
            density = np.expand_dims(density, axis=-1)

            macroscopic = np.concatenate([velocity, density], axis=-1)
            f_eq = convert_to_lattice_np(macroscopic, lattice_size)

            if 'lattice_map' in data:
                boundary = np.expand_dims(data['lattice_map'], axis=-1)
                boundary = np.tile(boundary, (1, 1, 4))
            else:
                H, W = density.shape[:2]
                boundary = np.zeros((H, W, 4), dtype=np.float32)

            return f_eq.astype(np.float32), boundary.astype(np.float32)

        state, boundary = tf.py_function(_load_npz, [filename], [tf.float32, tf.float32])
        dims = list(map(int, shape))
        state.set_shape(dims + [lattice_size])
        boundary.set_shape(dims + [FLAGS.boundary_size])
        return state, boundary

    dataset = dataset.map(_parse_npz, num_parallel_calls=NUM_PREPROCESS_THREADS)
    dataset = dataset.batch(seq_length, drop_remainder=True)
    return dataset

def lbm_inputs(batch_size, seq_length, shape, lattice_size=9, train=True):
    dataset = read_data_lbm(seq_length, shape, lattice_size)
    if train:
        dataset = dataset.shuffle(buffer_size=MIN_QUEUE_EXAMPLES)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    return dataset
