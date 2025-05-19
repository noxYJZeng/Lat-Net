
import os
import fnmatch
import argparse

# Flags that should NOT be included in the checkpoint directory path
non_checkpoint_flags = [
    'min_queue_examples', 'data_dir', 'tf_data_dir', 'num_preprocess_threads', 'train',
    'base_dir', 'restore', 'max_steps', 'restore_unroll_length', 'batch_size',
    'unroll_from_true', 'unroll_length', 'video_shape', 'video_length', 'test_length',
    'test_nr_runs', 'test_nr_per_simulation', 'test_dimensions', 'lstm', 'gan',
    'nr_discriminators', 'z_size', 'nr_downsamples_discriminator', 'nr_residual_discriminator',
    'keep_p_discriminator', 'filter_size_discriminator', 'lstm_size_discriminator',
    'lambda_reconstruction', 'nr_gpus', 'tf_store_images', 'gan_lr', 'init_unroll_length',
    'tf_seq_length', 'extract_type', 'extract_pos'
]

def str2bool(v):
    return str(v).lower() == 'true'

def make_checkpoint_path(base_path, args):
    """
    Build a unique checkpoint path based on the arguments,
    excluding non_checkpoint_flags.
    """
    args_dict = vars(args)
    for key in sorted(args_dict.keys()):
        if key not in non_checkpoint_flags:
            base_path = os.path.join(base_path, f"{key}.{args_dict[key]}")
    return base_path

def list_all_checkpoints(base_path):
    """
    Recursively find all folders with a 'checkpoint' file inside.
    """
    paths = []
    for root, _, filenames in os.walk(base_path):
        for filename in fnmatch.filter(filenames, 'checkpoint'):
            rel_path = os.path.relpath(root, base_path)
            paths.append(rel_path)
    return paths

def set_flags_given_checkpoint_path(path, args):
    """
    Update argparse args based on parsed values from checkpoint path string.
    """
    split_path = path.split(os.sep)
    for param in split_path:
        parts = param.split('.')
        if len(parts) < 2:
            continue
        param_name = parts[0]
        param_value = '.'.join(parts[1:])
        current_val = getattr(args, param_name, None)
        if current_val is not None:
            if isinstance(current_val, bool):
                setattr(args, param_name, str2bool(param_value))
            else:
                try:
                    setattr(args, param_name, type(current_val)(param_value))
                except Exception:
                    print(f"Warning: Could not cast {param_name}={param_value} to {type(current_val)}")
    return args

def make_flags_string_given_checkpoint_path(path):
    """
    Convert checkpoint path to string of command-line flags.
    """
    flag_string = ''
    split_path = path.split(os.sep)
    for param in split_path:
        parts = param.split('.')
        if len(parts) < 2:
            continue
        param_name = parts[0]
        param_value = '.'.join(parts[1:])
        flag_string += f" --{param_name}={param_value}"
    return flag_string
