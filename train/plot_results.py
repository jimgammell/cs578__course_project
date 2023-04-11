import os
import numpy as np
import pickle
from matplotlib import pyplot as plt
import imageio

def get_figs_dir(trial_dir):
    figs_dir = os.path.join(trial_dir, 'figures')
    os.makedirs(figs_dir, exist_ok=True)
    return figs_dir

def plot_traces(trial_dir, ax_width=4):
    dest_dir = get_figs_dir(trial_dir)
    results_dir = os.path.join(trial_dir, 'results')
    
    def list_available_metrics():
        metrics = []
        for phase in ['train', 'validation']:
            for res_path in os.listdir(os.path.join(results_dir, phase)):
                with open(os.path.join(results_dir, phase, res_path), 'rb') as F:
                    results = pickle.load(F)
                metrics.extend([k for k in results.keys() if 'images' not in k])
        metrics = np.unique(metrics)
        return metrics
    def get_trace(key, phase):
        epochs, vals = [], []
        for res_path in os.listdir(os.path.join(results_dir, phase)):
            with open(os.path.join(results_dir, phase, res_path), 'rb') as F:
                results = pickle.load(F)
            if not key in results.keys():
                continue
            val = results[key]
            epoch = int(res_path.split('.')[0].split('_')[-1])
            epochs.append(epoch)
            vals.append(val)
        epochs, vals = np.array(epochs), np.array(vals)
        if len(epochs) == len(vals) == 0:
            return None
        sorted_indices = np.argsort(epochs)
        epochs = epochs[sorted_indices]
        vals = vals[sorted_indices]
        return epochs, vals
    
    available_metrics = list_available_metrics()
    n_axes = len(available_metrics)
    (fig, axes) = plt.subplots(2, (n_axes//2)+int(n_axes%2), figsize=((n_axes//2)*ax_width, 2*ax_width))
    axes = axes.flatten()
    for metric, ax in zip(available_metrics, axes):
        ax.plot(*get_trace(metric, 'train'), linestyle='--', color='blue')
        ax.plot(*get_trace(metric, 'validation'), linestyle='-', color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('{} over time'.format(metric.capitalize()))
    plt.tight_layout()
    fig.savefig(os.path.join(dest_dir, 'traces.jpg'), dpi=50)
    plt.close('all')

def generate_animation(trial_dir):
    dest_dir = get_figs_dir(trial_dir)
    frames_dir = os.path.join(trial_dir, 'eg_frames')
    assert os.path.exists(frames_dir)
    frames_files = os.listdir(frames_dir)
    sorted_indices = np.argsort([int(f.split('.')[0].split('_')[-1]) for f in frames_files])
    frames_files = [frames_files[idx] for idx in sorted_indices]
    with imageio.get_writer(os.path.join(dest_dir, 'images_over_time.gif'), mode='I', duration=5/len(frames_files)) as writer:
        for frame_file in frames_files:
            image = imageio.imread(os.path.join(frames_dir, frame_file))
            writer.append_data(image)