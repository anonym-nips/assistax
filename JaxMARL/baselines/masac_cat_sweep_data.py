import os
import os.path as osp
import numpy as np
import jax.numpy as jnp
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--load-dirs",
    help="A list of directories containing to be concatenated.",
    nargs="+",
    default=["."],
)
parser.add_argument(
    "--save-metrics",
    help="A list of metrics to save.",
    nargs="+",
    default=["returned_episode_returns"],
)
parser.add_argument(
    "-o",
    "--output-dir",
    help="The directory to save the resulting npy files to. Metrics will be placed in the metrics subdirectory, and hyperparameters placed in the hparams subdirectory.",
)
args = parser.parse_args()


def ignore_dir(d):
    if "." in d:
        # ignores files with extensions
        # ignores hidden dirs like .hydra
        # assumes that the hparam key won't have '.'
        return True
    return False


all_hparams = defaultdict(list)
all_metrics = defaultdict(list)
all_returns = []

for dirpath in args.load_dirs:
    key_dirs = [d for d in os.listdir(dirpath) if not ignore_dir(d)]
    files = os.listdir(dirpath)
    for key_dir in key_dirs:
        hparams = jnp.load(
            os.path.join(dirpath, key_dir, f"hparams.npy"), allow_pickle=True
        ).item()
        returns = jnp.load(
            os.path.join(dirpath, key_dir, f"returns.npy"), allow_pickle=True
        )
        metrics = jnp.load(
            os.path.join(dirpath, key_dir, f"metrics.npy"), allow_pickle=True
        ).item()
        n_configs = metrics["log_probs"].shape[0]
        for hparam_key in hparams:
            all_hparams[hparam_key].append(
                hparams[hparam_key]
                if isinstance(hparams[hparam_key], jnp.ndarray)
                else jnp.full(n_configs, hparams[hparam_key])
            )
        # for metric_key in args.save_metrics:
        #     all_metrics[metric_key].append(metrics[metric_key])
        
        all_returns.append(returns)


all_hparams = {key: np.concatenate(val) for key, val in all_hparams.items()}
# all_hparams = {key: np.concatenate([v if len(np.shape(v)) >= 1 else np.array([v]) for v in val]) 
#                for key, val in all_hparams.items()}
# all_metrics = {key: np.concatenate(val) for key, val in all_metrics.items()}
all_returns = jnp.concat(all_returns)

output_dir = args.output_dir
os.makedirs(osp.join(output_dir, "hparam"), exist_ok=True)
os.makedirs(osp.join(output_dir, "metric"), exist_ok=True)
for key, hparam in all_hparams.items():
    np.save(osp.join(output_dir, "hparam", f"{key}.npy"), hparam)
# for key, metric in all_metrics.items():
#     np.save(osp.join(output_dir, "metric", f"{key}.npy"), metric)
np.save(osp.join(output_dir, "metric", f"eval_returns.npy"), all_returns)
