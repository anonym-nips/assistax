import os
# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_triton_softmax_fusion=true "
#     "--xla_gpu_triton_gemm_any=true "
#     "--xla_dump_to=xla_dump "
# )
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.95"
import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
import safetensors.flax
import optax
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
from jaxmarl.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Callable
from flax import struct

@struct.dataclass
class EvalNetworkState:
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict

def _tree_take(pytree, indices, axis=None):
    return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)

def _unstack_tree(pytree):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = zip(*leaves)
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]

def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    episodes = _tree_take(pipeline_states, eval_idx, axis=1)
    dones = dones.take(eval_idx, axis=1)
    return [
        state
        for state, done in zip(_unstack_tree(episodes), dones)
        if not (done)
    ]


@hydra.main(version_base=None, config_path="config", config_name="isac_mabrax")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # TODO: once I have more SAC variations include the matching code

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    # match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
    #     case (False, False):
    #         from ippo_ff_nps_mabrax import MultiActorCritic as NetworkArch
    #         from ippo_ff_nps_mabrax import make_evaluation as make_evaluation
    #     # make sure that all of these are called MultiActorCritic
    #     case (False, True):
    #         from ippo_ff_ps_mabrax import ActorCritic as NetworkArch
    #         from ippo_ff_ps_mabrax import make_evaluation as make_evaluation
    #     case (True, False):
    #         from ippo_rnn_nps_mabrax import MultiActorCriticRNN as NetworkArch
    #         from ippo_rnn_nps_mabrax import make_evaluation as make_evaluation
    #     case (True, True):
    #         from ippo_rnn_ps_mabrax import ActorCriticRNN as NetworkArch
    #         from ippo_rnn_ps_mabrax import make_evaluation as make_evaluation
    

    from isac_ff_nps_mabrax import MultiSACActor as NetworkArch
    from isac_ff_nps_mabrax import make_evaluation as make_evaluation

    rng = jax.random.PRNGKey(config["SEED"])
    rng, eval_rng = jax.random.split(rng)
    
    with jax.disable_jit(config["DISABLE_JIT"]):

        all_train_states = unflatten_dict(safetensors.flax.load_file(config["eval"]["path"]), sep='/')

        eval_env, run_eval = make_evaluation(config)
        eval_jit = jax.jit(run_eval, 
                        static_argnames=["log_env_state"],
                        )
        network = NetworkArch(config=config)
        # RENDER
        # Run episodes for render (saving env_state at each timestep)
        # I need to find a way to combine the parameters for the human and robot so I can load just the final parameters
        final_train_state = _tree_take(all_train_states, -1, axis=1)
        breakpoint()
        final_eval_network_state = EvalNetworkState(apply_fn=network.apply, params=final_train_state)
        final_eval = _tree_take(final_eval_network_state, 0, axis=0)
        # eval_final = eval_jit(eval_rng, _tree_take(final_eval_network_state, 0, axis=0), True)
        breakpoint()
        eval_final = eval_jit(eval_rng, final_eval, True)
        first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
        first_episode_rewards = eval_final.reward["__all__"] * (1-first_episode_done)
        first_episode_returns = first_episode_rewards.sum(axis=0)
        episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
        worst_idx = episode_argsort.take(0,axis=-1)
        best_idx = episode_argsort.take(-1, axis=-1)
        median_idx = episode_argsort.take(episode_argsort.shape[-1]//2, axis=-1)
        from brax.io import html
        worst_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, first_episode_done,
            time_idx=-1, eval_idx=worst_idx,
        )
        median_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, first_episode_done,
            time_idx=-1, eval_idx=median_idx,
        )
        best_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, first_episode_done,
            time_idx=-1, eval_idx=best_idx,
        )
        html.save(f"final_worst_r{int(first_episode_returns[worst_idx])}.html", eval_env.sys, worst_episode)
        html.save(f"final_median_r{int(first_episode_returns[median_idx])}.html", eval_env.sys, median_episode)
        html.save(f"final_best_r{int(first_episode_returns[best_idx])}.html", eval_env.sys, best_episode)


if __name__ == "__main__":
    main()