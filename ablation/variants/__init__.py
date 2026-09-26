# ablation/variants/__init__.py
from .variant_flat_mappo import FlatMAPPOAgent, FlatActor, FlatCritic
from .variant_mlp_encoder import MLPEncoder
from .variant_homo_gnn import HomoGNNEncoder
from .variant_no_attn import NoAttnEncoder, MeanAggregationLayer

# The released repository does not contain these legacy baseline modules.
# Resolve them only when explicitly requested, so available ablations remain
# importable. Missing implementations still raise an error rather than silently
# substituting another algorithm.
def __getattr__(name):
    legacy = {"DDQNAgent": "variant_ddqn", "EDQNAgent": "variant_edqn",
              "SACAgent": "variant_sac", "TD3Agent": "variant_td3"}
    if name not in legacy:
        raise AttributeError(name)
    from importlib import import_module
    return getattr(import_module(f"{__name__}.{legacy[name]}"), name)
