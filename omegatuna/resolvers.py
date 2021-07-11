from functools import partial
from typing import Any

from omegaconf import Node

from .omegatuna import OmegaTuna

SUGGEST_METHODS = {
    "ot.categorical": "suggest_categorical",
    "ot.discrete_uniform": "suggest_discrete_uniform",
    "ot.float": "suggest_float",
    "ot.int": "suggest_int",
    "ot.loguniform": "suggest_loguniform",
    "ot.uniform": "suggest_uniform",
}


def _suggest(resolver_name: str, *args, _root_: Node, _node_: Node) -> Any:
    if len(args) == 2:
        name, kwargs = args
    elif len(args) == 1:
        name = str(_node_._key())
        kwargs = args[0]
    try:
        trial = object.__getattribute__(_root_, "_optuna_trial")
    except AttributeError:
        if "default" in kwargs:
            return kwargs["default"]
        else:
            raise RuntimeError(
                f"No default value is set to the parameter '{name}' "
                "although a trial object is not specified."
            )

    kwargs = dict(**kwargs)
    if "default" in kwargs:
        del kwargs["default"]

    return getattr(trial, SUGGEST_METHODS[resolver_name])(name, **kwargs)


def register_ot_resolvers() -> None:
    for key in SUGGEST_METHODS:
        OmegaTuna.register_new_resolver(
            key, partial(_suggest, key), replace=True, use_cache=False
        )


register_ot_resolvers()
