#  Copyright 2021 Shuhei Yoshida
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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
