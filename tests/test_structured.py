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

from dataclasses import dataclass
from typing import Any

import pytest
from optuna.trial import BaseTrial, FixedTrial

from omegatuna import II, SI, OmegaTuna


@pytest.fixture(params=[(3, 0.3, None), (2, 0.2, True)])
def trial(request) -> BaseTrial:
    p_int, p_float, p_cat = request.param
    return FixedTrial(
        {
            "param_int": p_int,
            "param_float": p_float,
            "param_cat": p_cat,
        }
    )


@dataclass
class StructuredConf:
    param_int: int = "${ot.int: param_int, {low: -10, high: 10}}"  # type: ignore
    param_float: float = "${ot.float: param_float, {low: -10.0, high: 10.0}}"  # type: ignore # noqa
    param_cat: Any = (
        "${ot.categorical: param_cat, {choices: [null, true, 1, 0.3, test]}}"
    )


def test_structured(trial: BaseTrial):
    conf = OmegaTuna.structured(StructuredConf, trial=trial)
    assert conf.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert conf.param_float == trial.suggest_float("param_float", low=-10.0, high=10.0)
    assert conf.param_cat == trial.suggest_categorical(
        "param_cat", choices=[None, True, 1, 0.3, "test"]
    )


@dataclass
class StructuredConfSI:
    param_int: int = SI("${ot.int: param_int, {low: -10, high: 10}}")
    param_float: float = SI("${ot.float: param_float, {low: -10.0, high: 10.0}}")
    param_cat: Any = SI(
        "${ot.categorical: param_cat, {choices: [null, true, 1, 0.3, test]}}"
    )


def test_structured_SI(trial: BaseTrial):
    conf = OmegaTuna.structured(StructuredConfSI, trial=trial)
    assert conf.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert conf.param_float == trial.suggest_float("param_float", low=-10.0, high=10.0)
    assert conf.param_cat == trial.suggest_categorical(
        "param_cat", choices=[None, True, 1, 0.3, "test"]
    )


@dataclass
class StructuredConfII:
    param_int: int = II("ot.int: param_int, {low: -10, high: 10}")
    param_float: float = II("ot.float: param_float, {low: -10.0, high: 10.0}")
    param_cat: Any = II(
        "ot.categorical: param_cat, {choices: [null, true, 1, 0.3, test]}"
    )


def test_structured_II(trial: BaseTrial):
    conf = OmegaTuna.structured(StructuredConfII, trial=trial)
    assert conf.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert conf.param_float == trial.suggest_float("param_float", low=-10.0, high=10.0)
    assert conf.param_cat == trial.suggest_categorical(
        "param_cat", choices=[None, True, 1, 0.3, "test"]
    )
