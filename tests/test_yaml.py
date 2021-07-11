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

import pytest
from optuna.trial import BaseTrial, FixedTrial

from omegatuna import OmegaTuna


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


yaml_string = """
param_int: ${ot.int:param_int, {low:-10, high:10}}
param_float: '${ot.float: param_float, {low: -10.0, high: 10.0}}'
param_cat: '${ot.categorical: param_cat, {choices: [null, true, 1, 0.3, test]}}'
"""

yaml_string_default_name = """
param_int: ${ot.int:{low:-10, high:10}}
param_float: '${ot.float:  {low: -10.0, high: 10.0}}'
param_cat: '${ot.categorical:  {choices: [null, true, 1, 0.3, test]}}'
"""

yaml_string_default = """
param_int: ${ot.int:{low:-10, high:10, default:-1}}
param_float: '${ot.float:  {low: -10.0, high: 10.0, default:-0.1}}'
param_cat: '${ot.categorical:  {choices: [null, true, 1, 0.3, test], default:null}}'
"""


@pytest.fixture
def yaml_file(tmpdir):
    tmpfile = tmpdir.join("config.yml")
    with open(tmpfile, "w") as f:
        f.write(yaml_string)

    yield str(tmpfile)

    tmpfile.remove()


@pytest.fixture
def yaml_file_default_name(tmpdir):
    tmpfile = tmpdir.join("config.yml")
    with open(tmpfile, "w") as f:
        f.write(yaml_string_default_name)

    yield str(tmpfile)

    tmpfile.remove()


@pytest.fixture
def yaml_file_default(tmpdir):
    tmpfile = tmpdir.join("config.yml")
    with open(tmpfile, "w") as f:
        f.write(yaml_string_default)

    yield str(tmpfile)

    tmpfile.remove()


def test_load_yaml_file(trial: BaseTrial, yaml_file: str):
    conf = OmegaTuna.load(yaml_file, trial=trial)
    assert conf.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert conf.param_float == trial.suggest_float("param_float", low=-10.0, high=10.0)
    assert conf.param_cat == trial.suggest_categorical(
        "param_cat", choices=[None, True, 1, 0.3, "test"]
    )


def test_load_yaml_file_default_name(trial: BaseTrial, yaml_file_default_name: str):
    conf = OmegaTuna.load(yaml_file_default_name, trial=trial)
    assert conf.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert conf.param_float == trial.suggest_float("param_float", low=-10.0, high=10.0)
    assert conf.param_cat == trial.suggest_categorical(
        "param_cat", choices=[None, True, 1, 0.3, "test"]
    )


def test_load_yaml_file_default(trial: BaseTrial, yaml_file_default: str):
    conf = OmegaTuna.load(yaml_file_default)
    assert conf.param_int == -1
    assert conf.param_float == -0.1
    assert conf.param_cat is None

    conf = OmegaTuna.load(yaml_file_default, trial=trial)
    assert conf.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert conf.param_float == trial.suggest_float("param_float", low=-10.0, high=10.0)
    assert conf.param_cat == trial.suggest_categorical(
        "param_cat", choices=[None, True, 1, 0.3, "test"]
    )


def test_load_yaml_string(trial: BaseTrial):
    conf = OmegaTuna.create(yaml_string, trial=trial)
    assert conf.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert conf.param_float == trial.suggest_float("param_float", low=-10.0, high=10.0)
    assert conf.param_cat == trial.suggest_categorical(
        "param_cat", choices=[None, True, 1, 0.3, "test"]
    )


def test_load_yaml_string_default_name(trial: BaseTrial):
    conf = OmegaTuna.create(yaml_string_default_name, trial=trial)
    assert conf.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert conf.param_float == trial.suggest_float("param_float", low=-10.0, high=10.0)
    assert conf.param_cat == trial.suggest_categorical(
        "param_cat", choices=[None, True, 1, 0.3, "test"]
    )


def test_load_yaml_string_default(trial: BaseTrial):
    conf = OmegaTuna.create(yaml_string_default)
    assert conf.param_int == -1
    assert conf.param_float == -0.1
    assert conf.param_cat is None

    conf = OmegaTuna.create(yaml_string_default, trial=trial)
    assert conf.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert conf.param_float == trial.suggest_float("param_float", low=-10.0, high=10.0)
    assert conf.param_cat == trial.suggest_categorical(
        "param_cat", choices=[None, True, 1, 0.3, "test"]
    )
