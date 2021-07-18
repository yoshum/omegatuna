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


@pytest.fixture(params=[(3, 0.3, None), (2, 0.2, True)])
def trial2(request) -> BaseTrial:
    p_int, p_float, p_cat = request.param
    return FixedTrial(
        {
            "param2_int": p_int,
            "param2_float": p_float,
            "param2_cat": p_cat,
        }
    )


def test_raise(trial: BaseTrial, trial2: BaseTrial) -> None:
    d = {"param": "${ot.int: param_int, {low: -10, high: 10}}"}
    d2 = {"param2": "${ot.int: param2_int, {low: -10, high: 10}}"}
    conf = OmegaTuna.create(d, trial=trial)
    conf2 = OmegaTuna.create(d2, trial=trial2)

    with pytest.raises(RuntimeError):
        OmegaTuna.merge(conf, conf2)


def test_merge(trial: BaseTrial) -> None:
    d = {"param_int": "${ot.int: param_int, {low: -10, high: 10}}"}
    d2 = {"param_float": "${ot.float: param_float, {low: -10.0, high: 10.0}}"}
    conf = OmegaTuna.create(d, trial=trial)
    conf2 = OmegaTuna.create(d2, trial=trial)

    merged = OmegaTuna.merge(conf, conf2)

    assert merged.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert merged.param_float == trial.suggest_float(
        "param_float", low=-10.0, high=10.0
    )


def test_merge_2(trial: BaseTrial) -> None:
    d = {"param_int": "${ot.int: param_int, {low: -10, high: 10}}"}
    d2 = {"other": -1}
    conf = OmegaTuna.create(d, trial=trial)
    conf2 = OmegaTuna.create(d2)

    merged = OmegaTuna.merge(conf, conf2)

    assert merged.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert merged.other == -1
