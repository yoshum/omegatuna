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


def test_int(trial: BaseTrial) -> None:
    d = ["param=${ot.int:param_int, {low:-10, high:10}}"]
    conf = OmegaTuna.from_dotlist(d, trial=trial)

    assert isinstance(conf.param, int)
    assert conf.param == trial.suggest_int("param_int", -10, 10)


def test_int_log(trial: BaseTrial) -> None:
    d = ["param=${ot.int:param_int, {low:1, high:10, log:true}}"]
    conf = OmegaTuna.from_dotlist(d, trial=trial)

    assert isinstance(conf.param, int)
    assert conf.param == trial.suggest_int("param_int", 1, 10, log=True)


def test_int_default(trial: BaseTrial) -> None:
    d = ["param='${ot.int: param_int, {low: -10, high: 10, default: -1}}'"]
    conf = OmegaTuna.from_dotlist(d, trial=trial)

    assert isinstance(conf.param, int)
    assert conf.param == trial.suggest_int("param_int", -10, 10)

    conf = OmegaTuna.from_dotlist(d)

    assert isinstance(conf.param, int)
    assert conf.param == -1


def test_float(trial: BaseTrial) -> None:
    d = ["param='${ot.float: param_float, {low: -10.0, high: 10.0}}'"]
    conf = OmegaTuna.from_dotlist(d, trial=trial)

    assert isinstance(conf.param, float)
    assert conf.param == trial.suggest_float("param_float", -10, 10)


def test_uniform(trial: BaseTrial) -> None:
    d = ["param='${ot.uniform: param_float, {low: -10.0, high: 10.0}}'"]
    conf = OmegaTuna.from_dotlist(d, trial=trial)

    assert isinstance(conf.param, float)
    assert conf.param == trial.suggest_uniform("param_float", -10, 10)


def test_loguniform(trial: BaseTrial) -> None:
    d = ["param='${ot.loguniform: param_float, {low: 0.01, high: 10.0}}'"]
    conf = OmegaTuna.from_dotlist(d, trial=trial)

    assert isinstance(conf.param, float)
    assert conf.param == trial.suggest_loguniform("param_float", 0.01, 10)


def test_float_log(trial: BaseTrial) -> None:
    d = ["param='${ot.float: param_float, {low: 0.01, high: 10.0, log: true}}'"]
    conf = OmegaTuna.from_dotlist(d, trial=trial)

    assert isinstance(conf.param, float)
    assert conf.param == trial.suggest_float("param_float", 0.01, 10, log=True)


def test_float_default(trial: BaseTrial) -> None:
    d = ["param='${ot.float: param_float, {low: -10.0, high: 10.0, default: -0.1}}'"]
    conf = OmegaTuna.from_dotlist(d, trial=trial)

    assert isinstance(conf.param, float)
    assert conf.param == trial.suggest_float("param_float", -10, 10)

    conf = OmegaTuna.from_dotlist(d)

    assert isinstance(conf.param, float)
    assert conf.param == -0.1


def test_categorical(trial: BaseTrial) -> None:
    d = ["param='${ot.categorical: param_cat, {choices: [null, true, 1, 0.3, test]}}'"]
    conf = OmegaTuna.from_dotlist(d, trial=trial)

    assert conf.param == trial.suggest_categorical(
        "param_cat", [None, True, 1, 0.3, "test"]
    )


def test_categorical_default(trial: BaseTrial) -> None:
    d = [
        "param='${ot.categorical:param_cat, {choices: [null, true, 1, 0.3, test], default: 1}}'"  # noqa
    ]
    conf = OmegaTuna.from_dotlist(d, trial=trial)

    assert conf.param == trial.suggest_categorical(
        "param_cat", [None, True, 1, 0.3, "test"]
    )

    conf = OmegaTuna.from_dotlist(d)

    assert conf.param == 1


def test_resolve(trial: BaseTrial) -> None:
    d = ["param='${ot.int: param_int, {low: -10, high: 10, default: -1}}'"]
    conf1 = OmegaTuna.from_dotlist(d, trial=trial)
    OmegaTuna.resolve(conf1)

    conf2 = OmegaTuna.from_dotlist(d)

    assert conf1.param == trial.suggest_int("param_int", -10, 10)
    assert conf2.param == -1

    d = ["param='${ot.int: param_int, {low: -10, high: 10, default: -1}}'"]
    conf1 = OmegaTuna.from_dotlist(d, trial=trial)
    conf2 = OmegaTuna.from_dotlist(d)

    assert conf1.param == trial.suggest_int("param_int", -10, 10)
    assert conf2.param == -1
