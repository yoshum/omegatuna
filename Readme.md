# OmegaTuna: A thin wrapper of OmegaConf to integrate Optuna

_OmegaTuna_ facilitates management of experiments by offering a consice way of defining search spaces for [Optuna](https://optuna.org/). OmegaTuna allows us to specify hyperparameters and their search spaces _in one place_ in, e.g., a YAML file (or anywhere [OmegaConf](https://github.com/omry/omegaconf) can load from). [Hydra's Optuna Sweeper plugin](https://hydra.cc/docs/next/plugins/optuna_sweeper/) offers similar functionality, with which, however, hyperparameters and their search spaces are separately specified.

This library builds upon [OmegaConf](https://github.com/omry/omegaconf) and its interpolation system. Please refer to [OmegaConf documentation](https://omegaconf.readthedocs.io/en/latest/usage.html) for its usage.

## Features

- Concise notation of hyperparameter search spaces
- Works as a drop-in replacement for OmegaConf

## Basic usage

- Use `omegatuna.OmegaTuna` in place of `omegaconf.OmegaConf`
- You can pass an optional argument `trial` to the configuration-creation methods: `create`, `from_cli`, `from_dotlist`, `load`, and `structured`
- The following resolvers, each of which corresponds to one of `optuna.Trial.suggest_*`,
  are defined and can be used to specify a search space
  - `ot.categorical`
  - `ot.discrete_uniform`
  - `ot.float`
  - `ot.int`
  - `ot.loguniform`
  - `ot.uniform`
- The resolvers take two arguments, the name of a parameter (optional) and a dict of keyword arguments passed to `optuna.Trial.suggest_*` methods
  - If a parameter name is omitted, the configuration key of that parameter is used by default
  - The second argument looks like `{low:-10, high:10}`
  - You can also give a default value as `{low:-10, high:10, default:1}`
  - See below for more examples

## Note

### Restriction on the merge operation

Configuration objects to which different instances of `optuna.Trial` are bound cannot be
merged into a single configuration. In other words, multiple configurations can be
merged only if some or all of them are created by passing an identical `optuna.Trial`
object and the others, if any, are created without specifying any `optuna.Trial` object.
This ensures that one configuration object holds parameters for a single trial.

### To avoid a parse error

Interpolation clauses need to be carefully formatted to avoid a parse error.
Specifically, you have either

- to quote each interpolation block or
- to remove any whitespace after `:`.

### Lazy evaluation of interpolations

Because this library relies on the interpolation mechanism of OmegaConf, the resolvers
are called lazily, i.e., when the configuration node is accessed. This does not cause
any complication as long as you use the configuration object within an objective
function to which an trial object is passed. However, to record parameters for later
reference, it is recommended that the interpolations are resolved by using
`OmegaTuna.resolve` (alias of `OmegaConf.resolve`) before serializing the configuration
object.

## Examples

### From a dict object

```python
from omegatuna import OmegaTuna
from optuna.trial import BaseTrial

def objective(trial: BaseTrial) -> float:
  conf = OmegaTuna.create(
    {
      "p_i": "${ot.int: param_int, {low: -10, high: 10, default: -1}}",
      # parameter name is omitted below
      "p_c": "${ot.categorical: {choices: [null, true, 1, 0.3, test], default: 1}}"
    },
    trial=trial
  )

  assert conf.p_i == trial.suggest_int("param_int", low=-10, high=10)
  assert conf.p_c == trial.suggest_categorical(
    "p_c", choices=[None, True, 1, 0.3, "test"]
  )

  ret = calculate(conf)
  return ret
```

### From a dict object without a trial argument

When `trial` is not passed, default values are used to fill the interpolations.

```python
from omegatuna import OmegaTuna
from optuna.trial import BaseTrial

conf = OmegaTuna.create(
  {
    "p_i": "${ot.int: {low: -10, high: 10, default: -1}}",
    "p_c": "${ot.categorical: {choices: [null, true, 1, 0.3, test], default: 1}}"
  }
)

assert conf.p_i == -1
assert conf.p_c == 1
```

### From a YAML string

```python
from omegatuna import OmegaTuna
from optuna.trial import BaseTrial


# Remember to remove whitespace after `:` or to quote the interpolations
yaml_string = """
param_int: ${ot.int:{low:-10, high:10}}
param_float: '${ot.float: {low: -10.0, high: 10.0}}'
param_cat: '${ot.categorical: {choices: [null, true, 1, 0.3, test]}}'
"""

def objective(trial: BaseTrial):
    conf = OmegaTuna.create(yaml_string, trial=trial)
    assert conf.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert conf.param_float == trial.suggest_float("param_float", low=-10.0, high=10.0)
    assert conf.param_cat == trial.suggest_categorical(
        "param_cat", choices=[None, True, 1, 0.3, "test"]
    )
```

You can instead use `OmegaTuna.load` to create a configuration object from a YAML file.

### From a dot-list

```python
from omegatuna import OmegaTuna
from optuna.trial import BaseTrial

def objective(trial: BaseTrial) -> float:
  conf = OmegaTuna.from_dotlist(
    # Quoting the interpolation block
    ["param='${ot.int: {low: -10, high: 10, default: -1}}'"],
    # ["param=${ot.int:{low:-10, high:10, default:-1}}"],  # This also works
    trial=trial
  )
  assert conf.param == trial.suggest_int("param", low=-10, high=10)
  ret = calculate(conf)
  return ret
```

### From structured config

```python
from dataclasses import dataclass
from omegatuna import II, SI
from optuna.trial import BaseTrial


@dataclass
class StructuredConf:
    # This works, but type checkers will complain
    param_int: int = "${ot.int: {low: -10, high: 10}}"

    # Use `omegatuna.SI` or `omegatuna.II` to suppress type errors
    param_float: float = SI("${ot.float: {low: -10.0, high: 10.0}}")
    param_cat: Any = II("ot.categorical: {choices: [null, true, 1, 0.3, test]}")


def objective(trial: BaseTrial):
    conf = OmegaTuna.structured(StructuredConf, trial=trial)
    assert conf.param_int == trial.suggest_int("param_int", low=-10, high=10)
    assert conf.param_float == trial.suggest_float("param_float", low=-10.0, high=10.0)
    assert conf.param_cat == trial.suggest_categorical(
        "param_cat", choices=[None, True, 1, 0.3, "test"]
    )
```
