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

import pathlib
from typing import IO, Any, Dict, List, Optional, Sequence, Tuple, Union, overload

from omegaconf.omegaconf import (
    _DEFAULT_MARKER_,
    BaseContainer,
    DictConfig,
    DictKeyType,
    ListConfig,
    OmegaConf,
)
from optuna.trial import BaseTrial

_TRIAL_KEY = "_optuna_trial"


class OmegaTuna(OmegaConf):
    @staticmethod
    def structured(
        obj: Any,
        parent: Optional[BaseContainer] = None,
        flags: Optional[Dict[str, bool]] = None,
        trial: Optional[BaseTrial] = None,
    ) -> Any:
        return OmegaTuna.create(obj, parent, flags, trial=trial)

    @staticmethod
    @overload
    def create(
        obj: str,
        parent: Optional[BaseContainer] = None,
        flags: Optional[Dict[str, bool]] = None,
        trial: Optional[BaseTrial] = None,
    ) -> Union[DictConfig, ListConfig]:
        ...

    @staticmethod
    @overload
    def create(
        obj: Union[List[Any], Tuple[Any, ...]],
        parent: Optional[BaseContainer] = None,
        flags: Optional[Dict[str, bool]] = None,
        trial: Optional[BaseTrial] = None,
    ) -> ListConfig:
        ...

    @staticmethod
    @overload
    def create(
        obj: DictConfig,
        parent: Optional[BaseContainer] = None,
        flags: Optional[Dict[str, bool]] = None,
        trial: Optional[BaseTrial] = None,
    ) -> DictConfig:
        ...

    @staticmethod
    @overload
    def create(
        obj: ListConfig,
        parent: Optional[BaseContainer] = None,
        flags: Optional[Dict[str, bool]] = None,
        trial: Optional[BaseTrial] = None,
    ) -> ListConfig:
        ...

    @staticmethod
    @overload
    def create(
        obj: Optional[Dict[Any, Any]] = None,
        parent: Optional[BaseContainer] = None,
        flags: Optional[Dict[str, bool]] = None,
        trial: Optional[BaseTrial] = None,
    ) -> DictConfig:
        ...

    @staticmethod
    def create(  # noqa F811 # type: ignore
        obj: Any = _DEFAULT_MARKER_,
        parent: Optional[BaseContainer] = None,
        flags: Optional[Dict[str, bool]] = None,
        trial: Optional[BaseTrial] = None,
    ):
        conf = OmegaTuna._create_impl(obj=obj, parent=parent, flags=flags)
        return _set_trial(conf, trial)

    @staticmethod
    def load(
        file_: Union[str, pathlib.Path, IO[Any]], trial: Optional[BaseTrial] = None
    ) -> Union[DictConfig, ListConfig]:
        conf = OmegaConf.load(file_)
        return _set_trial(conf, trial)

    @staticmethod
    def from_cli(
        args_list: Optional[List[str]] = None, trial: Optional[BaseTrial] = None
    ) -> DictConfig:
        conf = OmegaConf.from_cli(args_list)
        return _set_trial(conf, trial)

    @staticmethod
    def from_dotlist(
        dotlist: List[str], trial: Optional[BaseTrial] = None
    ) -> DictConfig:
        conf = OmegaConf.from_dotlist(dotlist)
        return _set_trial(conf, trial)

    @staticmethod
    def merge(
        *configs: Union[
            DictConfig,
            ListConfig,
            Dict[DictKeyType, Any],
            List[Any],
            Tuple[Any, ...],
            Any,
        ],
    ) -> Union[ListConfig, DictConfig]:
        try:
            trial = _get_trial_or_raise(
                [cfg for cfg in configs if isinstance(cfg, (DictConfig, ListConfig))]
            )
        except Exception:
            raise RuntimeError(
                "Trial instances bound to Config objects to merged must be identical"
            )

        merged = OmegaConf.merge(*configs)
        return _set_trial(merged, trial)


@overload
def _set_trial(conf: DictConfig, trial: Optional[BaseTrial]) -> DictConfig:
    ...


@overload
def _set_trial(conf: ListConfig, trial: Optional[BaseTrial]) -> ListConfig:
    ...


def _set_trial(
    conf: Union[DictConfig, ListConfig], trial: Optional[BaseTrial]
) -> Union[DictConfig, ListConfig]:
    try:
        org_trial = object.__getattribute__(conf, _TRIAL_KEY)
    except AttributeError:
        pass
    else:
        if org_trial:
            raise RuntimeError("cannot set a trial if one has already been set")

    if trial:
        object.__setattr__(conf, _TRIAL_KEY, trial)

    return conf


def _get_trial(conf: Union[DictConfig, ListConfig]) -> Optional[BaseTrial]:
    try:
        trial = object.__getattribute__(conf, _TRIAL_KEY)
    except AttributeError:
        return None

    return trial


def _get_trial_or_raise(
    confs: Sequence[Union[DictConfig, ListConfig]]
) -> Optional[BaseTrial]:
    trial = _get_trial(confs[0])
    for cfg in confs[1:]:
        tmp_trial = _get_trial(cfg)
        if trial is not None and tmp_trial is not None and trial is not tmp_trial:
            raise
        if trial is None and tmp_trial is not None:
            trial = tmp_trial

    return trial
