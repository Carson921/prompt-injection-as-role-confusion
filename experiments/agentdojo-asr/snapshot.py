"""Restore typed runtime snapshots without rerunning model initialization.

AgentDojo's Inbox/CloudDrive/Calendar/etc. model validators build mutable maps
from initial_* lists. Ordinary model_validate(snapshot) can discard changes
made by the actual tool trajectory. Restore typed field values from a trusted
snapshot without rerunning those model-level initialization validators.
"""
from copy import deepcopy
from functools import lru_cache
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter


@lru_cache(maxsize=None)
def _adapter(annotation):
    return TypeAdapter(annotation)


def _is_model(annotation):
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _restore(annotation, value):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Annotated:
        return _restore(args[0], value)
    if _is_model(annotation):
        assert annotation.__pydantic_post_init__ is None, 'Unexpected snapshot post-init hook'
        fields = annotation.model_fields
        assert all(name in value for name in fields), 'Incomplete runtime snapshot'
        return annotation.model_construct(_fields_set=set(fields),
            **{name: _restore(field.annotation, value[name]) for name, field in fields.items()})
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        structural = [a for a in args if (_is_model(a) and isinstance(value, dict))
                      or (get_origin(a) is dict and isinstance(value, dict))
                      or (get_origin(a) is list and isinstance(value, list))]
        if len(structural) == 1:
            return _restore(structural[0], value)
        # Use union validation only to select a branch; never accept a model
        # initialized from initial_* instead of the actual saved state.
        probe = _adapter(annotation).validate_python(deepcopy(value))
        if isinstance(probe, BaseModel):
            return _restore(type(probe), value)
        assert not structural, 'Ambiguous structural union in runtime snapshot'
        return probe
    if origin is list:
        return [_restore(args[0], item) for item in value]
    if origin is dict:
        return {_restore(args[0], key): _restore(args[1], item) for key, item in value.items()}
    if annotation is Any:
        return deepcopy(value)
    return _adapter(annotation).validate_python(value)


def restore_environment(model_type, snapshot):
    restored = _restore(model_type, snapshot)
    assert type(restored) is model_type
    # No silent dropped state, reinitialization or scalar/date conversion drift.
    assert restored.model_dump(mode='json') == snapshot, 'Runtime snapshot did not round-trip exactly'
    return restored
