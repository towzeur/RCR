"""Model registry for the RCR release framework."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from argparse import Namespace
from typing import Callable

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
os.chdir(mammoth_path)

from models.utils.continual_model import ContinualModel
from utils import infer_args_from_signature, register_dynamic_module_fn
from utils.conf import warn_once


REGISTERED_MODELS = {}


def _cli_name(name: str) -> str:
    return name.replace("_", "-").lower()


def _module_name(name: str) -> str:
    return name.replace("-", "_").lower()


def register_model(name: str) -> Callable:
    """Register a continual-learning model class."""
    if hasattr(get_model_names, "names"):
        del get_model_names.names
    return register_dynamic_module_fn(_module_name(name), REGISTERED_MODELS, ContinualModel)


def get_all_models_legacy() -> list[str]:
    return [
        file[:-3]
        for file in os.listdir(os.path.dirname(__file__))
        if file.endswith(".py") and file != "__init__.py"
    ]


def get_all_models() -> dict[str, str]:
    """Return parser-facing model names mapped to Python module names."""
    return {_cli_name(model): model for model in get_all_models_legacy()}


def get_model_names(names_only: bool = False):
    """Return available model classes and parser-inferred arguments."""

    def _model_names():
        names = {}
        base_signature = inspect.signature(ContinualModel.__init__)

        for model, model_conf in REGISTERED_MODELS.items():
            names[_cli_name(model)] = {
                "class": model_conf["class"],
                "parsable_args": model_conf["parsable_args"],
            }

        for model in get_all_models_legacy():
            key = _cli_name(model)
            if key in names:
                continue
            try:
                mod = importlib.import_module("models." + model)
                for attr in mod.__dir__():
                    candidate = getattr(mod, attr)
                    if not inspect.isclass(candidate):
                        continue
                    try:
                        mro = inspect.getmro(candidate)
                    except TypeError:
                        continue
                    if ContinualModel not in mro[1:]:
                        continue
                    signature = inspect.signature(candidate.__init__)
                    parsable_args = infer_args_from_signature(
                        signature, excluded_signature=base_signature
                    )
                    names[_cli_name(candidate.NAME)] = {
                        "class": candidate,
                        "parsable_args": parsable_args,
                    }
            except Exception as exc:
                warn_once(f"Error in model {model}")
                warn_once(exc)
                names[key] = exc
        return names

    if not hasattr(get_model_names, "names"):
        setattr(get_model_names, "names", _model_names())
    names = getattr(get_model_names, "names")
    if names_only:
        return list(names.keys())
    return names


def get_model_class(args: Namespace, return_args: bool = False):
    names = get_model_names()
    key = _cli_name(args.model)
    assert key in names, f"Model {args.model} is not available."
    if isinstance(names[key], Exception):
        raise names[key]
    if return_args:
        return names[key]["class"], names[key]["parsable_args"]
    return names[key]["class"]


def get_model(args: Namespace, backbone, loss, transform, dataset=None):
    model_class, model_args = get_model_class(args, return_args=True)
    missing_args = [arg for arg in model_args.keys() if arg not in vars(args)]
    assert len(missing_args) == 0, "Missing arguments for the model: " + ", ".join(missing_args)
    return model_class(backbone, loss, args, transform, dataset=dataset)


for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and file != "__init__.py":
        try:
            importlib.import_module(f"models.{file[:-3]}")
        except Exception:
            pass
