"""Load a downloaded PyTorch checkpoint without allowing arbitrary code: a restricted unpickler resolves only an
explicit allowlist of classes (found beforehand by listing the pickle's GLOBAL opcodes with pickletools) and
refuses everything else. Used for the Mammo-CLIP checkpoint, whose config is an omegaconf object inside a
collections.defaultdict, which torch's weights_only loader cannot rebuild.
"""
import pickle
import types

import torch

ALLOWED = {
    ("__builtin__", "dict"), ("__builtin__", "list"), ("__builtin__", "long"),
    ("builtins", "dict"), ("builtins", "list"), ("builtins", "int"),
    ("collections", "OrderedDict"), ("collections", "defaultdict"),
    ("omegaconf.base", "ContainerMetadata"), ("omegaconf.base", "Metadata"),
    ("omegaconf.dictconfig", "DictConfig"), ("omegaconf.listconfig", "ListConfig"),
    ("omegaconf.nodes", "AnyNode"), ("typing", "Any"),
    ("torch", "FloatStorage"), ("torch", "LongStorage"), ("torch._utils", "_rebuild_tensor_v2"),
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name} (not in allowlist)")


_restricted = types.SimpleNamespace(Unpickler=RestrictedUnpickler, load=pickle.load, __name__="restricted_pickle")


def load_restricted(path):
    return torch.load(path, map_location="cpu", pickle_module=_restricted, weights_only=False)
