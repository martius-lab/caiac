import collections
from copy import deepcopy
import json


class AttributeDict(dict):
    """ A dict which allows attribute access to its keys."""

    def __getattr__(self, *args, **kwargs):
        try:
            return self.__getitem__(*args, **kwargs)
        except KeyError as e:
            raise AttributeError(e)

    def __deepcopy__(self, memo):
        """ In order to support deepcopy"""
        return self.__class__(
            [(deepcopy(k, memo=memo), deepcopy(v, memo=memo)) for k, v in self.items()])

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, item):
        self.__delitem__(item)

    def __repr__(self):
        return json.dumps(self, indent=4, sort_keys=True)


class ImmutableAttributeDict(AttributeDict):
    """ A dict which allows attribute access to its keys. Forced immutable."""

    def __delattr__(self, item):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __delitem__(self, item):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __setattr__(self, key, value):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __setitem__(self, key, value):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __reduce__(self):
        # For overwriting dict pickling
        # https://stackoverflow.com/questions/21144845/how-can-i-unpickle-a-subclass-of-dict-that-validates-with-setitem-in-pytho
        return (AttributeDict, (self.__getstate__(),))

    def _mutable_copy(self):
        return recursive_objectify(self, make_immutable=False)

    def __getstate__(self):
        return self._mutable_copy()


def recursive_objectify(nested_dict, make_immutable=True):
    "Turns a nested_dict into a nested AttributeDict"
    result = deepcopy(nested_dict)
    for k, v in result.items():
        if isinstance(v, collections.Mapping):
            result = dict(result)
            result[k] = recursive_objectify(v, make_immutable)
    if make_immutable:
        returned_result = ImmutableAttributeDict(result)
    else:
        returned_result = AttributeDict(result)
    return returned_result