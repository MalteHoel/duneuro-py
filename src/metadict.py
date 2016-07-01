import itertools as it
import copy as cop
from collections import Iterable

def isiterable(obj):
    return not isinstance(obj, str) and isinstance(obj, Iterable)

class expand:
    def __init__(self, values):
        self.values = values

class transform:
    def __init__(self, keys, func = lambda x: x):
        self.keys = keys
        self.func = func

def flatten(md):
    newmd = dict()
    for k in md:
        if type(md[k]) is dict:
            flattened = flatten(md[k])
            for sk in flattened:
                newmd['.'.join([k,sk])] = flattened[sk]
        else:
            newmd[k] = md[k]
    return newmd

class NestedDict:
    def __init__(self):
        self.d = dict()
    def insert(self, key, value):
        l = key.split('.', 1)
        if len(l) == 1:
            self.d[key] = value
        else:
            if not l[0] in self.d:
                self.d[l[0]] = NestedDict()
            self.d[l[0]].insert(l[1], value)
    def get(self):
        result = dict()
        for k in self.d:
            if type(self.d[k]) is NestedDict:
                result[k] = self.d[k].get()
            else:
                result[k] = self.d[k]
        return result

def nest(md):
    flat = flatten(md)
    res = NestedDict()
    for k in flat:
        res.insert(k, flat[k])
    return res.get()

def apply_expansions(md):
    result = list()
    expansions = [k for k in md if type(md[k]) is expand]
    for v in it.product(*[range(len(md[k].values)) for k in expansions]):
        current = cop.deepcopy(md)
        for i in range(len(v)):
            current[expansions[i]] = md[expansions[i]].values[v[i]]
        result.append(current)
    return result

def resolve_transforms(md):
    result = cop.deepcopy(md)
    transforms = [k for k in result if type(result[k]) is transform]
    while len(transforms) > 0:
        for k in transforms:
            keys = result[k].keys
            if isiterable(keys):
                if sum(int(type(result[ck]) is transform) for ck in keys) == 0:
                    result[k] = result[k].func([result[ck] for ck in keys])
            else:
                if type(result[keys]) is not transform:
                    result[k] = result[k].func(result[keys])
        transforms = [k for k in result if type(result[k]) is transform]
    return result

def apply_transformations(md):
    result = flatten(md)
    result = apply_expansions(result)
    result = [resolve_transforms(d) for d in result]
    return [nest(x) for x in result]
