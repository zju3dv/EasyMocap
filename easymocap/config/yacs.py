# Copyright (c) 2018-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""YACS -- Yet Another Configuration System is designed to be a simple
configuration management system for academic and industrial research
projects.
See README.md for usage and examples.
"""

import copy
import io
import logging
import os
from ast import literal_eval

import yaml


# Flag for py2 and py3 compatibility to use when separate code paths are necessary
# When _PY2 is False, we assume Python 3 is in use
_PY2 = False

# Filename extensions for loading configs from files
_YAML_EXTS = {"", ".yaml", ".yml"}
_PY_EXTS = {".py"}

# py2 and py3 compatibility for checking file object type
# We simply use this to infer py2 vs py3
try:
    _FILE_TYPES = (file, io.IOBase)
    _PY2 = True
except NameError:
    _FILE_TYPES = (io.IOBase,)

# CfgNodes can only contain a limited set of valid types
_VALID_TYPES = {tuple, list, str, int, float, bool}
# py2 allow for str and unicode
if _PY2:
    _VALID_TYPES = _VALID_TYPES.union({unicode})  # noqa: F821

# Utilities for importing modules from file paths
if _PY2:
    # imp is available in both py2 and py3 for now, but is deprecated in py3
    import imp
else:
    import importlib.util

logger = logging.getLogger(__name__)


class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    IMMUTABLE = "__immutable__"
    DEPRECATED_KEYS = "__deprecated_keys__"
    RENAMED_KEYS = "__renamed_keys__"

    def __init__(self, init_dict=None, key_list=None):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
                if '_parent_' in v.keys():
                    parent_ = CfgNode()
                    parent_.merge_from_file(v['_parent_'])
                    init_dict[k].pop('_parent_')
                    parent_.merge_from_other_cfg(init_dict[k])
                    init_dict[k] = parent_
                if '_parents_' in v.keys():
                    parent_ = CfgNode()
                    for parent in v['_parents_']:
                        parent_.merge_from_file(parent)
                    init_dict[k].pop('_parents_')
                    parent_.merge_from_other_cfg(init_dict[k])
                    init_dict[k] = parent_
                if '_const_' in v.keys() and v['_const_']:
                    init_dict[k].__dict__[CfgNode.IMMUTABLE] = True
                    init_dict[k].pop('_const_')
            elif type(v) is str and v.startswith('_file_/'):
                filename = v.replace('_file_/', '')
                init_dict[k] = CfgNode()
                init_dict[k].merge_from_file(filename)
            else:
                # Check for valid leaf type or nested CfgNode
                _assert_with_logging(
                    _valid_type(v, allow_cfg_node=True),
                    "Key {} with value {} is not a valid type; valid types: {}".format(
                        ".".join(key_list + [k]), type(v), _VALID_TYPES
                    ),
                )
        super(CfgNode, self).__init__(init_dict)
        # Manage if the CfgNode is frozen or not
        self.__dict__[CfgNode.IMMUTABLE] = False
        # Deprecated options
        # If an option is removed from the code and you don't want to break existing
        # yaml configs, you can add the full config key as a string to the set below.
        self.__dict__[CfgNode.DEPRECATED_KEYS] = set()
        # Renamed options
        # If you rename a config option, record the mapping from the old name to the new
        # name in the dictionary below. Optionally, if the type also changed, you can
        # make the value a tuple that specifies first the renamed key and then
        # instructions for how to edit the config file.
        self.__dict__[CfgNode.RENAMED_KEYS] = {
            # 'EXAMPLE.OLD.KEY': 'EXAMPLE.NEW.KEY',  # Dummy example to follow
            # 'EXAMPLE.OLD.KEY': (                   # A more complex example to follow
            #     'EXAMPLE.NEW.KEY',
            #     "Also convert to a tuple, e.g., 'foo' -> ('foo',) or "
            #     + "'foo:bar' -> ('foo', 'bar')"
            # ),
        }

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if self.is_frozen():
            raise AttributeError(
                "Attempted to set {} to {}, but CfgNode is immutable".format(
                    name, value
                )
            )

        _assert_with_logging(
            name not in self.__dict__,
            "Invalid attempt to modify internal CfgNode state: {}".format(name),
        )
        _assert_with_logging(
            _valid_type(value, allow_cfg_node=True),
            "Invalid type {} for key {}; valid types = {}".format(
                type(value), name, _VALID_TYPES
            ),
        )

        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        if len(self.keys()) == 0:
            return "{}"
        for k, v in self.items():
            seperator = "\n" if isinstance(v, CfgNode) and len(v.keys()) > 0 else " "
            if isinstance(v, float):
                attr_str = "{}:{}{:f}".format(str(k), seperator, v)
            else:
                attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 4)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())

    def dump(self):
        """Dump to a string."""
        self_as_dict = _to_dict(self)
        return yaml.safe_dump(self_as_dict)

    def merge_from_file(self, cfg_filename):
        """Load a yaml config file and merge it this CfgNode."""
        with open(cfg_filename, "r", encoding='utf8') as f:
            cfg = load_cfg(f)
        if 'parent' in cfg.keys():
            if cfg.parent != 'none':
                print('[Config] merge from parent file: {}'.format(cfg.parent))
                self.merge_from_file(cfg.parent)
        if 'parents' in cfg.keys():
            for parent in cfg['parents']:
                print('[Config] merge from parent file: {}'.format(parent))
                self.merge_from_file(parent)
            cfg.pop('parents')
        self.merge_from_other_cfg(cfg)

    def merge_from_other_cfg(self, cfg_other):
        """Merge `cfg_other` into this CfgNode."""
        _merge_a_into_b(cfg_other, self, self, [])

    def merge_from_list(self, cfg_list):
        """Merge config (keys, values) in a list (e.g., from command line) into
        this CfgNode. For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        _assert_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".format(
                cfg_list
            ),
        )
        root = self
        cfg_list_new = []
        alias = self.pop('_alias_', {})
        for i in range(len(cfg_list)//2):
            if cfg_list[2*i] in alias.keys():
                for name in alias[cfg_list[2*i]]:
                    cfg_list_new.append(name)
                    cfg_list_new.append(cfg_list[2*i+1])
            else:
                cfg_list_new.append(cfg_list[2*i])
                cfg_list_new.append(cfg_list[2*i+1])
        cfg_list = cfg_list_new
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            if root.key_is_deprecated(full_key):
                continue
            if root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            key_list = full_key.split(".")
            d = self
            for subkey in key_list[:-1]:
                _assert_with_logging(
                    subkey in d, "Non-existent key: {}".format(full_key)
                )
                d = d[subkey]
            subkey = key_list[-1]
            value = _decode_cfg_value(v)
            if subkey not in d.keys():
                logger.warning("Key is not in the template: {}".format(full_key))
                d[subkey] = value
            else:
                value = _decode_cfg_value(v)
                value = _check_and_coerce_cfg_value_type(value, d[subkey], subkey, full_key)
            d[subkey] = value

    def freeze(self):
        """Make this CfgNode and all of its children immutable."""
        self._immutable(True)

    def defrost(self):
        """Make this CfgNode and all of its children mutable."""
        self._immutable(False)

    def is_frozen(self):
        """Return mutability."""
        return self.__dict__[CfgNode.IMMUTABLE]

    def _immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested CfgNodes.
        """
        self.__dict__[CfgNode.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, CfgNode):
                v._immutable(is_immutable)
        for v in self.values():
            if isinstance(v, CfgNode):
                v._immutable(is_immutable)

    def clone(self):
        """Recursively copy this CfgNode."""
        return copy.deepcopy(self)

    def register_deprecated_key(self, key):
        """Register key (e.g. `FOO.BAR`) a deprecated option. When merging deprecated
        keys a warning is generated and the key is ignored.
        """
        _assert_with_logging(
            key not in self.__dict__[CfgNode.DEPRECATED_KEYS],
            "key {} is already registered as a deprecated key".format(key),
        )
        self.__dict__[CfgNode.DEPRECATED_KEYS].add(key)

    def register_renamed_key(self, old_name, new_name, message=None):
        """Register a key as having been renamed from `old_name` to `new_name`.
        When merging a renamed key, an exception is thrown alerting to user to
        the fact that the key has been renamed.
        """
        _assert_with_logging(
            old_name not in self.__dict__[CfgNode.RENAMED_KEYS],
            "key {} is already registered as a renamed cfg key".format(old_name),
        )
        value = new_name
        if message:
            value = (new_name, message)
        self.__dict__[CfgNode.RENAMED_KEYS][old_name] = value

    def key_is_deprecated(self, full_key):
        """Test if a key is deprecated."""
        if full_key in self.__dict__[CfgNode.DEPRECATED_KEYS]:
            logger.warning("Deprecated config key (ignoring): {}".format(full_key))
            return True
        return False

    def key_is_renamed(self, full_key):
        """Test if a key is renamed."""
        return full_key in self.__dict__[CfgNode.RENAMED_KEYS]

    def raise_key_rename_error(self, full_key):
        new_key = self.__dict__[CfgNode.RENAMED_KEYS][full_key]
        if isinstance(new_key, tuple):
            msg = " Note: " + new_key[1]
            new_key = new_key[0]
        else:
            msg = ""
        raise KeyError(
            "Key {} was renamed to {}; please update your config.{}".format(
                full_key, new_key, msg
            )
        )


def load_cfg(cfg_file_obj_or_str):
    """Load a cfg. Supports loading from:
        - A file object backed by a YAML file
        - A file object backed by a Python source file that exports an attribute
          "cfg" that is either a dict or a CfgNode
        - A string that can be parsed as valid YAML
    """
    _assert_with_logging(
        isinstance(cfg_file_obj_or_str, _FILE_TYPES + (str,)),
        "Expected first argument to be of type {} or {}, but it was {}".format(
            _FILE_TYPES, str, type(cfg_file_obj_or_str)
        ),
    )
    if isinstance(cfg_file_obj_or_str, str):
        return _load_cfg_from_yaml_str(cfg_file_obj_or_str)
    elif isinstance(cfg_file_obj_or_str, _FILE_TYPES):
        return _load_cfg_from_file(cfg_file_obj_or_str)
    else:
        raise NotImplementedError("Impossible to reach here (unless there's a bug)")


def _load_cfg_from_file(file_obj):
    """Load a config from a YAML file or a Python source file."""
    _, file_extension = os.path.splitext(file_obj.name)
    if file_extension in _YAML_EXTS:
        return _load_cfg_from_yaml_str(file_obj.read())
    elif file_extension in _PY_EXTS:
        return _load_cfg_py_source(file_obj.name)
    else:
        raise Exception(
            "Attempt to load from an unsupported file type {}; "
            "only {} are supported".format(file_obj, _YAML_EXTS.union(_PY_EXTS))
        )


def _load_cfg_from_yaml_str(str_obj):
    """Load a config from a YAML string encoding."""
    cfg_as_dict = yaml.safe_load(str_obj)
    return CfgNode(cfg_as_dict)


def _load_cfg_py_source(filename):
    """Load a config from a Python source file."""
    module = _load_module_from_file("yacs.config.override", filename)
    _assert_with_logging(
        hasattr(module, "cfg"),
        "Python module from file {} must have 'cfg' attr".format(filename),
    )
    VALID_ATTR_TYPES = {dict, CfgNode}
    _assert_with_logging(
        type(module.cfg) in VALID_ATTR_TYPES,
        "Imported module 'cfg' attr must be in {} but is {} instead".format(
            VALID_ATTR_TYPES, type(module.cfg)
        ),
    )
    if type(module.cfg) is dict:
        return CfgNode(module.cfg)
    else:
        return module.cfg


def _to_dict(cfg_node):
    """Recursively convert all CfgNode objects to dict objects."""

    def convert_to_dict(cfg_node, key_list):
        if not isinstance(cfg_node, CfgNode):
            _assert_with_logging(
                _valid_type(cfg_node),
                "Key {} with value {} is not a valid type; valid types: {}".format(
                    ".".join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
            return cfg_node
        else:
            cfg_dict = dict(cfg_node)
            for k, v in cfg_dict.items():
                cfg_dict[k] = convert_to_dict(v, key_list + [k])
            return cfg_dict

    return convert_to_dict(cfg_node, [])


def _valid_type(value, allow_cfg_node=False):
    return (type(value) in _VALID_TYPES) or (allow_cfg_node and type(value) == CfgNode)


def _merge_a_into_b(a, b, root, key_list):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    _assert_with_logging(
        isinstance(a, CfgNode),
        "`a` (cur type {}) must be an instance of {}".format(type(a), CfgNode),
    )
    _assert_with_logging(
        isinstance(b, CfgNode),
        "`b` (cur type {}) must be an instance of {}".format(type(b), CfgNode),
    )
    if '_no_merge_' in a.keys() and a['_no_merge_']:
        b.clear()
        # TODO:这里好像b好像有时候是a的拷贝，有时候不是
        if '_no_merge_' in a.keys():
            a.pop('_no_merge_')

    for k, v_ in a.items():
        full_key = ".".join(key_list + [k])
        # a must specify keys that are in b
        if k not in b:
            if root.key_is_deprecated(full_key):
                continue
            elif root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            else:
                v = copy.deepcopy(v_)
                v = _decode_cfg_value(v)
                b.update({k: v})
        else:
            v = copy.deepcopy(v_)
            v = _decode_cfg_value(v)
            v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, CfgNode):
            try:
                _merge_a_into_b(v, b[k], root, key_list + [k])
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to CfgNode objects
    if isinstance(v, dict):
        return CfgNode(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple), (int, float), (float, int)]
    # For py2: allow converting from str (bytes) to a unicode string
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def _assert_with_logging(cond, msg):
    if not cond:
        logger.debug(msg)
    assert cond, msg


def _load_module_from_file(name, filename):
    if _PY2:
        module = imp.load_source(name, filename)
    else:
        spec = importlib.util.spec_from_file_location(name, filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module