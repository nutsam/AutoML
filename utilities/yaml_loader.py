import os
import re

import yaml

TAG = "!ENV"
PATTERN = re.compile(r".*\$\{([^}^{]+)\}.*")
SPLITTER = "|"


def convert_to_boolean(value):
    if value.lower() in {"true", "1"}:
        return True
    elif value.lower() in {"false", "0"}:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")


MODE = {"int": int, "float": float, "str": str, "bool": convert_to_boolean}


def constructor_env_variables(loader, node):
    """
    Extracts the environment variable from the node's value

    This function is a custom constructor for PyYAML's SafeLoader that allows parsing
    environment variables in YAML files. It searches for environment variable placeholders
    in the node's value and replaces them with their corresponding values from the system's
    environment variables.

    :param loader: The YAML loader instance.
    :param node: The current node in the YAML document.
    :return: The parsed value that contains the resolved environment variable(s) or the original
             value if no environment variable is found.

    :raises ValueError: If the datatype specified for an environment variable is not one of 'int',
                        'float', 'bool', or 'str'.

    Example:
    is_str: !ENV ${IS_STR|str}
    is_true_o: !ENV ${IS_TRUE|bool}
    is_int: !ENV ${IS_INT|int}
    is_float: !ENV ${IS_FLOAT|float}

    """
    value = loader.construct_scalar(node)
    match = PATTERN.findall(value)  # to find all env variables in line
    if match:
        full_value = value
        for g in match:
            g, var_type = g.split(SPLITTER)
            if var_type not in MODE:
                raise ValueError(f"Datatype `{var_type}` not in 'int, float, bool, str'.")
            env_value = os.environ.get(g)
            if env_value is not None:
                full_value = MODE[var_type](env_value)
            else:
                # Environment variable not set, raise an error
                raise ValueError(f"Environment variable `{g}` not set.")
        return full_value
    return value


LOADER = yaml.SafeLoader
LOADER.add_implicit_resolver(TAG, PATTERN, None)
LOADER.add_constructor(TAG, constructor_env_variables)

__all__ = ["LOADER"]
