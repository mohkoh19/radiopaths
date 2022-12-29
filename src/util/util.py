import argparse
from distutils.util import strtobool


def semi_flag(v):
    """ This enables using boolean switches more flexibly in VS Code launch.json settings.

    Args:
        v (str): Value to be transformed to bool.

    Raises:
        argparse.ArgumentTypeError: Value can't be casted to bool.

    Returns:
        bool: Value casted to bool.
    """
    try:
        return bool(strtobool(v))
    except:
        raise argparse.ArgumentTypeError("Boolean value expected.")
