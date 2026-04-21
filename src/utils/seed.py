"""Deterministic-execution helpers for reproducible scientific results."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int = 42) -> None:
    """Seed all sources of randomness used in the pipeline.

    This function seeds Python's ``random`` module, NumPy, TensorFlow and
    configures the hash-seed environment variable. It should be called at
    the very top of every script before any model is instantiated.

    Parameters
    ----------
    seed : int
        The integer seed used for every RNG.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np  # imported lazily so docs build without it

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
