"""Light-weight logging helper."""

from __future__ import annotations

import logging


def get_logger(name: str = "geo-btc", level: int = logging.INFO) -> logging.Logger:
    """Return a module logger with a uniform format.

    Parameters
    ----------
    name : str
        Logger name (usually ``__name__`` of the caller).
    level : int
        Logging level. Default :data:`logging.INFO`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
