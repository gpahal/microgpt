import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)",
    datefmt="%H:%M:%S",
)

_LOGGERS: list[logging.Logger] = []
_LOGGING_LEVEL = logging.INFO


def _new_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(_LOGGING_LEVEL)
    _LOGGERS.append(logger)
    return logger


def _set_logging_level(level: int) -> None:
    global _LOGGING_LEVEL
    _LOGGING_LEVEL = level
    for logger in _LOGGERS:
        logger.setLevel(level)
