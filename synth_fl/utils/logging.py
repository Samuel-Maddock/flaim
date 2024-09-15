import sys

from loguru import logger

DEFAULT_LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
SYNTHFL_FORMAT = "<b><c>synth_fl:</c></b> <green>{time:DD-MM-YYYY HH:mm:ss}</green> | {name}:<level>{level}</level> | <level>{message}</level>"


def init_logger():
    logger.remove(logger._current_handler_id)
    logger._current_handler_id = logger.add(
        sys.stderr,
        format=SYNTHFL_FORMAT,
        colorize=True,
    )


def newline():
    logger.remove(logger._current_handler_id)
    logger._current_handler_id = logger.add(
        sys.stderr,
        format="",
        colorize=True,
    )
    logger.info("")
    init_logger()


logger.newline = newline
logger._current_handler_id = 0
init_logger()
