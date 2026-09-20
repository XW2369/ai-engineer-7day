"""Logging utilities. 工程代码里不许出现 print，全部走 logging。"""

import logging

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """返回一个配好 handler 和 formatter 的 logger。

    Args:
        name: logger 的名字，一般传调用方的 __name__。

    Returns:
        挂好 StreamHandler 的 Logger 对象。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger
