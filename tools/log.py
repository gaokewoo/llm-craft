import logging
import sys


def get_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """创建并返回一个格式化日志记录器。

    Args:
        name: 日志名称，通常传 __name__。为 None 时取调用者模块名。
        level: 日志级别，默认 INFO。

    Returns:
        配置好的 Logger 实例。
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-5s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    return logger


# 默认实例，方便直接 import 使用
log = get_logger("llm-craft")
