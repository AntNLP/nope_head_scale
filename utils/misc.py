from functools import cache
from logging import Logger


# useful to prevent some functions from being called
# just replace the function with this one like a monkey patch
def forbidden_func(*args, **kwargs):
    raise NotImplementedError("This function is intentionally removed.")


def init_op():
    from objprint import op

    op.config(color=True, line_number=True, arg_name=True)
    op.install("print")
    # op.install()
    # op.disable()


# cache parameters so the same message will only be printed once
@cache
def log_once(logger: Logger, msg: str):
    logger.warning(msg)
