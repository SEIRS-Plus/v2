"""
Defines decorators for communicating usage guidelines to developers and users
"""
# Standard Libraries
import warnings
from functools import wraps


def experimental(func):
    """
    Decorator for marking APIs experimental in the docstring.

    Args:
        func: A function to mark

    Returns:
        Decorated function.
    """
    notice = (
        ".. Note:: Experimental: This method may change or "
        + "be removed in a future release without warning.\n"
    )
    func.__doc__ = notice + func.__doc__
    return func


def deprecated(alternative=None, since=None):
    """
    Decorator for marking APIs deprecated in the docstring.

    Args:
        alternative: alternative feature to use
        since: version number

    Returns:
        Decorated function.
    """

    def deprecated_decorator(func):
        since_str = " since %s" % since if since else ""
        notice = (
            ".. Warning:: ``{function_name}`` is deprecated {since_string}. "
            "This method will be removed in a near future release."
        ).format(
            function_name=".".join([func.__module__, func.__name__]),
            since_string=since_str,
        )
        if alternative is not None and alternative.strip():
            notice += f" Use ``{alternative}`` instead."

        @wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(notice, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        if func.__doc__ is not None:
            deprecated_func.__doc__ = notice + "\n" + func.__doc__

        return deprecated_func

    return deprecated_decorator


def keyword_only(func):
    """
    A decorator that forces keyword arguments in the wrapped method.

    Args:
        func: A function to mark
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            raise TypeError(f"Method {func.__name__} only takes keyword arguments.")
        return func(**kwargs)

    notice = ".. Note:: This method requires all argument be specified by keyword.\n"
    wrapper.__doc__ = notice + wrapper.__doc__
    return wrapper
