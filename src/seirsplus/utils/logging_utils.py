"""
Defines default logging configurations and methods for sending logging messages
directly to stderr
"""
# Standard Libraries
import sys
import logging
import logging.config

LOGGING_LINE_FORMAT = "{asctime} {levelname} {name} lineno:{lineno}: {message}"
LOGGING_DATETIME_FORMAT = "[%Y-%m-%d %H:%M:%S]"
LOGGING_STYLE = "{"


class SEIRSPlusLoggingStream:
    """A Python stream for use with event logging APIs"""

    def __init__(self):
        self._enabled = True

    def write(self, text):
        if self._enabled:
            sys.stderr.flush()

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value


SEIRSPLUS_LOGGING_STREAM = SEIRSPlusLoggingStream()


def disable_logging():
    SEIRSPLUS_LOGGING_STREAM.enabled = False


def enable_logging():
    SEIRSPLUS_LOGGING_STREAM.enabled = True


def _configure_seirsplus_loggers(root_module_name: str):
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_formatter": False,
            "formatters": {
                "seirsplus_formatter": {
                    "format": LOGGING_LINE_FORMAT,
                    "datefmt": LOGGING_DATETIME_FORMAT,
                    "style": LOGGING_STYLE,
                }
            },
            # Handler can output any 'format' to any 'target', e.g. output
            # a log to a structlog service
            "handlers": {
                "seirsplus_handler": {
                    "level": "INFO",
                    "formatter": "seirsplus_formatter",
                    "class": "logging.StreamHandler",
                    "stream": SEIRSPLUS_LOGGING_STREAM,
                }
            },
            "loggers": {
                root_module_name: {
                    "handlers": ["seirsplus_handler"],
                    "level": "INFO",
                    "propagate": False,
                }
            },
        }
    )


def eprint(*args, **kwargs):
    """prints to stderr"""
    print(*args, file=SEIRSPLUS_LOGGING_STREAM, **kwargs)
