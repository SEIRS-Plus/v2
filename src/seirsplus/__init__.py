# Standard Libraries
from importlib import metadata

# Internal Libraries
from .utils.logging_utils import _configure_seirsplus_loggers


_configure_seirsplus_loggers(root_module_name=__name__)
__version__ = metadata.version(__package__)

__all__ = []
