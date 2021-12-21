# Standard Libraries
try:
	from importlib import metadata
except ImportError:
	# Try backported to PY<37 `importlib_metadata`.
	import importlib_metadata as metadata

# Internal Libraries
from .dev_tools.logging_utils import _configure_seirsplus_loggers


_configure_seirsplus_loggers(root_module_name=__name__)
__version__ = metadata.version(__package__)

__all__ = []
