try:
	import importlib.resources as pkg_resources
except ImportError:
	# Try backported to PY<37 `importlib_resources`.
	import importlib_resources as pkg_resources

import numpy as np
import json
from seirsplus.models import configs


def load_config(filename):
	try:
		with pkg_resources.open_text(configs, filename) as cfg_file:
			if('.json' in filename):
				cfg_dict = json.load(cfg_file)
				return cfg_dict
			elif('.csv' in filename):
				cfg_array = np.genfromtxt(cfg_file, delimiter=',')
				return cfg_array
			elif('.tsv' in filename):
				cfg_array = np.genfromtxt(cfg_file, delimiter='\t')
				return cfg_array
			else:
				print("load_config error: File type not supported (support for .json, .csv, .tsv)")
				exit()
	except FileNotFoundError:
		print("load_config error: Config file \'"+filename+"\' not found in seirsplus.models.configs.")
		exit()


