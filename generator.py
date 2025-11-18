from __future__ import annotations
import os
from typing import Dict, Tuple

def load_tle_file() -> Dict[str, Tuple[str, str]]:
	"""Load TLEs from a file and return {name: (line1, line2)}.

	The loader expects entries in the form:
	  NAME_LINE
	  1 <line1 data>
	  2 <line2 data>

	It is robust to extra blank lines and scans forward when the pattern
	does not match at the current position.
	"""

def load_tle_file(path: str = "tle.txt") -> Dict[str, Tuple[str, str]]:
		"""Load TLEs from a file and return {name: (line1, line2)}.

		The loader expects entries in the form:
		  NAME_LINE
		  1 <line1 data>
		  2 <line2 data>

		It is robust to extra blank lines and scans forward when the pattern
		does not match at the current position.
		"""
		tle_dict: Dict[str, Tuple[str, str]] = {}
		with open(path, 'r') as f:
			lines = [ln.rstrip('\n') for ln in f]

		i = 0
		n = len(lines)
		while i < n - 2:
			name = lines[i].strip()
			l1 = lines[i + 1].strip()
			l2 = lines[i + 2].strip()
			if l1.startswith('1 ') and l2.startswith('2 '):
				key = name if name else l1[2:7].strip()
				# Avoid overwriting existing entries with the same name
				if key in tle_dict:
					# If duplicate, append a suffix to keep entries unique
					suffix = 2
					new_key = f"{key}_{suffix}"
					while new_key in tle_dict:
						suffix += 1
						new_key = f"{key}_{suffix}"
					key = new_key
				tle_dict[key] = (l1, l2)
				i += 3
			else:
				# Advance one line and try again (handles files without a name line)
				i += 1

		return tle_dict


def write_tle_module(tle_dict: Dict[str, Tuple[str, str]], out_path: str = 'tle_data.py') -> None:
		"""Write a Python module that defines `tle_data` dict.

		The generated file will contain a top-level variable `tle_data` which is a
		dict mapping satellite name to a 2-tuple of the TLE lines. Strings are
		written with Python repr() so the module is importable.
		"""
		with open(out_path, 'w') as f:
			f.write('# Auto-generated from tle.txt\n')
			f.write('# Do not edit manually (regenerate from tle.txt)\n\n')
			f.write('tle_data = {\n')
			for key, (l1, l2) in tle_dict.items():
				f.write(f"    {repr(key)}: (\n")
				f.write(f"        {repr(l1)},\n")
				f.write(f"        {repr(l2)}\n")
				f.write("    ),\n")
			f.write('}\n')


def main() -> None:
		write_tle_module(load_tle_file())

if __name__ == '__main__':
	main()
