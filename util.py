from __future__ import annotations
import os
import csv
from typing import Dict, Tuple
##############################################################################################################


def lookup(object_name: str, key_column: str = 'OBJECT_NAME'):
    """
    Reads a CSV file and finds the row data associated with a specific object name.

    Args:
        object_name (str): The exact value to search for in the key_column.
        key_column (str): The name of the column to search within (default: 'OBJECT_NAME').

    Returns:
        dict or None: A dictionary containing the row data if found, otherwise None.
    """
    filename = "tle.csv"
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        return None

    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
            # Use csv.DictReader to read the CSV data into a list of dictionaries.
            # This allows accessing columns by their header name (e.g., row['OBJECT_NAME']).
            reader = csv.DictReader(csvfile)

            for row in reader:
                # Check if the value in the key_column matches the search term
                if row.get(key_column) == object_name:
                    print(f"--- Data Found for {object_name} ---")
                    return row  # Return the entire dictionary for the matching row

            # If the loop completes without a match
            print(f"No data found for object name: '{object_name}' in the column '{key_column}'.")
            return None

    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None
## END TLE PARSING UTILITY FUNCTIONS ###

print(lookup("KUIPER-00015"))


def load_tle_file() -> Dict[str, Tuple[str, str]]:
	"""Load TLEs from a file and return {name: (line1, line2)}.

	The loader expects entries in the form:
	  NAME_LINE
	  1 <line1 data>
	  2 <line2 data>

	It is robust to extra blank lines and scans forward when the pattern
	does not match at the current position.
	"""
	path = "tle.txt"
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
