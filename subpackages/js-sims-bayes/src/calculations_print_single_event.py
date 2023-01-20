#!/usr/bin/env python3
import sys
import numpy as np
from collections.abc import Iterable


# Input data format
from configurations import *
from calculations_file_format_single_event import return_result_dtype

filename=sys.argv[1]

result_dtype = return_result_dtype('ALICE')
data = np.fromfile(filename, dtype=result_dtype)

# Loop over data structure
# Assumes that "data" is a numpy array with dtype given
# by the array "structure" (though the latter is not a dtype object)
def print_data_structure(data, structure):

    n_items=len(structure)

    if (n_items > 0):
        for n, item in enumerate(structure):
            tmp_struct=structure[n]
            # If the item has substructure, recurse on it
            if (not isinstance(tmp_struct[1], str)) and (isinstance(tmp_struct[1], Iterable)):
                print(tmp_struct[0])
                print_data_structure(data[tmp_struct[0]],tmp_struct[1])
            # If no substructure, just output the result
            else:
                print(tmp_struct[0],data[tmp_struct[0]])

print_data_structure(data, result_dtype)
