#!/usr/bin/env python
"""
- For all 2D pickle files, change class from Model to Model2D
- For all pickle files with module references not at the package level,
  meaning referring to `x` rather than `bannock.x`, change this.
"""

from glob import glob
import utils

dirnames = glob('autochemo_*')

good_model_name = '\nModel2D\n'
bad_model_name = '\nModel\n'

good_module_name = '\n(cbannock.model\n'
bad_module_name = '\n(cmodel\n'


good_walls_name = '\n(cbannock.walls\n'
bad_walls_name = '\n(cwalls\n'

for dirname in dirnames:
    print(dirname)
    pickle_fnames = utils.get_filenames(dirname)
    for pickle_fname in pickle_fnames:
        with open(pickle_fname, 'r') as pickle_file:
            pickle_str = pickle_file.read()
            changed = False
            if good_module_name not in pickle_str:
                print('Found broken pickle (module): {}'.format(pickle_fname))
                if bad_module_name in pickle_str:
                    pickle_str = pickle_str.replace(bad_module_name,
                                                    good_module_name)
                    changed = True
            if ('dim=2' in dirname and good_model_name not in pickle_str):
                print('Found broken pickle (model): {}'.format(pickle_fname))
                if bad_model_name in pickle_str:
                    pickle_str = pickle_str.replace(bad_model_name,
                                                    good_model_name)
                    changed = True
            if good_walls_name not in pickle_str:
                print('Found broken pickle (walls): {}'.format(pickle_fname))
                if bad_walls_name in pickle_str:
                    pickle_str = pickle_str.replace(bad_walls_name,
                                                    good_walls_name)
                    changed = True
            if changed:
                with open(pickle_fname, 'w') as pickle_file:
                    pickle_file.write(pickle_str)
