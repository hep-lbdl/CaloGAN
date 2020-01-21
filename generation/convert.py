#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: convert.py
description: Convert GEANT4 root files into hdf5 files
author: Luke de Oliveira (lukedeo@manifold.ai)
"""

from rootpy.io import root_open
from root_numpy import tree2array
import pandas as pd
import numpy as np
from h5py import File as HDF5File

LAYER_SPECS = [(3, 96), (12, 12), (12, 6)]

LAYER_DIV = np.cumsum(map(np.prod, LAYER_SPECS)).tolist()
LAYER_DIV = zip([0] + LAYER_DIV, LAYER_DIV)

OVERFLOW_BINS = 3


def write_out_file(infile, outfile, tree=None):
    f = root_open(infile)
    T = f[tree]

    cells = filter(lambda x: x.startswith('cell'), T.branchnames)

    assert len(cells) == sum(map(np.prod, LAYER_SPECS)) + OVERFLOW_BINS

    X = pd.DataFrame(tree2array(T, branches=cells)).values
    E = pd.DataFrame(tree2array(T, branches=['TotalEnergy'])).values.ravel()

    with HDF5File(outfile, 'w') as h5:
        for layer, (sh, (l, u)) in enumerate(zip(LAYER_SPECS, LAYER_DIV)):
            h5['layer_{}'.format(layer)] = X[:, l:u].reshape((-1, ) + sh)

        h5['overflow'] = X[:, -OVERFLOW_BINS:]
        h5['energy'] = E.reshape(-1, 1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert GEANT4 output files into ML-able HDF5 files')

    parser.add_argument('--in-file', '-i', action="store", required=True,
                        help='input ROOT file')
    parser.add_argument('--out-file', '-o', action="store", required=True,
                        help='output HDF5 file')
    parser.add_argument('--tree', '-t', action="store", required=True,
                        help='input tree for the ROOT file')

    args = parser.parse_args()

    write_out_file(infile=args.in_file, outfile=args.out_file, tree=args.tree)
