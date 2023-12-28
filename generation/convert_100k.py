#!/usr/bin/env python
# -*- coding: utf-8 -*-
from rootpy.io import root_open
from root_numpy import tree2array
import pandas as pd
import numpy as np
from h5py import File as HDF5File

LAYER_SPECS = [(3, 96), (12, 12), (12, 6)]

LAYER_DIV = np.cumsum(map(np.prod, LAYER_SPECS)).tolist()
LAYER_DIV = zip([0] + LAYER_DIV, LAYER_DIV)

OVERFLOW_BINS = 3

KEEP_NR = 100000

def write_out_file(infiles, outfile, tree=None):
    for idx, infile in enumerate(infiles):
        f = root_open(infile)
	T = f[tree]

        cells = filter(lambda x: x.startswith('cell'), T.branchnames)


        assert len(cells) == sum(map(np.prod, LAYER_SPECS)) + OVERFLOW_BINS

        if idx == 0:
            X_frame = pd.DataFrame(tree2array(T, branches=cells))
            E_frame = pd.DataFrame(tree2array(T, branches=['TotalEnergy']))
        else:
            X_frame = X_frame.append(pd.DataFrame(tree2array(T, branches=cells)))
            E_frame = E_frame.append(pd.DataFrame(tree2array(T, branches=['TotalEnergy'])))
        print("Done with file {} of {}".format(idx+1, len(infiles)), '\r')
        print(len(E_frame))
    orig_length = len(E_frame)
    print("Files have {} events total".format(orig_length))
    duplicates = X_frame.duplicated()
    X = X_frame[~duplicates].values
    E = E_frame[~duplicates].values.ravel()
    new_length = len(E)
    print("There were {} duplicates removed, continue with {}".format(orig_length-new_length,
                                                                      KEEP_NR))
    if new_length < KEEP_NR:
        raise ValueError('Not enough events in the files '
                         '(got {} unique ones, need {})'.format(new_length, KEEP_NR))
    X = X[:KEEP_NR]
    E = E[:KEEP_NR]
    with HDF5File(outfile, 'w') as h5:
        for layer, (sh, (l, u)) in enumerate(zip(LAYER_SPECS, LAYER_DIV)):
            h5['layer_{}'.format(layer)] = X[:, l:u].reshape((-1, ) + sh)*5.

        h5['overflow'] = X[:, -OVERFLOW_BINS:]*5.
        h5['energy'] = E.reshape(-1, 1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert GEANT4 output files into ML-able HDF5 files')

    parser.add_argument('--in-files', '-i', nargs='+', action="store", required=True,
                        help='input ROOT files')
    parser.add_argument('--out-file', '-o', action="store", required=True,
                        help='output HDF5 file')
    parser.add_argument('--tree', '-t', action="store", required=True,
                        help='input tree for the ROOT file')

    args = parser.parse_args()

    write_out_file(infiles=args.in_files, outfile=args.out_file, tree=args.tree)
