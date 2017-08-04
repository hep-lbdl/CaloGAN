#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: load.py
description: load the file that you download
author: Luke de Oliveira (lukedeoliveira@lbl.gov)
"""

import h5py


def load_calodata(fpath):
    with h5py.File(fpath, 'r') as h5:
        data = [h5['layer_{}'.format(i)][:] for i in xrange(3)]
    return data
