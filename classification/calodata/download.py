#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: download.py
description: utils to download data from [mendeley:10.17632/pvn3xc3wy5.1]
author: Luke de Oliveira (lukedeoliveira@lbl.gov)
"""

from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle

import argparse
import os
from six.moves import range
import sys

from h5py import File as HDF5File
import numpy as np


from tqdm import tqdm
import requests


def data_url(identifier='pvn3xc3wy5', version=1):
    return 'https://data.mendeley.com/archiver/{}?version={}'.format(identifier, version)

if __name__ == '__main__':

    _report_start_time = 0

    def _reporthook(count, block_size, total_size):
        global _report_start_time
        if count == 1:
            _report_start_time = time.time()
            return
        duration = time.time() - _report_start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            '\r[downloading...] %d%%, %d MB, %d KB/s, %d seconds passed' %
            (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    if sys.version_info[0] == 2:
        def urlretrieve(url, filename, reporthook=None, data=None):
            """
            Credit: Keras (https://github.com/fchollet/keras)
            """
            def chunk_read(response, chunk_size=8192, reporthook=None):
                content_type = response.info().get('Content-Length')
                total_size = -1
                if content_type is not None:
                    total_size = int(content_type.strip())
                count = 0
                while 1:
                    chunk = response.read(chunk_size)
                    count += 1
                    if not chunk:
                        reporthook(count, total_size, total_size)
                        break
                    if reporthook:
                        reporthook(count, chunk_size, total_size)
                    yield chunk

            response = urlopen(url, data)
            with open(filename, 'wb') as fd:
                for chunk in chunk_read(response, reporthook=reporthook):
                    fd.write(chunk)
    else:
        from six.moves.urllib.request import urlretrieve

    import argparse

    parser = argparse.ArgumentParser(
        description=('Download the data from [mendeley:10.17632/pvn3xc3wy5.1] '
                     'locally. Simply, this will download the file in a '
                     'standard manner.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--quiet', '-q', action='store_true',
                        help="Don't show progress information during download")

    parser.add_argument('--no-unzip', action='store_true',
                        help="Don't unzip final file")

    parser.add_argument('filepath', nargs=1, help='Filepath to write to')

    # add arguments here
    results = parser.parse_args()
