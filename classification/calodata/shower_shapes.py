#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: shower_shapes.py
description: calculate shower shapes efficiently
author: Luke de Oliveira (lukedeoliveira@lbl.gov), 
        Michela Paganini (michela.paganini@yale.edu)

notes: TODO - docstrings everywhere
"""

import numpy as np


def depth(data):
    '''
    For each event, it finds the deepest layer in which the shower has 
    deposited some E.

    '''
    maxdepth = 2 * (data[2].sum(axis=(1, 2)) != 0)
    maxdepth[maxdepth == 0] = 1 * (data[1][maxdepth == 0].sum(axis=(1, 2)) != 0)
    return maxdepth


def total_energy(data):
    '''
    Calculates the total energy for each event across all layers.
    '''
    return (data[0].sum(axis=(1, 2)) +
            data[1].sum(axis=(1, 2)) +
            data[2].sum(axis=(1, 2)))


def energy(layer, data):
    '''
    Finds total E deposited in a given layer for each event.
    '''
    return data[layer].sum(axis=(1, 2))


def sparsity(layer, data):
    '''
    '''
    return np.divide(1.0 * (data[layer] > 0.0).sum(axis=(1, 2)),
                     np.prod(data[layer].shape[1:]))


def efrac(elayer, total_energy):
    '''
    Finds the fraction of E deposited in a given layer for each event.
    '''
    return elayer / total_energy


def lateral_depth(data):
    '''
    Sum_{i} E_i * d_i
    '''
    return (data[2] * 2).sum(axis=(1, 2)) + (data[1]).sum(axis=(1, 2))


def lateral_depth2(data):
    '''
    Sum_{i} E_i * d_i^2
    '''
    return (data[2] * 2 * 2).sum(axis=(1, 2)) + (data[1]).sum(axis=(1, 2))


def shower_depth(lateral_depth, total_energy):
    '''
    lateral_depth / total_energy
    '''
    return lateral_depth / total_energy


def shower_depth_width(lateral_depth, lateral_depth2, total_energy):
    '''
    sqrt[lateral_depth2 / total_energy - (lateral_depth / total_energy)^2]
    '''
    total_energy += 1e-9
    return np.sqrt(
        (lateral_depth2 / total_energy) -
        (lateral_depth / total_energy) ** 2
    )


def lateral_width(layer, data):
    '''
    '''
    e = energy(layer, data) + 1e-9
    eta_cells = {
        0: 3,
        1: 12,
        2: 12
    }

    eta_bins = np.linspace(-240, 240, eta_cells[layer] + 1)

    bin_centers = (eta_bins[1:] + eta_bins[:-1]) / 2.
    x = (data[layer] * bin_centers.reshape(-1, 1)).sum(axis=(1, 2))
    x2 = (data[layer] * (bin_centers.reshape(-1, 1) ** 2)).sum(axis=(1, 2))
    return np.sqrt((x2 / e) - (x / e) ** 2)


def eratio(layer, data):
    '''
    '''
    top2 = np.array([np.sort(row.ravel())[::-1][:2] for row in data[layer]])
    return (top2[:, 0] - top2[:, 1]) / (top2[:, 0] + top2[:, 1] + 1e-7)
