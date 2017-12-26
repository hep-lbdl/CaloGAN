
'''
Note:
- layer naming convention: {0, 1, 2}
- no overflow bins in hdf5
'''
import numpy as np

#a small number to protect float division by 0
myepsilon = 0.000001

def depth(d):
    '''
    For each event, it finds the deepest layer in which the shower has deposited some E.
    Args:
    -----
d: an h5py File with fields 'layer_2', 'layer_1', 'layer_0'
   that represent the 2d cell grids and the corresponding
   E depositons.
    Returns:
    --------

    '''
    maxdepth = 2 * (d['layer_2'][:].sum(axis=(1,2)) != 0)
    maxdepth[maxdepth == 0] = 1 * (d['layer_1'][:][maxdepth == 0].sum(axis=(1,2)) != 0)
    return maxdepth


def total_energy(d):
    '''
    Calculates the total energy for each event across all layers.
    Args:
    -----
d: an h5py File with fields 'layer_2', 'layer_1', 'layer_0'
   that represent the 2d cell grids and the corresponding
   E depositons.
    '''
    return d['layer_0'][:].sum(axis=(1, 2)) + d['layer_1'][:].sum(axis=(1, 2)) + d['layer_2'][:].sum(axis=(1, 2))

def energy(layer, d):
    '''
    Finds total E deposited in a given layer for each event.
    Args:
    -----
layer: int in {0, 1, 2} that labels the layer
d: an h5py File with fields 'layer_2', 'layer_1', 'layer_0'
   that represent the 2d cell grids and the corresponding
   E depositons.
    Returns:
    --------
    the total enery deposited in a given layer for each event
    '''
    return d['layer_{}'.format(layer)][:].sum(axis=(1, 2))


def efrac(elayer, total_energy):
    '''
    Finds the fraction of E deposited in a given layer for each event.
    Args:
    -----
elayer: float, energy in a given layer for each event
total_energy: float, total energy per event
    Returns:
    --------
E_layer / E_total
    '''
    return  elayer / total_energy


def lateral_depth(d):
    '''
    Sum_{i} E_i * d_i
    '''
    return (d['layer_2'][:] * 2).sum(axis=(1,2)) + (d['layer_1'][:]).sum(axis=(1,2))


def lateral_depth2(d):
    '''
    Sum_{i} E_i * d_i^2
    '''
    return (d['layer_2'][:] * 2 * 2).sum(axis=(1,2)) + (d['layer_1'][:]).sum(axis=(1,2))


def shower_depth(lateral_depth, total_energy):
    '''
    lateral_depth / total_energy
    Args:
    -----
lateral_depth: float, Sum_{i} E_i * d_i
total_energy: float, total energy per event
    '''
    return lateral_depth / total_energy


def shower_depth_width(lateral_depth, lateral_depth2, total_energy):
    '''
    sqrt[lateral_depth2 / total_energy - (lateral_depth / total_energy)^2]
    Args:
    -----
lateral_depth: float, Sum_{i} E_i * d_i
lateral_depth2: float, Sum_{i} E_i * d_i * d_i
total_energy: float, total energy per event
    '''
    return np.sqrt((lateral_depth2 / total_energy) - (lateral_depth / total_energy)**2)


def layer_lateral_width(layer, d):
    '''
    Args:
    -----
layer: int in {0, 1, 2} that labels the layer
d: an h5py File with fields 'layer_2', 'layer_1', 'layer_0'
   that represent the 2d cell grids and the corresponding
   E depositons.
    '''
    e = energy(layer, d)
    eta_cells = {'layer_0' : 3, 'layer_1' : 12, 'layer_2' : 12}
    eta_bins = np.linspace(-240, 240, eta_cells['layer_' + str(layer)] + 1)
    bin_centers = (eta_bins[1:] + eta_bins[:-1]) / 2.
    x = (d['layer_{}'.format(layer)] * bin_centers.reshape(-1, 1)).sum(axis=(1,2))
    x2 = (d['layer_{}'.format(layer)] * (bin_centers.reshape(-1, 1) ** 2)).sum(axis=(1,2))
    return np.sqrt((x2 / (e+myepsilon)) - (x / (e+myepsilon)) ** 2)

def eratio(images):
    top2 = np.array([np.sort(row.ravel())[::-1][:2] for row in images])
    return (top2[:, 0] - top2[:, 1]) / (myepsilon+top2[:, 0] + top2[:, 1])
