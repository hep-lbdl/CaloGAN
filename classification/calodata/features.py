import shower_shapes as sh
import pandas as pd

def extract_dataframe(data):

    total_energy = sh.total_energy(data)
    lateral_depth = sh.lateral_depth(data)
    lateral_depth2 = sh.lateral_depth2(data)

    features = {
        'max_depth': sh.depth(data),
        'total_energy': total_energy,
        'lateral_depth': lateral_depth,
        'lateral_depth2': lateral_depth2,
        'shower_depth_width': sh.shower_depth_width(
            lateral_depth,
            lateral_depth2,
            total_energy
        )
    }

    for layer in xrange(3):
        l_energy = sh.energy(layer, data)
        features.update({
            'sparsity_layer_{}'.format(layer): sh.sparsity(layer, data),
            'energy_layer_{}'.format(layer): l_energy,
            'eratio_layer_{}'.format(layer): sh.eratio(layer, data),
            'lat_width_layer_{}'.format(layer): sh.lateral_width(layer, data),
            'efrac_layer_{}'.format(layer): sh.efrac(l_energy, total_energy)
        })

    return pd.DataFrame(features)


def extract_features(data):
    return extract_dataframe(data).values
