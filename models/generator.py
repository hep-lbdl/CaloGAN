#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: generators.py
"""
import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Embedding, merge,
                          Dropout, BatchNormalization, Activation)

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.local import LocallyConnected2D
from keras.layers.convolutional import UpSampling2D, Conv2D, ZeroPadding2D

from keras.models import Model, Sequential

K.set_image_dim_ordering('tf')


def generator(latent_size, return_intermediate=False):

 
    z = Input(shape=(latent_size, ))

    # DCGAN-style project & reshape,
    x = Dense(5 * 98 * 12, input_dim=latent_size)(z)
    x = Reshape((5, 98, 12))(x)
    
    # block 1: (None, 5, 98, 12) => (None, 5, 98, 8),
    x = Conv2D(8, 2, 2, border_mode='same', init='he_uniform')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    
    # block 2: (None, 5, 98, 32) => (None, 4, 97, 6),
    #ZeroPadding2D((2, 2)),
    x = LocallyConnected2D(6, 2, 2, init='he_uniform')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
            
    # block 3: (None, 4, 97, 6) => (None, 3, 96, 1),
    x = LocallyConnected2D(1, 2, 2, bias=False, init='glorot_normal')(x)
    y = Activation('relu')(x)
    
    return Model(z, y)
