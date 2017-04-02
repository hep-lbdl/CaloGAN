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


#def generator(latent_size, return_intermediate=False):
# def generator(latent_size, img_shape, return_intermediate=False):
def generator(z, img_shape, return_intermediate=False):
    '''
    img_shape = tuple of image eta-phi dimensions (e.g. (3, 96))
    '''

    # z = Input(shape=(latent_size, ))

    # DCGAN-style project & reshape,
    #x = Dense(5 * 98 * 12, input_dim=latent_size)(z)
    #x = Reshape((5, 98, 12))(x)
    x = Dense((img_shape[0] + 2) * (img_shape[1] + 2) * 12, name='gen_project')(z)
    x = Reshape((img_shape[0] + 2, img_shape[1] + 2, 12), name='gen_reshape')(x)

    # block 1: (None, 5, 98, 12) => (None, 5, 98, 8),
    x = Conv2D(8, (2, 2), padding='same', kernel_initializer='he_uniform', name='gen_conv2d')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    
    # block 2: (None, 5, 98, 32) => (None, 4, 97, 6),
    #ZeroPadding2D((2, 2)),
    x = LocallyConnected2D(6, (2, 2), kernel_initializer='he_uniform', name='gen_lc2d_1')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
            
    # block 3: (None, 4, 97, 6) => (None, 3, 96, 1),
    x = LocallyConnected2D(1, (2, 2), use_bias=False, kernel_initializer='glorot_normal', name='gen_lc2d_2')(x)
    y = Activation('relu')(x)

    return y
    
    # return Model(z, y)
