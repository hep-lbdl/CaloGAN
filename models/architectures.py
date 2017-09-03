#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: architectures.py
description: sub-architectures for [arXiv/1705.02355]
author: Luke de Oliveira (lukedeo@manifold.ai)
"""

import keras.backend as K
from keras.initializers import constant
from keras.layers import (Dense, Reshape, Conv2D, LeakyReLU, BatchNormalization,
                          LocallyConnected2D, Activation, ZeroPadding2D,
                          Dropout, Lambda, Flatten)
from keras.layers.merge import concatenate, multiply
import numpy as np


from ops import (minibatch_discriminator, minibatch_output_shape,
                 Dense3D, sparsity_level, soft_sparsity_level,
                 sparsity_output_shape, hparams)


def sparse_softmax(x):
    x = K.relu(x)
    e = K.exp(x - K.max(x, axis=(1, 2, 3), keepdims=True))
    s = K.sum(e, axis=(1, 2, 3), keepdims=True)
    return e / s


def build_generator(x, nb_rows, nb_cols):
    """ Generator sub-component for the CaloGAN

    Args:
    -----
        x: a keras Input with shape (None, latent_dim)
        nb_rows: int, number of desired output rows
        nb_cols: int, number of desired output cols

    Returns:
    --------
        a keras tensor with the transformation applied
    """

    x = Dense((nb_rows + 2) * (nb_cols + 2) * 36, **hparams())(x)
    x = Reshape((nb_rows + 2, nb_cols + 2, 36))(x)

    x = Conv2D(64, (3, 3), padding='same', **hparams())(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(16, (2, 2), **hparams())(x)
    #    x = Conv2D(6, (2, 2), kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)

    x = Conv2D(1, (2, 2), use_bias=True,
               **hparams())(x)
    #x = Conv2D(1, (2, 2), use_bias=False, kernel_initializer='glorot_uniform')(x)
    return x


def build_discriminator(image, mbd=False, sparsity=False, sparsity_mbd=False,
                        soft_sparsity=True):
    """ Generator sub-component for the CaloGAN

    Args:
    -----
        image: keras tensor of 4 dimensions (i.e. the output of one calo layer)
        mdb: bool, perform feature level minibatch discrimination
        sparsiry: bool, whether or not to calculate and include sparsity
        sparsity_mdb: bool, perform minibatch discrimination on the sparsity 
            values in a batch

    Returns:
    --------
        a keras tensor of features

    """

    x = Conv2D(16, (5, 5), padding='same', **hparams())(image)
    x = LeakyReLU()(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(32, (3, 3), padding='valid', strides=(1, 2), **hparams())(x)
    #x = Conv2D(16, (3, 3), padding='valid', strides=(1, 2))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (2, 2), padding='valid', **hparams())(x)
    #x = Conv2D(8, (2, 2), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    # x = ZeroPadding2D((1, 1))(x)
    # x = Conv2D(128, (2, 2), padding='valid', strides=(1, 2))(x)
    # #x = Conv2D(8, (2, 2), padding='valid', strides=(1, 2))(x)
    # x = LeakyReLU()(x)
    # x = BatchNormalization()(x)

    x = Flatten()(x)

    if mbd or sparsity or sparsity_mbd:
        minibatch_featurizer = Lambda(minibatch_discriminator,
                                      output_shape=minibatch_output_shape)

        features = [x]
        nb_features = 10
        vspace_dim = 10

        # creates the kernel space for the minibatch discrimination
        if mbd:
            K_x = Dense3D(nb_features, vspace_dim)(x)
            features.append(Activation('tanh')(minibatch_featurizer(K_x)))

        if sparsity or sparsity_mbd:
            sparsity_detector = Lambda(
                sparsity_level if not soft_sparsity else soft_sparsity_level,
                sparsity_output_shape
            )
            empirical_sparsity = sparsity_detector(image)
            if sparsity:
                features.append(empirical_sparsity)
            if sparsity_mbd:
                K_sparsity = Dense3D(nb_features, vspace_dim)(empirical_sparsity)
                features.append(Activation('tanh')(minibatch_featurizer(K_sparsity)))

        return concatenate(features)
    else:
        return x
