#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: generators.py
"""
import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Embedding, merge,
                          Dropout, BatchNormalization, Activation, Lambda, Conv1D)

from keras.layers.merge import add, concatenate, multiply

from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.layers.local import LocallyConnected2D
from keras.layers.convolutional import UpSampling2D, Conv2D, ZeroPadding2D

from ops import (minibatch_discriminator, minibatch_output_shape, Dense3D,
                 single_layer_energy, single_layer_energy_output_shape,
                 sparsity_level, sparsity_output_shape)


from keras.models import Model, Sequential

K.set_image_data_format('channels_last')


# def generator(latent_size, return_intermediate=False):
# def generator(latent_size, image_shapee, return_intermediate=False):
def layer_0_generator(latent_size, *args, **kwargs):
    '''
    image_shapee = tuple of image eta-phi dimensions (e.g. (3, 96))
    '''

    IMAGE_SHAPE = (3, 96)

    z = Input(shape=(latent_size, ))

    # DCGAN-style project & reshape,
    #x = Dense(5 * 98 * 12, input_dim=latent_size)(z)
    #x = Reshape((5, 98, 12))(x)

    # def _1dconv_block(x):

    #     x = Dense(100 * 96)(x)

    #     x = Reshape((96, 100))(x)
    #     x = LeakyReLU()(x)

    #     x = Conv1D(filters=100, kernel_size=, padding='same')(x)
    #     x = LeakyReLU()(x)

    #     x = Conv1D(filters=50, kernel_size=, padding='same')(x)
    #     x = LeakyReLU()(x)

    #     x = Conv1D(filters=3, kernel_size=, padding='same')(x)
    #     x = Reshape((3, 96, 1))(x)
    #     return x
    # x = LeakyReLU()(x)

    x = Dense((IMAGE_SHAPE[0] + 1) * (IMAGE_SHAPE[1] + 1) * 32)(z)
    x = PReLU()(x)

    def _lcn_block(z):
        x = Dense((IMAGE_SHAPE[0] + 3) * (IMAGE_SHAPE[1] + 3) * 12)(z)
        x = Reshape((IMAGE_SHAPE[0] + 3, IMAGE_SHAPE[1] + 3, 12))(x)

        x = Conv2D(8, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        # block 2: (None, 5, 98, 32) => (None, 4, 97, 6),
        #ZeroPadding2D((2, 2)),
        x = LocallyConnected2D(8, (2, 2), kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        x = LocallyConnected2D(6, (2, 2), kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)

        # block 3: (None, 4, 97, 6) => (None, 3, 96, 1),
        x = LocallyConnected2D(1, (2, 2), use_bias=False,
                               kernel_initializer='glorot_normal')(x)
        return x

    def _conv_block(x):
        # reshape to conv dims
        x = Reshape((IMAGE_SHAPE[0] + 1, IMAGE_SHAPE[1] + 1, 32))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        x = Conv2D(32, (2, 2), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        x = Conv2D(16, (2, 2), padding='same')(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization()(x)

        x = LocallyConnected2D(1, (2, 2), padding='valid')(x)

        return x

    def _dense_block(x):
        x = Dense((IMAGE_SHAPE[0] + 2) * (IMAGE_SHAPE[1] + 2) * 2)(x)
        x = Activation('tanh')(x)
        x = Dense((IMAGE_SHAPE[0] + 2) * (IMAGE_SHAPE[1] + 2) * 2)(x)
        x = Activation('tanh')(x)
        x = Dense(IMAGE_SHAPE[0] * IMAGE_SHAPE[1])(x)
        x = Reshape(IMAGE_SHAPE + (1, ))(x)
        return x

    # x = add([_1dconv_block(z), _lcn_block(z)])
    x = _lcn_block(z)
    # x = mean([_conv_block(x), _dense_block(x), _lcn_block(z)])

    y = Activation('relu')(x)

    return Model(z, y)

    # return Model(z, y)


def layer_0_discriminator(*args, **kwargs):
    '''
    image_shapee = tuple of image eta-phi dimensions (e.g. (3, 96))
    '''

    IMAGE_SHAPE = (3, 96)

    image = Input(shape=IMAGE_SHAPE + (1, ))

    def _conv_block(x):
        x = Conv2D(128, (2, 4), padding='same')(x)
        x = LeakyReLU()(x)
        # x = BatchNormalization()(x)

        x = Conv2D(64, (2, 4), strides=(1, 2), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        x = Conv2D(32, (2, 4), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        x = Conv2D(16, (2, 4), strides=(1, 2), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        x = Conv2D(8, (2, 4), strides=(1, 2), padding='same')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)

        return x

    def _dense_block(x):

        x = Flatten()(x)

        x = Dense((IMAGE_SHAPE[0]) * (IMAGE_SHAPE[1]))(x)
        x = PReLU()(x)

        x = Dense((IMAGE_SHAPE[0]) * (IMAGE_SHAPE[1]))(x)
        x = PReLU()(x)

        x = Dense((IMAGE_SHAPE[0]) * (IMAGE_SHAPE[1]))(x)
        x = PReLU()(x)

        return x

    def _lcn_block(x):
        # block 1: normal 2x2 conv,
        # *NO* batchnorm (recommendation from [arXiv/1511.06434])
        x = Conv2D(16, (2, 2), padding='same', name='disc_conv2d')(image)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        # block 2: 'same' bordered 3x3 locally connected block with batchnorm and
        # 2x2 subsampling
        x = ZeroPadding2D((1, 1))(x)
        x = LocallyConnected2D(8, (3, 3), padding='valid',
                               strides=(1, 2), name='disc_lc2d_1')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # block 2: 'same' bordered 5x5 locally connected block with batchnorm
        x = ZeroPadding2D((1, 1))(x)
        x = LocallyConnected2D(8, (2, 2), padding='valid', name='disc_lc2d_2')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # block 3: 2x2 locally connected block with batchnorm and
        # 1x2 subsampling
        x = ZeroPadding2D((1, 1))(x)
        x = LocallyConnected2D(8, (2, 2), padding='valid',
                               strides=(1, 2), name='disc_lc2d_3')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        #x = AveragePooling2D((2, 2))(x)
        h = Flatten()(x)

        return h

    # x = concatenate([_conv_block(image), _dense_block(image), _lcn_block(image)])
    x = _lcn_block(image)

    # # nb of features to obtain
    nb_features = 20

    # dim of kernel space
    vspace_dim = 10

    # creates the kernel space for the minibatch discrimination

    minibatch_featurizer = Lambda(minibatch_discriminator,
                                  output_shape=minibatch_output_shape)

    energy_detector = Lambda(single_layer_energy, single_layer_energy_output_shape)
    sparsity_detector = Lambda(sparsity_level, sparsity_output_shape)

    energy = energy_detector(image)
    sparsity = sparsity_detector(image)

    K_x = Dense3D(nb_features, vspace_dim)(x)
    K_energy = Dense3D(nb_features, vspace_dim)(energy)
    K_sparsity = Dense3D(nb_features, vspace_dim)(sparsity)

    # concat the minibatch features with the normal ones
    features = merge([
        minibatch_featurizer(K_x),
        minibatch_featurizer(K_energy),
        minibatch_featurizer(K_sparsity),
        x,
        energy,
        sparsity
    ], mode='concat')

    y = Dense(1, activation='sigmoid')(features)

    return Model(image, y)

    # return Model(z, y)
