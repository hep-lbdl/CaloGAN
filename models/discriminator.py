#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: discriminators.py
description: discrimination submodel for [arXiv/1701.05927]
author: Luke de Oliveira (lukedeoliveira@lbl.gov)
"""

import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling2D, Conv2D, ZeroPadding2D,
                                        AveragePooling2D)
from keras.layers.local import LocallyConnected2D
from keras.models import Model

from ops import minibatch_discriminator, minibatch_output_shape, Dense3D

K.set_image_dim_ordering('tf')


def discriminator(image):

    # image = Input(shape=(3, 96, 1))
    #image = Input(shape=(img_shape[0], img_shape[1], 1))

    # block 1: normal 2x2 conv,
    # *NO* batchnorm (recommendation from [arXiv/1511.06434])
    x = Conv2D(32, (2, 2), padding='same', name='disc_conv2d')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    
    # block 2: 'same' bordered 3x3 locally connected block with batchnorm and
    # 2x2 subsampling
    x = ZeroPadding2D((1, 1))(x)
    x = LocallyConnected2D(8, (3, 3), padding='valid', strides=(1, 2), name='disc_lc2d_1')(x)
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
    x = LocallyConnected2D(8, (2, 2), padding='valid', strides=(1, 2), name='disc_lc2d_3')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    #x = AveragePooling2D((2, 2))(x)
    h = Flatten()(x)

    #dnn = Model(image, h)

    # evt_image = Input(shape=(3, 96, 1))
    #evt_image = Input(shape=(img_shape[0], img_shape[1], 1))

    #out = dnn(evt_image)
    out = h 

    # nb of features to obtain
    nb_features = 20

    # dim of kernel space
    vspace_dim = 10

    # creates the kernel space for the minibatch discrimination
    K_x = Dense3D(nb_features, vspace_dim)(h)#(out)

    minibatch_featurizer = Lambda(minibatch_discriminator,
                              output_shape=minibatch_output_shape)

    # concat the minibatch features with the normal ones
    features = merge([
            minibatch_featurizer(K_x),
            h
            ], mode='concat')

    # fake output tracks binary fake / not-fake, and the auxiliary requires
    # reconstruction of latent features, in this case, labels
    # fake = Dense(1, activation='sigmoid', name='generation')(features)
    #aux = Dense(1, activation='sigmoid', name='auxiliary')(features)

    #discriminator = Model(evt_image, features)#fake) #Model(image, fake) #
    # inp = Input(shape=image.output_shape)
    m = Model(image, features)
    return m(image)
    # return features #discriminator

def multistream_discriminator(sizes):
    
    discr_inputs = [Input(shape=sizes[:2] + [1]), Input(shape=sizes[2:4] + [1]), Input(shape=sizes[4:] + [1])]
    #discr_inputs_middle = [Input(shape=sizes[:2] + [1]), Input(shape=sizes[2:4] + [1]), Input(shape=sizes[4:] + [1])]
    features = []
    # for image, image_middle in zip(discr_inputs, discr_inputs_middle):
    for image in discr_inputs:
        x = Conv2D(32, (2, 2), padding='same')(image)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)
        # block 2: 'same' bordered 3x3 locally connected block with batchnorm and
        # 2x2 subsampling
        x = ZeroPadding2D((1, 1))(x)
        x = LocallyConnected2D(8, (3, 3), padding='valid', strides=(1, 2))(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        # block 2: 'same' bordered 5x5 locally connected block with batchnorm
        x = ZeroPadding2D((1, 1))(x)
        x = LocallyConnected2D(8, (2, 2), padding='valid')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        # block 3: 2x2 locally connected block with batchnorm and
        # 1x2 subsampling
        x = ZeroPadding2D((1, 1))(x)
        x = LocallyConnected2D(8, (2, 2), padding='valid', strides=(1, 2))(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        h = Flatten()(x)

        dnn = Model(inputs=image, outputs=h)
        #evt_image = Input(shape=[int(a) for a in image.shape[1:]])#(img_shape[0], img_shape[1], 1))
        out = dnn(image)
        # out = dnn.outputs[0]
        #out = h 


        # TO-DO: minibatch discrim not working in Keras 2!
        # # nb of features to obtain
        # nb_features = 20
        # # dim of kernel space
        # vspace_dim = 10

        # # creates the kernel space for the minibatch discrimination
        # K_x = Dense3D(nb_features, vspace_dim)(out)#(h)#(out)

        # minibatch_featurizer = Lambda(minibatch_discriminator,
        #                           output_shape=minibatch_output_shape)

        # # concat the minibatch features with the normal ones
        # features.append( 
        #     merge(
        #         [
        #             minibatch_featurizer(K_x),
        #             out #h
        #         ],
        #         mode='concat'
        #     )
        # )

    combined_output = Dense(1, activation='sigmoid', name='discr_output')(
        Dense(64, activation='relu')(
            Dense(128, activation='relu')(
                merge(out, mode='concat')))) #features, mode='concat'))))

    discriminator = Model(
        inputs=discr_inputs,
        outputs=combined_output)

    return discriminator


def old_discriminator():

    image = Input(shape=(3, 96, 1))

    # block 1: normal 2x2 conv,
    # *NO* batchnorm (recommendation from [arXiv/1511.06434])
    x = Conv2D(32, (2, 2), padding='same')(image)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)
    
    # block 2: 'same' bordered 3x3 locally connected block with batchnorm and
    # 2x2 subsampling
    x = ZeroPadding2D((1, 1))(x)
    x = LocallyConnected2D(8, (3, 3), padding='valid', subsample=(1, 2))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # block 2: 'same' bordered 5x5 locally connected block with batchnorm
    x = ZeroPadding2D((1, 1))(x)
    x = LocallyConnected2D(8, (2, 2), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # block 3: 2x2 locally connected block with batchnorm and
    # 1x2 subsampling
    x = ZeroPadding2D((1, 1))(x)
    x = LocallyConnected2D(8, (2, 2), padding='valid', subsample=(1, 2))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    #x = AveragePooling2D((2, 2))(x)
    h = Flatten()(x)

    # nb of features to obtain
    nb_features = 20

    # dim of kernel space
    vspace_dim = 10

    # creates the kernel space for the minibatch discrimination
    K_x = Dense3D(nb_features, vspace_dim)(h)

    minibatch_featurizer = Lambda(minibatch_discriminator,
                              output_shape=minibatch_output_shape)

    # concat the minibatch features with the normal ones
    features = merge([
            minibatch_featurizer(K_x),
            h
            ], mode='concat')

    # fake output tracks binary fake / not-fake, and the auxiliary requires
    # reconstruction of latent features, in this case, labels
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    #aux = Dense(1, activation='sigmoid', name='auxiliary')(features)

    discriminator = Model(image, fake)
    return discriminator
