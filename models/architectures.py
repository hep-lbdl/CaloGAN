from keras.layers import (Dense, Reshape, Conv2D, LeakyReLU, BatchNormalization,
                          LocallyConnected2D, Activation, ZeroPadding2D, Dropout,
                          Lambda, Flatten)

from keras.layers.merge import concatenate
import keras.backend as K

from ops import (minibatch_discriminator, minibatch_output_shape,
                 Dense3D, sparsity_level, sparsity_output_shape)


from keras.layers.merge import concatenate, multiply
from keras.initializers import constant

import numpy as np
# def image_softmax(x):
#     x = x / 0.8  # temperature
#     e = K.exp(x - K.max(x, axis=(1, 2, 3), keepdims=True))
#     s = K.sum(e, axis=(1, 2, 3), keepdims=True)
#     return e / s


def sparse_softmax(x):
    x = K.relu(x)
    e = K.exp(x - K.max(x, axis=(1, 2, 3), keepdims=True))
    s = K.sum(e, axis=(1, 2, 3), keepdims=True)
    return e / s
    # s = K.sum(x, axis=(1, 2, 3), keepdims=True)
    # return x / K.clip(s, K.epsilon(), None)


def determine_energy_distribution(z, E, nb_outputs=3):

    # h = concatenate([z, E, Lambda(lambda x: x[0] * x[1])([z, E])])
    h = concatenate([z, E, multiply([z, E])])

    h = Dense(512)(h)
    h = Activation('relu')(h)

    final = Dense(nb_outputs, activation='softmax')
    attn = final(h)

    W, b = final.get_weights()
    b = (np.array([10, 90, 5]) / 4).reshape(b.shape)
    final.set_weights([W, b])
    leak = Dense(1, activation='sigmoid', bias_initializer=constant(10.))(h)

    energy_deposited = multiply([E, leak])
    dist = multiply([attn, energy_deposited])

    return [
        # select the individual columns
        Lambda(lambda x: K.reshape(x[:, i], (-1, 1)))(dist)
        for i in xrange(nb_outputs)
    ]
    # return d


# def determine_energy_captured(z, E):

#     # h = concatenate([z, E, Lambda(lambda x: x[0] * x[1])([z, E])])
#     h = concatenate([z, E, multiply([z, E])])

#     h = Dense(512)(h)
#     h = LeakyReLU()(h)
#     h = Dense(128)(h)
#     h = LeakyReLU()(h)
#     attn = Dense(nb_outputs, activation='softmax')
#     return multiply([attn, E])


def build_generator(x, nb_rows, nb_cols):

    x = Dense((nb_rows + 2) * (nb_cols + 2) * 36)(x)
    x = Reshape((nb_rows + 2, nb_cols + 2, 36))(x)
    # x = LeakyReLU()(x)

    # x = Conv2D(16, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    # x = LeakyReLU()(x)
    # x = BatchNormalization()(x)

    x = Conv2D(16, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = LocallyConnected2D(6, (2, 2), kernel_initializer='he_uniform')(x)
    x = LeakyReLU()(x)
    # x = BatchNormalization()(x)

    x = LocallyConnected2D(1, (2, 2), use_bias=False,
                           kernel_initializer='glorot_normal')(x)

    return x
    # return Activation('relu')(x)


def build_discriminator(image, mbd=False, sparsity=False, sparsity_mbd=False):

    # x = Conv2D(64, (2, 2), padding='same')(image)
    # x = LeakyReLU()(x)
    # # x = Dropout(0.2)(x)

    x = Conv2D(64, (2, 2), padding='same')(image)
    x = LeakyReLU()(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)
    # block 2: 'same' bordered 3x3 locally connected block with batchnorm and
    # 2x2 subsampling
    x = ZeroPadding2D((1, 1))(x)
    x = LocallyConnected2D(16, (3, 3), padding='valid', strides=(1, 2))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)
    # block 2: 'same' bordered 5x5 locally connected block with batchnorm
    x = ZeroPadding2D((1, 1))(x)
    x = LocallyConnected2D(8, (2, 2), padding='valid')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)
    # block 3: 2x2 locally connected block with batchnorm and
    # 1x2 subsampling
    x = ZeroPadding2D((1, 1))(x)
    x = LocallyConnected2D(8, (2, 2), padding='valid', strides=(1, 2))(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.2)(x)
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
            sparsity_detector = Lambda(sparsity_level, sparsity_output_shape)
            empirical_sparsity = sparsity_detector(image)
            if sparsity:
                features.append(empirical_sparsity)
            if sparsity_mbd:
                K_sparsity = Dense3D(nb_features, vspace_dim)(empirical_sparsity)
                features.append(Activation('tanh')(minibatch_featurizer(K_sparsity)))

        return concatenate(features)
    else:
        return x
