#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle

import argparse
from six.moves import range
import sys
from itertools import izip

from h5py import File as HDF5File
import numpy as np
import pandas as pd
import keras.backend as K
from keras.layers import (Input, Dense, Reshape, Flatten, Lambda, merge,
                          Dropout, BatchNormalization, Embedding, Activation)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (UpSampling2D, Conv2D, ZeroPadding2D,
                                        AveragePooling2D)
from keras.layers.local import LocallyConnected2D
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model

K.set_image_dim_ordering('tf')

from ops import minibatch_discriminator, minibatch_output_shape, Dense3D


def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run CalGAN training.'
        'Sensible defaults come from [arXiv/1511.06434]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nb-epochs', action='store', type=int, default=50,
                        help='Number of epochs to train for.')
    parser.add_argument('--batch-size', action='store', type=int, default=100,
                        help='batch size per update')
    parser.add_argument('--latent-size', action='store', type=int, default=500,
                        help='size of random N(0, 1) latent space to sample')

    # Adam parameters suggested in [arXiv/1511.06434]
    parser.add_argument('--adam-lr', action='store', type=float, default=0.0002,
                        help='Adam learning rate')

    parser.add_argument('--adam-beta', action='store', type=float, default=0.5,
                        help='Adam beta_1 parameter')

    parser.add_argument('--dataset', action='store', type=str,
                        help='txt file')

    parser.add_argument('--prog-bar', action='store_true',
                        help='Whether or not to use a progress bar')

    parser.add_argument('--d-pfx', action='store',
                        default='params_discriminator_epoch_',
                        help='Default prefix for discriminator network weights')

    parser.add_argument('--g-pfx', action='store',
                        default='params_generator_epoch_',
                        help='Default prefix for generator network weights')

    return parser


if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()

    # delay the imports so running train.py -h doesn't take 50 years
    import keras.backend as K

    K.set_image_dim_ordering('tf')

    from keras.layers import Input
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.utils.generic_utils import Progbar
    from sklearn.model_selection import train_test_split

    # from generator import generator as build_generator
    # from discriminator import discriminator as build_discriminator

    # batch, latent size, and whether or not to be verbose with a progress bar
    nb_epochs = parse_args.nb_epochs
    batch_size = parse_args.batch_size
    latent_size = parse_args.latent_size
    verbose = parse_args.prog_bar

    adam_lr = parse_args.adam_lr
    adam_beta_1 = parse_args.adam_beta

    datafile = parse_args.dataset

    # -- read in data
    if '.txt' in datafile:
        d = pd.read_csv(datafile, delimiter=",", header=None, skiprows=1).values
        with open(datafile) as f:
            sizes = map(int, f.readline().strip().split(","))
        first, second, third = np.split(
            d,
            indices_or_sections=[sizes[0]*sizes[1], sizes[0]*sizes[1] + sizes[2]*sizes[3]],
            axis=1
        )
        # -- reshape to put them into unravelled, 2D image format
        first = np.expand_dims(first.reshape(-1, sizes[0], sizes[1]), -1)
        second = np.expand_dims(second.reshape(-1, sizes[2], sizes[3]), -1)
        third = np.expand_dims(third.reshape(-1, sizes[4], sizes[5]), -1)
    elif '.hdf5' in datafile:
        import h5py
        d = h5py.File(datafile, 'r')
        first = np.expand_dims(d['layer_0'][:10000], -1)
        second = np.expand_dims(d['layer_1'][:10000], -1)
        third = np.expand_dims(d['layer_2'][:10000], -1)
        sizes = [first.shape[1], first.shape[2], second.shape[1], second.shape[2], third.shape[1], third.shape[2]]
    else:
        raise IOError('The file must be either the usual .txt or .hdf5 format')


    # we don't really need validation data as it's a bit meaningless for GANs,
    # but since we have an auxiliary task, it can be helpful to debug mode
    # collapse to a particularly signal or background-like image
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
    from sklearn.utils import shuffle
    first, second, third = shuffle(first, second, third, random_state=0)

    # tensorflow ordering
    # X_train = np.expand_dims(X_train, axis=-1)
    # X_test = np.expand_dims(X_test, axis=-1)

    # nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    # scale the pT levels by 100 (help neural nets w/ dynamic range - they
    # need all the help they can get)
    first, second, third = [X.astype(np.float32) / 500 for X in [first, second, third]]
    #X_test = X_test.astype(np.float32) / 500

    # train_history = defaultdict(list)
    # test_history = defaultdict(list)

    ###################################
    # build the discriminator
    print('Building discriminator')
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

        # nb of features to obtain
        nb_features = 20
        # dim of kernel space
        vspace_dim = 10

        # creates the kernel space for the minibatch discrimination
#        K_x = Dense3D(nb_features, vspace_dim)(out)#(h)#(out)

#        minibatch_featurizer = Lambda(minibatch_discriminator,
#                                  output_shape=minibatch_output_shape)

        features.append(out)
        # concat the minibatch features with the normal ones
#        features.append( 
#            merge(
#                [
#                    minibatch_featurizer(K_x),
#                    out #h
#                ],
#                mode='concat'
#            )
#        )

    combined_output = Dense(1, activation='sigmoid', name='discr_output')(
        LeakyReLU()(
            Dense(64)(
                LeakyReLU()(
                    Dense(128)(
                        merge(features, mode='concat'))))))

    discriminator = Model(
        inputs=discr_inputs,
        outputs=combined_output)

    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )

    plot_model(discriminator,
               to_file='discriminator.png',
               show_shapes=True,
               show_layer_names=True)

    ###################################
    # build the generator
    print('Building generator')
    latent = Input(shape=(latent_size, ), name='z')

    def _pairwise(iterable):
        '''s -> (s0, s1), (s2, s3), (s4, s5), ...'''
        a = iter(iterable)
        return izip(a, a)

    outputs = []
    for img_shape in _pairwise(sizes):
        x = Dense((img_shape[0] + 2) * (img_shape[1] + 2) * 12)(latent)
        x = Reshape((img_shape[0] + 2, img_shape[1] + 2, 12))(x)
        # block 1: (None, 5, 98, 12) => (None, 5, 98, 8),
        x = Conv2D(8, (2, 2), padding='same', kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        # block 2: (None, 5, 98, 32) => (None, 4, 97, 6),
        #ZeroPadding2D((2, 2)),
        x = LocallyConnected2D(6, (2, 2), kernel_initializer='he_uniform')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        # block 3: (None, 4, 97, 6) => (None, 3, 96, 1),
        x = LocallyConnected2D(1, (2, 2), use_bias=False, kernel_initializer='glorot_normal')(x)
        y = Activation('relu')(x)
        outputs.append(y)

    generator = Model(inputs=latent, outputs=outputs)
    generator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )
    plot_model(generator,
               to_file='generator.png',
               show_shapes=True,
               show_layer_names=True)
    
    # load in previous training
    #generator.load_weights('./params_generator_epoch_099.hdf5')

    ###################################
    # build combined model
    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    # isfake = discriminator(gan_image)
    isfake = discriminator(outputs)
    combined = Model(
        input=latent,
        output=isfake,
        name='combined_model'
    )
    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy')

    # plot_model(discriminator,
    #            to_file='discriminator2.png',
    #            show_shapes=True,
    #            show_layer_names=True)

    plot_model(combined,
           to_file='combined.png',
           show_shapes=True,
           show_layer_names=True)

    discriminator.load_weights('./test_params_discriminator_epoch_049.hdf5')
    generator.load_weights('./test_params_generator_epoch_049.hdf5')
    # # train_history = defaultdict(list)
    # # test_history = defaultdict(list)

    ###################################
    # training procedure
    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(first.shape[0] / batch_size)
        if verbose:
            progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(nb_batches):
            if verbose:
                progress_bar.update(index)
            else:
                if index % 100 == 0:
                    print('processed {}/{} batches'.format(index + 1, nb_batches))

            # generate a new batch of noise
            noise = np.random.normal(0, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch_1 = first[index * batch_size:(index + 1) * batch_size]
            image_batch_2 = second[index * batch_size:(index + 1) * batch_size]
            image_batch_3 = third[index * batch_size:(index + 1) * batch_size]
            #label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c (note: we have a flat prior here, so
            # we can just sample randomly)
            #sampled_labels = np.random.randint(0, nb_classes, batch_size)

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(noise, verbose=0)
            # generated_images_2 = generator_2.predict(noise, verbose=0)
            # generated_images_3 = generator_3.predict(noise, verbose=0)

            # see if the discriminator can figure itself out...
            real_batch_loss = discriminator.train_on_batch(
                [image_batch_1, image_batch_2, image_batch_3],
                #bit_flip(np.ones(batch_size))
                np.ones(batch_size)
            )

            # note that a given batch should have either *only* real or *only* fake,
            # as we have both minibatch discrimination and batch normalization, both
            # of which rely on batch level stats
            fake_batch_loss = discriminator.train_on_batch(
                generated_images,
                #bit_flip(np.zeros(batch_size))
                np.zeros(batch_size) #????
            )

            # print(fake_batch_loss)
            # print(real_batch_loss)

            epoch_disc_loss.append((fake_batch_loss +  real_batch_loss) / 2)

            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, real} labels to say
            # real
            trick = np.ones(batch_size)

            gen_losses = []

            # we do this twice simply to match the number of batches per epoch used to
            # train the discriminator
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                #sampled_labels = np.random.randint(0, nb_classes, batch_size)

                gen_losses.append(combined.train_on_batch(
                    noise,
                    trick
                ))

            epoch_gen_loss.append(np.mean(gen_losses))

        print('=' * 60)
        print('    Generator loss: {}'.format(np.mean(epoch_gen_loss)))
        print('Discriminator loss: {}'.format(np.mean(epoch_disc_loss)))
        print('=' * 60)
        # print('\nTesting for epoch {}:'.format(epoch + 1))

        # # generate a new batch of noise
        # noise = np.random.normal(0, 1, (nb_test, latent_size))

        # # sample some labels from p_c and generate images from them
        # #sampled_labels = np.random.randint(0, nb_classes, nb_test)
        # generated_images = generator.predict(
        #     noise, verbose=False)

        # # X = np.concatenate((X_test, generated_images))
        # y = np.array([1] * nb_test + [0] * nb_test)
        # # aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # # see if the discriminator can figure itself out...
        # discriminator_test_loss = discriminator.evaluate(
        #     X, y, verbose=False, batch_size=batch_size)

        # discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # # make new noise
        # noise = np.random.normal(0, 1, (2 * nb_test, latent_size))
        # sampled_labels = np.random.randint(0, nb_classes, 2 * nb_test)

        # trick = np.ones(2 * nb_test)

        # generator_test_loss = combined.evaluate(
        #     [noise, sampled_labels.reshape((-1, 1))],
        #     [trick, sampled_labels], verbose=False, batch_size=batch_size)

        # generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # # generate an epoch report on performance. **NOTE** that these values
        # # don't mean a whole lot, but they can be helpful for diagnosing *serious*
        # # instabilities with the training
        # train_history['generator'].append(generator_train_loss)
        # train_history['discriminator'].append(discriminator_train_loss)

        # test_history['generator'].append(generator_test_loss)
        # test_history['discriminator'].append(discriminator_test_loss)

        # print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
        #     'component', *discriminator.metrics_names))
        # print('-' * 65)

        # ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        # print(ROW_FMT.format('generator (train)',
        #                      *train_history['generator'][-1]))
        # print(ROW_FMT.format('generator (test)',
        #                      *test_history['generator'][-1]))
        # print(ROW_FMT.format('discriminator (train)',
        #                      *train_history['discriminator'][-1]))
        # print(ROW_FMT.format('discriminator (test)',
        #                      *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights('{0}{1:03d}.hdf5'.format(parse_args.g_pfx, epoch),
                               overwrite=True)
        # generator_2.save_weights('{0}{1:03d}_b2.hdf5'.format(parse_args.g_pfx, epoch),
        #                        overwrite=True)
        # generator_3.save_weights('{0}{1:03d}_b3.hdf5'.format(parse_args.g_pfx, epoch),
        #                        overwrite=True)
        discriminator.save_weights('{0}{1:03d}.hdf5'.format(parse_args.d_pfx, epoch),
                                   overwrite=True)

    # pickle.dump({'train': train_history, 'test': test_history},
    #             open('acgan-history.pkl', 'wb'))
