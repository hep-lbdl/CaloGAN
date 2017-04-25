#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: train.py
description: training script for [arXiv/1701.05927]
author: Luke de Oliveira (lukedeoliveira@lbl.gov)
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

from h5py import File as HDF5File
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense


def bit_flip(x, prob=0.02):
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

    parser.add_argument('--layer', action='store', type=int, default=1,
                        help='Layer of the calorimeter to reproduce.', choices=[1, 2, 3])
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

    from painter import layer_0_generator as build_generator
    from painter import layer_0_discriminator as build_discriminator

    # batch, latent size, and whether or not to be verbose with a progress bar
    nb_epochs = parse_args.nb_epochs
    batch_size = parse_args.batch_size
    latent_size = parse_args.latent_size
    verbose = parse_args.prog_bar
    layer = parse_args.layer

    adam_lr = parse_args.adam_lr
    adam_beta_1 = parse_args.adam_beta

    datafile = parse_args.dataset

    if '.txt' in datafile:
        d = pd.read_csv(datafile, delimiter=",", header=None, skiprows=1).values
        with open(datafile) as f:
            sizes = map(int, f.readline().strip().split(","))
        first, second, third = np.split(
            d,
            indices_or_sections=[sizes[0] * sizes[1],
                                 sizes[0] * sizes[1] + sizes[2] * sizes[3]],
            axis=1
        )
        # -- reshape to put them into unravelled, 2D image format
        first = np.expand_dims(first.reshape(-1, sizes[0], sizes[1]), -1)
        second = np.expand_dims(second.reshape(-1, sizes[2], sizes[3]), -1)
        third = np.expand_dims(third.reshape(-1, sizes[4], sizes[5]), -1)
    elif '.hdf5' in datafile:
        import h5py
        d = h5py.File(datafile, 'r')
        first = np.expand_dims(d['layer_0'][:], -1)
        second = np.expand_dims(d['layer_1'][:], -1)
        third = np.expand_dims(d['layer_2'][:], -1)
        sizes = [first.shape[1], first.shape[2], second.shape[
            1], second.shape[2], third.shape[1], third.shape[2]]
    else:
        raise IOError('The file must be either the usual .txt or .hdf5 format')

    # we don't really need validation data as it's a bit meaningless for GANs,
    # but since we have an auxiliary task, it can be helpful to debug mode
    # collapse to a particularly signal or background-like image
    #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
    from sklearn.utils import shuffle
    if layer == 1:
        X = shuffle(first)
        image_shape = sizes[:2]
    elif layer == 2:
        X = shuffle(second)
        image_shape = sizes[2:4]
    else:
        X = shuffle(third)
        image_shape = sizes[4:]

    image_shape += (1, )

    # tensorflow ordering
    # X_train = np.expand_dims(X_train, axis=-1)
    # X_test = np.expand_dims(X_test, axis=-1)

    # nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    # scale the pT levels by 100 (help neural nets w/ dynamic range - they
    # need all the help they can get)
    X = X.astype(np.float32) / 500
    #X_test = X_test.astype(np.float32) / 500

    # train_history = defaultdict(list)
    # test_history = defaultdict(list)

    # build the discriminator
    print('Building discriminator')
    input_image = Input(shape=image_shape)

    featurizer = build_discriminator(image_shape)
    features = featurizer(input_image)

    disc_output = Dense(1, activation='sigmoid', name='generation')(features)

    discriminator = Model(inputs=input_image, outputs=disc_output)

    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )

    # load in previous training
    # generator.load_weights('./params_generator_epoch_099.hdf5')

    # disc_submodel = Model(inputs=d_in, outputs=primary_output)
    discriminator.trainable = False
    # disc_submodel.trainable = False

    # build the generator
    print('Building generator')
    #generator = build_generator(latent_size)
    # latent = Input(shape=(latent_size, ), name='z')

    # generator_model = build_generator(latent_size, image_shape)
    generator = build_generator(latent_size, image_shape)

    # generated_image = generator_model(latent)

    # generator = Model(latent, generated_image)

    generator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )

    # symbolic predict
    # gan_image = generator(latent)

    # we only want to be able to train generation for the combined model

    # isfake = discriminator(gan_image)

    combined_latent = Input(shape=(latent_size, ), name='combined_z')

    fake = discriminator(generator(combined_latent))
    # isfake = disc_submodel(gan_image)
    combined = Model(combined_latent, fake, name='combined_model')

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )

    # MOVED ABOVE:
    # datafile = parse_args.dataset

    # d = pd.read_csv(datafile, delimiter=",", header=None, skiprows=1).values
    # with open(datafile) as f:
    #     sizes = map(int, f.readline().strip().split(","))
    # first, second, third = np.split(
    #     d,
    #     indices_or_sections=[sizes[0]*sizes[1], sizes[0]*sizes[1] + sizes[2]*sizes[3]],
    #     axis=1
    # )

    # # -- reshape to put them into unravelled, 2D image format
    # first = np.expand_dims(first.reshape(-1, sizes[0], sizes[1]), -1)
    # # second = np.expand_dims(second.reshape(-1, sizes[2], sizes[3]), -1)
    # # third = np.expand_dims(third.reshape(-1, sizes[4], sizes[5]), -1)

    # # we don't really need validation data as it's a bit meaningless for GANs,
    # # but since we have an auxiliary task, it can be helpful to debug mode
    # # collapse to a particularly signal or background-like image
    # #X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
    # from sklearn.utils import shuffle
    # X = shuffle(first)

    # # tensorflow ordering
    # # X_train = np.expand_dims(X_train, axis=-1)
    # # X_test = np.expand_dims(X_test, axis=-1)

    # # nb_train, nb_test = X_train.shape[0], X_test.shape[0]

    # # scale the pT levels by 100 (help neural nets w/ dynamic range - they
    # # need all the help they can get)
    # X = X.astype(np.float32) / 500
    # #X_test = X_test.astype(np.float32) / 500

    # # train_history = defaultdict(list)
    # # test_history = defaultdict(list)

    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(X.shape[0] / batch_size)
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
            image_batch = X[index * batch_size:(index + 1) * batch_size]
            #label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # sample some labels from p_c (note: we have a flat prior here, so
            # we can just sample randomly)
            #sampled_labels = np.random.randint(0, nb_classes, batch_size)

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(noise, verbose=0)

            # see if the discriminator can figure itself out...
            real_batch_loss = discriminator.train_on_batch(
                image_batch, bit_flip(np.ones(batch_size))
            )

            # note that a given batch should have either *only* real or *only* fake,
            # as we have both minibatch discrimination and batch normalization, both
            # of which rely on batch level stats
            fake_batch_loss = discriminator.train_on_batch(
                generated_images,
                bit_flip(np.zeros(batch_size))
            )

            epoch_disc_loss.append((fake_batch_loss + real_batch_loss) / 2)

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
        discriminator.save_weights('{0}{1:03d}.hdf5'.format(parse_args.d_pfx, epoch),
                                   overwrite=True)

    # pickle.dump({'train': train_history, 'test': test_history},
    #             open('acgan-history.pkl', 'wb'))
