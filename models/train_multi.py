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

from h5py import File as HDF5File
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, merge


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

    from generator import generator as build_generator
    from discriminator import discriminator as build_discriminator

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
        first = np.expand_dims(d['layer_0'][:1000], -1)
        second = np.expand_dims(d['layer_1'][:1000], -1)
        third = np.expand_dims(d['layer_2'][:1000], -1)
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


    # build the discriminator
    print('Building discriminator')
    d_in_1 = Input(shape=sizes[:2] + [1])
    d_in_2 = Input(shape=sizes[2:4] + [1])
    d_in_3 = Input(shape=sizes[4:] + [1])
    features_1 = build_discriminator(d_in_1)
    features_2 = build_discriminator(d_in_2)
    features_3 = build_discriminator(d_in_3)
    combined_output = Dense(1, activation='sigmoid', name='discr_output')(
        merge([features_1, features_2, features_3], mode='concat'))

    discriminator = Model(
        inputs=[d_in_1, d_in_2, d_in_3],
        outputs=combined_output)

    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )

    # build the generator
    print('Building generator')
    latent = Input(shape=(latent_size, ), name='z')
    #generator = build_generator(latent_size)
    gan_image_1 = build_generator(latent, sizes[:2])
    generator_1 = Model(latent, gan_image_1)
    generator_1.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )
    gan_image_2 = build_generator(latent, sizes[2:4])
    generator_2 = Model(latent, gan_image_2)
    generator_2.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )
    gan_image_3 = build_generator(latent, sizes[4:])
    generator_3 = Model(latent, gan_image_3)
    generator_3.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )

    # load in previous training
    #generator.load_weights('./params_generator_epoch_099.hdf5')

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    # isfake = discriminator(gan_image)
    isfake = discriminator([gan_image_1, gan_image_2, gan_image_3])
    combined = Model(
        input=latent,
        output=isfake,
        name='combined_model'
    )

    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy')

    # generator_1.load_weights('../weights/params_generator_init_layer1.hdf5')
    # generator_2.load_weights('../weights/params_generator_init_layer2.hdf5')
    # generator_3.load_weights('../weights/params_generator_init_layer3.hdf5')
    # discriminator.load_weights('../weights/params_discriminator_epoch_037.hdf5')
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
            generated_images_1 = generator_1.predict(noise, verbose=0)
            generated_images_2 = generator_2.predict(noise, verbose=0)
            generated_images_3 = generator_3.predict(noise, verbose=0)

            # see if the discriminator can figure itself out...
            real_batch_loss = discriminator.train_on_batch(
                [image_batch_1, image_batch_2, image_batch_3],
                bit_flip(np.ones(batch_size))
            )

            # note that a given batch should have either *only* real or *only* fake,
            # as we have both minibatch discrimination and batch normalization, both
            # of which rely on batch level stats
            fake_batch_loss = discriminator.train_on_batch(
                [generated_images_1, generated_images_2, generated_images_3],
                bit_flip(np.zeros(batch_size))
            )

            # TODO: why are there 5 numbers if there are only 4 outputs???
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
        generator_1.save_weights('{0}{1:03d}_b1.hdf5'.format(parse_args.g_pfx, epoch),
                               overwrite=True)
        generator_2.save_weights('{0}{1:03d}_b2.hdf5'.format(parse_args.g_pfx, epoch),
                               overwrite=True)
        generator_3.save_weights('{0}{1:03d}_b3.hdf5'.format(parse_args.g_pfx, epoch),
                               overwrite=True)
        discriminator.save_weights('{0}{1:03d}.hdf5'.format(parse_args.d_pfx, epoch),
                                   overwrite=True)

    # pickle.dump({'train': train_history, 'test': test_history},
    #             open('acgan-history.pkl', 'wb'))
