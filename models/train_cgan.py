#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: train_cgan.py
description: conditional GAN training script
author: Michela Paganini (michela.paganini@yale.edu)
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
import yaml

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
from keras.layers.merge import concatenate, multiply, add
# from keras.utils import plot_model
from keras.losses import mean_absolute_error as _mean_absolute_error
from ops import (minibatch_discriminator, minibatch_output_shape, Dense3D,
                 single_layer_energy, single_layer_energy_output_shape,
                 sparsity_level, sparsity_output_shape, calculate_energy,
                 energy_error, scale, inpainting_attention)

K.set_image_dim_ordering('tf')

from ops import minibatch_discriminator, minibatch_output_shape, Dense3D


def mean_absolute_error(c):
    def _(y_true, y_pred):
        return _mean_absolute_error(y_true, y_pred) * c
    return _


def bit_flip(x, prob=0.05):
    """ flips a int array's values with some probability """
    x = np.array(x)
    selection = np.random.uniform(0, 1, x.shape) < prob
    x[selection] = 1 * np.logical_not(x[selection])
    return x


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run CalGAN training. '
        'Sensible defaults come from [arXiv/1511.06434]',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nb-epochs', action='store', type=int, default=50,
                        help='Number of epochs to train for.')
    parser.add_argument('--batch-size', action='store', type=int, default=200,
                        help='batch size per update')
    parser.add_argument('--latent-size', action='store', type=int, default=500,
                        help='size of random N(0, 1) latent space to sample')

    # Adam parameters suggested in [arXiv/1511.06434]
    parser.add_argument('--adam-lr', action='store', type=float, default=0.0002,
                        help='Adam learning rate')

    parser.add_argument('--adam-beta', action='store', type=float, default=0.5,
                        help='Adam beta_1 parameter')

    parser.add_argument('--dataset', action='store', type=str,
                        help='yaml file with particles and hdf paths')

    parser.add_argument('--prog-bar', action='store_true',
                        help='Whether or not to use a progress bar')

    parser.add_argument('--in-paint', action='store_true',
                        help='Whether or not to inpaint')

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

    from architectures import build_generator, build_discriminator

    # batch, latent size, and whether or not to be verbose with a progress bar
    nb_epochs = parse_args.nb_epochs
    batch_size = parse_args.batch_size
    latent_size = parse_args.latent_size
    verbose = parse_args.prog_bar

    adam_lr = parse_args.adam_lr
    adam_beta_1 = parse_args.adam_beta

    yaml_file = parse_args.dataset

    # -- read in data
    with open(yaml_file, 'r') as stream:
        try:
            s = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    nb_classes = len(s.keys())
    print('{} particle types found: {}'.format(nb_classes, s.keys()))

    # for particle, datafile in s.iteritems():
    def _load_data(particle, datafile):

        import h5py
        d = h5py.File(datafile, 'r')
        first = np.expand_dims(d['layer_0'][:], -1)
        second = np.expand_dims(d['layer_1'][:], -1)
        third = np.expand_dims(d['layer_2'][:], -1)
        energy = d['energy'][:].reshape(-1, 1) * 1000  # convert to MeV
        sizes = [first.shape[1], first.shape[2], second.shape[
            1], second.shape[2], third.shape[1], third.shape[2]]
        y = [particle] * first.shape[0]

        return first, second, third, y, energy, sizes

    first, second, third, y, energy, sizes = [
        np.concatenate(t) for t in [
            a for a in zip(*[_load_data(p, f) for p, f in s.iteritems()])
        ]
    ]

    #first, second, third, _, energy, sizes = _load_data(None, parse_args.dataset)

    # TO-DO: check that all sizes match, so I could be taking any of them
    sizes = sizes[:6].tolist()

    ###################################
    # preprocessing

    # scale the pT levels by 100 (help neural nets w/ dynamic range - they
    # need all the help they can get)
    # first, second, third = [X.astype(np.float32) / 500 for X in [first, second, third]]
    first, second, third, energy = [
        (X.astype(np.float32) / 1000)[:100000]
        for X in [first, second, third, energy]
    ]
    y = y[:100000]

    # energy between 2 and 200 now

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)
    from sklearn.utils import shuffle
    first, second, third, y, energy = shuffle(
        first, second, third, y, energy, random_state=0)

    #from sklearn.utils import shuffle
    # first, second, third, energy = shuffle(
    #    first, second, third, energy, random_state=0)
    # we don't really need validation data as it's a bit meaningless for GANs,
    # but since we have an auxiliary task, it can be helpful to debug mode
    # collapse to a particularly signal or background-like image
    # first_train, first_test,\
    #     second_train, second_test,\
    #     third_train, third_test,\
    #     y_train, y_test = train_test_split(first, second, third, y, train_size=0.99)

    # nb_train, nb_test = y_train.shape[0], y_test.shape[0]

    nb_train = first.shape[0]

    ###################################
    # build the discriminator
    print('Building discriminator')

#    calorimeter = [Input(shape=(3, 96, 1)),
#                   Input(shape=(12, 12, 1)),
#                   Input(shape=(12, 6, 1))]

    calorimeter = [Input(shape=sizes[:2] + [1]),
                   Input(shape=sizes[2:4] + [1]),
                   Input(shape=sizes[4:] + [1])]

    input_energy = Input(shape=(1, ))

    features = []
    energies = []

    for l in xrange(3):
        # build features per
        features.append(build_discriminator(
            image=calorimeter[l],
            mbd=True,
            sparsity=True,
            sparsity_mbd=True
        ))

        energies.append(calculate_energy(calorimeter[l]))

    discriminator_inputs = calorimeter + [input_energy]

    # conditional gan
    if nb_classes > 1:
        input_class = Input(shape=(1, ), dtype='int32')
        class_embedding = Flatten()(Embedding(
            nb_classes, 50, input_length=1, embeddings_initializer='glorot_normal')(input_class))
        features = concatenate(features + [class_embedding])
        discriminator_inputs.append(input_class)
    else:
        features = concatenate(features)

    energies = concatenate(energies)

    # calculate the total energy across all rows
    total_energy = Lambda(
        lambda x: K.reshape(K.sum(x, axis=-1), (-1, 1)),
        name='total_energy'
    )(energies)

    nb_features = 10
    vspace_dim = 10
    minibatch_featurizer = Lambda(minibatch_discriminator,
                                  output_shape=minibatch_output_shape)
    K_energy = Dense3D(nb_features, vspace_dim)(energies)
    mbd_energy = Activation('tanh')(minibatch_featurizer(K_energy))

    # # binary y/n if it is over the input energy
    energy_well = Lambda(
        lambda x: K.abs(x[0] - x[1])
    )([total_energy, input_energy])

    well_too_big = Lambda(lambda x: 10 * K.cast(x > 5, K.floatx()))(energy_well)

    # p = concatenate([features, energies, total_energy])
    p = concatenate([
        features,
        scale(energies, 10),
        scale(total_energy, 100),
        # scale(input_energy, 100),
        energy_well,
        well_too_big,
        mbd_energy
    ])

    fake = Dense(1, activation='sigmoid', name='fakereal_output')(p)
    discriminator_outputs = [fake, total_energy]
    discriminator_losses = ['binary_crossentropy', mean_absolute_error(1)]

    # if nb_classes > 1:  # acgan
    #aux = Dense(1, activation='sigmoid', name='auxiliary_output')(p)
    # discriminator_outputs.append(aux)

    # if nb_classes > 2:
    #    discriminator_losses.append('sparse_categorical_crossentropy')
    # else:
    #    discriminator_losses.append('binary_crossentropy')

    # fake = Dense(1, activation='sigmoid', name='fakereal_output')(p)

    # discriminator = Model(
    #     inputs=discr_inputs,
    #     outputs=[fake, aux])

    # discriminator = Model(calorimeter, [fake, total_energy])
    # discriminator = Model(calorimeter, fake)

    # discriminator = Model(calorimeter + [input_energy], fake)
    discriminator = Model(discriminator_inputs, discriminator_outputs)

    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=discriminator_losses
        # loss=['binary_crossentropy', energy_error]
    )

    # if nb_classes == 2:
    #     aux_loss = 'binary_crossentropy'
    # else:
    #     aux_loss = 'sparse_categorical_crossentropy'

    # discriminator.compile(
    #     optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
    #     loss={
    #         'fakereal_output': 'binary_crossentropy',
    #         'auxiliary_output': aux_loss
    #     }
    # )

    # plot_model(discriminator,
    #           to_file='discriminator.png',
    #           show_shapes=True,
    #           show_layer_names=True)

    ###################################
    # build the generator
    print('Building generator')
    latent = Input(shape=(latent_size, ), name='z')
    input_energy = Input(shape=(1, ), dtype='float32')
    generator_inputs = [latent, input_energy]

    def _pairwise(iterable):
        '''s -> (s0, s1), (s2, s3), (s4, s5), ...'''
        a = iter(iterable)
        return izip(a, a)

    output_layers = []
    if nb_classes > 1:  # acgan
        image_class = Input(shape=(1, ), dtype='int32')  # label
        # not the same as the discriminator
        emb = Flatten()(Embedding(nb_classes, latent_size, input_length=1,
                                  embeddings_initializer='glorot_normal')(image_class))
        # hadamard product between z-space and a class conditional embedding
        hc = merge([latent, emb], mode='mul')
        h = Lambda(lambda x: x[0] * x[1])([hc, scale(input_energy, 100)])
        generator_inputs.append(image_class)
    else:
        h = Lambda(lambda x: x[0] * x[1])([latent, scale(input_energy, 100)])

    # h = concatenate([latent, input_energy])

    # emb = Flatten()(Embedding(nb_classes, latent_size, input_length=1,
    #                           embeddings_initializer='glorot_normal')(image_class))
    # # hadamard product between z-space and a class conditional embedding
    # h = merge([latent, emb], mode='mul')

    img_layer0 = build_generator(h, 3, 96)
    img_layer1 = build_generator(h, 12, 12)
    img_layer2 = build_generator(h, 12, 6)

    if parse_args.in_paint:

        # inpainting
        zero2one = AveragePooling2D(pool_size=(1, 8))(
            UpSampling2D(size=(4, 1))(img_layer0))
        # final_img_layer1 = add([zero2one, img_layer1])
        img_layer1 = inpainting_attention(img_layer1, zero2one)

        one2two = AveragePooling2D(pool_size=(1, 2))(img_layer1)
        # final_img_layer2 = add([one2two, img_layer2])
        img_layer2 = inpainting_attention(img_layer2, one2two)

    generator_outputs = [
        Activation('relu')(img_layer0),
        Activation('relu')(img_layer1),
        Activation('relu')(img_layer2)
    ]

    generator = Model(generator_inputs, generator_outputs)

    generator.compile(
        optimizer=Adam(lr=10 * adam_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )
    # plot_model(
    #    generator,
    #    to_file='generator.png',
    #    show_shapes=True,
    #    show_layer_names=True
    # )

    # load in previous training
    # generator.load_weights('./params_generator_epoch_099.hdf5')

    ###################################
    # build combined model
    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    # isfake = discriminator(gan_image)

    # by_layer_energy = Input(shape=(3, ))
    # isfake = discriminator(generator([latent, input_energy]))
    # isfake, aux_energy = discriminator(generator([latent, input_energy]))
    # isfake, aux_energy = discriminator(generator(generator_inputs) + [input_energy])
    temp_inputs = generator(generator_inputs) + [input_energy]
    combined_inputs = generator_inputs  # from D
    if nb_classes > 1:
        temp_inputs.append(image_class)
        combined_inputs.append(input_class)

    combined_outputs = discriminator(temp_inputs)
    combined = Model(
        inputs=combined_inputs,  # generator_inputs,
        # outputs=[isfake, ,
        outputs=combined_outputs,
        name='combined_model'
    )
    combined.compile(
        optimizer=Adam(lr=10 * adam_lr, beta_1=adam_beta_1),
        loss=discriminator_losses
        # loss=['binary_crossentropy', energy_error]

    )

   # plot_model(combined,
   #            to_file='combined.png',
    #           show_shapes=True,
    #          show_layer_names=True)

#    discriminator.load_weights('./params_discriminator_epoch_000.hdf5')
#    generator.load_weights('./params_generator_epoch_000.hdf5')

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
            # image_batch_1 = first_train[index * batch_size:(index + 1) * batch_size]
            # image_batch_2 = second_train[index * batch_size:(index + 1) * batch_size]
            # image_batch_3 = third_train[index * batch_size:(index + 1) * batch_size]
            # label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # get a batch of real images
            image_batch_1 = first[index * batch_size:(index + 1) * batch_size]
            image_batch_2 = second[index * batch_size:(index + 1) * batch_size]
            image_batch_3 = third[index * batch_size:(index + 1) * batch_size]
            label_batch = y[index * batch_size:(index + 1) * batch_size]
            energy_batch = energy[index * batch_size:(index + 1) * batch_size]

            # energy_breakdown

            # sample some labels from p_c (note: we have a flat prior here, so
            # we can just sample randomly)
            sampled_labels = np.random.randint(0, nb_classes, batch_size)
            sampled_energies = np.random.uniform(1, 100, (batch_size, 1))

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generator_inputs = [noise, sampled_energies]
            if nb_classes > 1:
                generator_inputs.append(sampled_labels)
            generated_images = generator.predict(generator_inputs, verbose=0)
            # [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            # see if the discriminator can figure itself out...
            discriminator_inputs_real = [image_batch_1,
                                         image_batch_2, image_batch_3, energy_batch]
            discriminator_inputs_fake = generated_images + [sampled_energies]
            discriminator_outputs_real = [np.ones(batch_size), energy_batch]
            discriminator_outputs_fake = [np.zeros(batch_size), sampled_energies]
            loss_weights = [np.ones(batch_size), 0.05 * np.ones(batch_size)]
            if nb_classes > 1:  # cgan
                discriminator_inputs_real.append(label_batch)
                discriminator_inputs_fake.append(sampled_labels)
            # if nb_classes > 1: # acgan
            #    discriminator_outputs_real.append(label_batch)
            #    discriminator_outputs_fake.append(bit_flip(sampled_labels, 0.3))
            #    loss_weights.append(0.2 * np.ones(batch_size))

            real_batch_loss = discriminator.train_on_batch(
                discriminator_inputs_real,
                discriminator_outputs_real,
                loss_weights
                # np.ones(batch_size)
                #[np.ones(batch_size), energy_batch]
                # [np.ones(batch_size), 0.25 * np.ones(batch_size)]  # weights
            )

            # note that a given batch should have either *only* real or *only* fake,
            # as we have both minibatch discrimination and batch normalization, both
            # of which rely on batch level stats
            fake_batch_loss = discriminator.train_on_batch(
                discriminator_inputs_fake,
                discriminator_outputs_fake,
                loss_weights
                # np.zeros(batch_size)
                #[np.zeros(batch_size), sampled_energies],
                # [np.ones(batch_size), 0.25 * np.ones(batch_size)]  # weights
            )

            # print(fake_batch_loss)
            # print(real_batch_loss)

            epoch_disc_loss.append(
                (np.array(fake_batch_loss) + np.array(real_batch_loss)) / 2)

            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, real} labels to say
            # real
            trick = np.ones(batch_size)

            gen_losses = []

            # we do this twice simply to match the number of batches per epoch used to
            # train the discriminator
            for _ in range(2):
                noise = np.random.normal(0, 1, (batch_size, latent_size))
                sampled_labels = np.random.randint(0, nb_classes, batch_size)
                sampled_energies = np.random.uniform(1, 100, (batch_size, 1))
                combined_inputs = [noise, sampled_energies]
                combined_outputs = [trick, sampled_energies]
                if nb_classes > 1:
                    combined_inputs.append(sampled_labels)
                    combined_inputs.append(sampled_labels)  # twice!
                    # combined_outputs.append(sampled_labels)
                gen_losses.append(combined.train_on_batch(
                    combined_inputs,
                    combined_outputs,
                    loss_weights
                    # [noise, sampled_labels.reshape(-1, 1)],
                    #[noise, sampled_energies],
                    #[trick, sampled_energies],
                    # trick,
                    # [trick, bit_flip(sampled_labels.reshape(-1, 1), 0.1)],
                    # [np.ones(len(trick)), 0.25 * np.ones(len(trick))]  # weights
                ))

            epoch_gen_loss.append(np.mean(np.array(gen_losses), axis=0))

        print('=' * 60)
        print('    Generator loss: {}'.format(np.mean(epoch_gen_loss, axis=0)))
        print('Discriminator loss: {}'.format(np.mean(epoch_disc_loss, axis=0)))
        print('=' * 60)

        # print('\nTesting for epoch {}:'.format(epoch + 1))

        # # generate a new batch of noise
        # noise = np.random.normal(0, 1, (nb_test, latent_size))

        # # sample some labels from p_c and generate images from them
        # sampled_labels = np.random.randint(0, nb_classes, nb_test)
        # generated_images = generator.predict(
        #     noise, verbose=False)

        # X = np.concatenate((X_test, generated_images))
        # y = np.array([1] * nb_test + [0] * nb_test)
        # aux_y = np.concatenate((y_test, sampled_labels), axis=0)

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
