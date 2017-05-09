#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
file: train.py
description: main training script for [arXiv/1705.02355]
author: Luke de Oliveira (lukedeo@manifold.ai), 
        Michela Paganini (michela.paganini@yale.edu)
"""

from __future__ import print_function

import argparse
from collections import defaultdict
import logging


import numpy as np
import os
from six.moves import range
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import sys
import yaml


if __name__ == '__main__':
    logger = logging.getLogger(
        '%s.%s' % (
            __package__, os.path.splitext(os.path.split(__file__)[-1])[0]
        )
    )
    logger.setLevel(logging.INFO)
else:
    logger = logging.getLogger(__name__)


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

    parser.add_argument('--batch-size', action='store', type=int, default=256,
                        help='batch size per update')

    parser.add_argument('--latent-size', action='store', type=int, default=1024,
                        help='size of random N(0, 1) latent space to sample')

    parser.add_argument('--disc-lr', action='store', type=float, default=2e-5,
                        help='Adam learning rate for discriminator')

    parser.add_argument('--gen-lr', action='store', type=float, default=2e-4,
                        help='Adam learning rate for generator')

    parser.add_argument('--adam-beta', action='store', type=float, default=0.5,
                        help='Adam beta_1 parameter')

    parser.add_argument('--prog-bar', action='store_true',
                        help='Whether or not to use a progress bar')

    parser.add_argument('--no-attn', action='store_true',
                        help='Whether to turn off the layer to layer attn.')

    parser.add_argument('--debug', action='store_true',
                        help='Whether to run debug level logging')

    parser.add_argument('--d-pfx', action='store',
                        default='params_discriminator_epoch_',
                        help='Default prefix for discriminator network weights')

    parser.add_argument('--g-pfx', action='store',
                        default='params_generator_epoch_',
                        help='Default prefix for generator network weights')

    parser.add_argument('dataset', action='store', type=str,
                        help='yaml file with particles and HDF5 paths (see '
                        'github.com/hep-lbdl/CaloGAN/blob/master/models/'
                        'particles.yaml)')

    return parser


if __name__ == '__main__':

    parser = get_parser()
    parse_args = parser.parse_args()

    # delay the imports so running train.py -h doesn't take 5,234,807 years
    import keras.backend as K
    from keras.layers import (Activation, AveragePooling2D, Dense, Embedding,
                              Flatten, Input, Lambda, UpSampling2D)
    from keras.layers.merge import add, concatenate, multiply
    from keras.models import Model
    from keras.optimizers import Adam
    from keras.utils.generic_utils import Progbar

    K.set_image_dim_ordering('tf')

    from ops import (minibatch_discriminator, minibatch_output_shape, Dense3D,
                     calculate_energy, scale, inpainting_attention)

    from architectures import build_generator, build_discriminator

    # batch, latent size, and whether or not to be verbose with a progress bar

    if parse_args.debug:
        logger.setLevel(logging.DEBUG)

    # set up all the logging stuff
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s'
        '[%(levelname)s]: %(message)s'
    )
    hander = logging.StreamHandler(sys.stdout)
    hander.setFormatter(formatter)
    logger.addHandler(hander)

    nb_epochs = parse_args.nb_epochs
    batch_size = parse_args.batch_size
    latent_size = parse_args.latent_size
    verbose = parse_args.prog_bar
    no_attn = parse_args.no_attn

    disc_lr = parse_args.disc_lr
    gen_lr = parse_args.gen_lr
    adam_beta_1 = parse_args.adam_beta

    yaml_file = parse_args.dataset

    logger.debug('parameter configuration:')

    logger.debug('number of epochs = {}'.format(nb_epochs))
    logger.debug('batch size = {}'.format(batch_size))
    logger.debug('latent size = {}'.format(latent_size))
    logger.debug('progress bar enabled = {}'.format(verbose))
    logger.debug('Using attention = {}'.format(no_attn == False))
    logger.debug('discriminator learning rate = {}'.format(disc_lr))
    logger.debug('generator learning rate = {}'.format(gen_lr))
    logger.debug('Adam $\beta_1$ parameter = {}'.format(adam_beta_1))
    logger.debug('Will read YAML spec from {}'.format(yaml_file))

    # read in data file spec from YAML file
    with open(yaml_file, 'r') as stream:
        try:
            s = yaml.load(stream)
        except yaml.YAMLError as exc:
            logger.error(exc)
            raise exc
    nb_classes = len(s.keys())
    logger.info('{} particle types found.'.format(nb_classes))
    for name, pth in s.iteritems():
        logger.debug('class {} <= {}'.format(name, pth))

    def _load_data(particle, datafile):

        import h5py

        d = h5py.File(datafile, 'r')

        # make our calo images channels-last
        first = np.expand_dims(d['layer_0'][:], -1)
        second = np.expand_dims(d['layer_1'][:], -1)
        third = np.expand_dims(d['layer_2'][:], -1)
        # convert to MeV
        energy = d['energy'][:].reshape(-1, 1) * 1000

        sizes = [
            first.shape[1], first.shape[2],
            second.shape[1], second.shape[2],
            third.shape[1], third.shape[2]
        ]

        y = [particle] * first.shape[0]

        d.close()

        return first, second, third, y, energy, sizes

    logger.debug('loading data from {} files'.format(nb_classes))

    first, second, third, y, energy, sizes = [
        np.concatenate(t) for t in [
            a for a in zip(*[_load_data(p, f) for p, f in s.iteritems()])
        ]
    ]

    # TO-DO: check that all sizes match, so I could be taking any of them
    sizes = sizes[:6].tolist()

    # scale the energy depositions by 1000 to convert MeV => GeV
    first, second, third, energy = [
        (X.astype(np.float32) / 1000)
        for X in [first, second, third, energy]
    ]

    le = LabelEncoder()
    y = le.fit_transform(y)

    first, second, third, y, energy = shuffle(first, second, third, y, energy,
                                              random_state=0)

    logger.info('Building discriminator')

    calorimeter = [Input(shape=sizes[:2] + [1]),
                   Input(shape=sizes[2:4] + [1]),
                   Input(shape=sizes[4:] + [1])]

    input_energy = Input(shape=(1, ))

    features = []
    energies = []

    for l in range(3):
        # build features per layer of calorimeter
        features.append(build_discriminator(
            image=calorimeter[l],
            mbd=True,
            sparsity=True,
            sparsity_mbd=True
        ))

        energies.append(calculate_energy(calorimeter[l]))

    features = concatenate(features)

    # This is a (None, 3) tensor with the individual energy per layer
    energies = concatenate(energies)

    # calculate the total energy across all rows
    total_energy = Lambda(
        lambda x: K.reshape(K.sum(x, axis=-1), (-1, 1)),
        name='total_energy'
    )(energies)

    # construct MBD on the raw energies
    nb_features = 10
    vspace_dim = 10
    minibatch_featurizer = Lambda(minibatch_discriminator,
                                  output_shape=minibatch_output_shape)
    K_energy = Dense3D(nb_features, vspace_dim)(energies)

    # constrain w/ a tanh to dampen the unbounded nature of energy-space
    mbd_energy = Activation('tanh')(minibatch_featurizer(K_energy))

    # absolute deviation away from input energy. Technically we can learn
    # this, but since we want to get as close as possible to conservation of
    # energy, just coding it in is better
    energy_well = Lambda(
        lambda x: K.abs(x[0] - x[1])
    )([total_energy, input_energy])

    # binary y/n if it is over the input energy
    well_too_big = Lambda(lambda x: 10 * K.cast(x > 5, K.floatx()))(energy_well)

    p = concatenate([
        features,
        scale(energies, 10),
        scale(total_energy, 100),
        energy_well,
        well_too_big,
        mbd_energy
    ])

    fake = Dense(1, activation='sigmoid', name='fakereal_output')(p)
    discriminator_outputs = [fake, total_energy]
    discriminator_losses = ['binary_crossentropy', 'mae']
    # ACGAN case
    if nb_classes > 1:
        logger.info('running in ACGAN for discriminator mode since found {} '
                    'classes'.format(nb_classes))

        aux = Dense(1, activation='sigmoid', name='auxiliary_output')(p)
        discriminator_outputs.append(aux)

        # change the loss depending on how many outputs on the auxiliary task
        if nb_classes > 2:
            discriminator_losses.append('sparse_categorical_crossentropy')
        else:
            discriminator_losses.append('binary_crossentropy')

    discriminator = Model(calorimeter + [input_energy], discriminator_outputs)

    discriminator.compile(
        optimizer=Adam(lr=disc_lr, beta_1=adam_beta_1),
        loss=discriminator_losses
    )

    logger.info('Building generator')

    latent = Input(shape=(latent_size, ), name='z')
    input_energy = Input(shape=(1, ), dtype='float32')
    generator_inputs = [latent, input_energy]

    # ACGAN case
    if nb_classes > 1:
        logger.info('running in ACGAN for generator mode since found {} '
                    'classes'.format(nb_classes))

        # label of requested class
        image_class = Input(shape=(1, ), dtype='int32')
        lookup_table = Embedding(nb_classes, latent_size, input_length=1,
                                 embeddings_initializer='glorot_normal')
        emb = Flatten()(lookup_table(image_class))

        # hadamard product between z-space and a class conditional embedding
        hc = multiply([latent, emb])

        # requested energy comes in GeV
        h = Lambda(lambda x: x[0] * x[1])([hc, scale(input_energy, 100)])
        generator_inputs.append(image_class)
    else:
        # requested energy comes in GeV
        h = Lambda(lambda x: x[0] * x[1])([latent, scale(input_energy, 100)])

    # each of these builds a LAGAN-inspired [arXiv/1701.05927] component with
    # linear last layer
    img_layer0 = build_generator(h, 3, 96)
    img_layer1 = build_generator(h, 12, 12)
    img_layer2 = build_generator(h, 12, 6)

    if not no_attn:

        logger.info('using attentional mechanism')

        # resizes from (3, 96) => (12, 12)
        zero2one = AveragePooling2D(pool_size=(1, 8))(
            UpSampling2D(size=(4, 1))(img_layer0))
        img_layer1 = inpainting_attention(img_layer1, zero2one)

        # resizes from (12, 12) => (12, 6)
        one2two = AveragePooling2D(pool_size=(1, 2))(img_layer1)
        img_layer2 = inpainting_attention(img_layer2, one2two)

    generator_outputs = [
        Activation('relu')(img_layer0),
        Activation('relu')(img_layer1),
        Activation('relu')(img_layer2)
    ]

    generator = Model(generator_inputs, generator_outputs)

    generator.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss='binary_crossentropy'
    )

    discriminator.trainable = False

    combined_outputs = discriminator(
        generator(generator_inputs) + [input_energy]
    )

    combined = Model(generator_inputs, combined_outputs, name='combined_model')
    combined.compile(
        optimizer=Adam(lr=gen_lr, beta_1=adam_beta_1),
        loss=discriminator_losses
    )

    logger.info('commencing training')

    for epoch in range(nb_epochs):
        logger.info('Epoch {} of {}'.format(epoch + 1, nb_epochs))

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
                    logger.info('processed {}/{} batches'.format(index + 1, nb_batches))
                elif index % 10 == 0:
                    logger.debug('processed {}/{} batches'.format(index + 1, nb_batches))

            # generate a new batch of noise
            noise = np.random.normal(0, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch_1 = first[index * batch_size:(index + 1) * batch_size]
            image_batch_2 = second[index * batch_size:(index + 1) * batch_size]
            image_batch_3 = third[index * batch_size:(index + 1) * batch_size]
            label_batch = y[index * batch_size:(index + 1) * batch_size]
            energy_batch = energy[index * batch_size:(index + 1) * batch_size]

            # energy_breakdown

            sampled_labels = np.random.randint(0, nb_classes, batch_size)
            sampled_energies = np.random.uniform(1, 100, (batch_size, 1))

            generator_inputs = [noise, sampled_energies]
            if nb_classes > 1:
                # in the case of the ACGAN, we need to append the requested
                # class to the pre-image of the generator
                generator_inputs.append(sampled_labels)

            generated_images = generator.predict(generator_inputs, verbose=0)

            disc_outputs_real = [np.ones(batch_size), energy_batch]
            disc_outputs_fake = [np.zeros(batch_size), sampled_energies]

            # downweight the energy reconstruction loss ($\lambda_E$ in paper)
            loss_weights = [np.ones(batch_size), 0.05 * np.ones(batch_size)]
            if nb_classes > 1:
                # in the case of the ACGAN, we need to append the realrequested
                # class to the target
                disc_outputs_real.append(label_batch)
                disc_outputs_fake.append(bit_flip(sampled_labels, 0.3))
                loss_weights.append(0.2 * np.ones(batch_size))

            real_batch_loss = discriminator.train_on_batch(
                [image_batch_1, image_batch_2, image_batch_3, energy_batch],
                disc_outputs_real,
                loss_weights
            )

            # note that a given batch should have either *only* real or *only* fake,
            # as we have both minibatch discrimination and batch normalization, both
            # of which rely on batch level stats
            fake_batch_loss = discriminator.train_on_batch(
                generated_images + [sampled_energies],
                disc_outputs_fake,
                loss_weights
            )

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

                sampled_energies = np.random.uniform(1, 100, (batch_size, 1))
                combined_inputs = [noise, sampled_energies]
                combined_outputs = [trick, sampled_energies]
                if nb_classes > 1:
                    sampled_labels = np.random.randint(0, nb_classes,
                                                       batch_size)
                    combined_inputs.append(sampled_labels)
                    combined_outputs.append(sampled_labels)

                gen_losses.append(combined.train_on_batch(
                    combined_inputs,
                    combined_outputs,
                    loss_weights
                ))

            epoch_gen_loss.append(np.mean(np.array(gen_losses), axis=0))

        logger.info('Epoch {:3d} Generator loss: {}'.format(
            epoch + 1, np.mean(epoch_gen_loss, axis=0)))
        logger.info('Epoch {:3d} Discriminator loss: {}'.format(
            epoch + 1, np.mean(epoch_disc_loss, axis=0)))

        # save weights every epoch
        generator.save_weights('{0}{1:03d}.hdf5'.format(parse_args.g_pfx, epoch),
                               overwrite=True)

        discriminator.save_weights('{0}{1:03d}.hdf5'.format(parse_args.d_pfx, epoch),
                                   overwrite=True)
