#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: ops.py
description: ancillary ops for [arXiv/1705.02355] 
    (borrowing from [arXiv/1701.05927])
author: Luke de Oliveira (lukedeo@manifold.ai)
"""

import keras.backend as K
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints, activations
from keras.layers import Lambda, ZeroPadding2D, LocallyConnected2D
from keras.layers.merge import concatenate, multiply

import numpy as np


def channel_softmax(x):
    e = K.exp(x - K.max(x, axis=-1, keepdims=True))
    s = K.sum(e, axis=-1, keepdims=True)
    return e / s


def scale(x, v):
    return Lambda(lambda _: _ / v)(x)


def inpainting_attention(primary, carryover, constant=-10):

    def _initialize_bias(const=-5):
        def _(shape, dtype=None):
            assert len(shape) == 3, 'must be a 3D shape'
            x = np.zeros(shape)
            x[:, :, -1] = const
            return x
        return _

    x = concatenate([primary, carryover], axis=-1)
    h = ZeroPadding2D((1, 1))(x)
    lcn = LocallyConnected2D(
        filters=2,
        kernel_size=(3, 3),
        bias_initializer=_initialize_bias(constant)
    )

    h = lcn(h)
    weights = Lambda(channel_softmax)(h)

    channel_sum = Lambda(K.sum, arguments={'axis': -1, 'keepdims': True})

    return channel_sum(multiply([x, weights]))


def energy_error(requested_energy, recieved_energy):
    difference = (recieved_energy - requested_energy) / 10000

    over_energized = K.cast(difference > 0., K.floatx())

    too_high = 100 * K.abs(difference)
    too_low = 10 * K.abs(difference)

    return over_energized * too_high + (1 - over_energized) * too_low


def minibatch_discriminator(x):
    """ Computes minibatch discrimination features from input tensor x"""
    diffs = K.expand_dims(x, 3) - \
        K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    l1_norm = K.sum(K.abs(diffs), axis=2)
    return K.sum(K.exp(-l1_norm), axis=2)


def minibatch_output_shape(input_shape):
    """ Computes output shape for a minibatch discrimination layer"""
    shape = list(input_shape)
    assert len(shape) == 3  # only valid for 3D tensors
    return tuple(shape[:2])


def single_layer_energy(x):
    shape = K.get_variable_shape(x)
    return K.reshape(K.sum(x, axis=range(1, len(shape))), (-1, 1))


def single_layer_energy_output_shape(input_shape):
    shape = list(input_shape)
    # assert len(shape) == 3
    return (shape[0], 1)


def calculate_energy(x):
    return Lambda(single_layer_energy, single_layer_energy_output_shape)(x)


def threshold_indicator(x, thresh):
    return K.cast(x > thresh, K.floatx())


def sparsity_level(x):
    _shape = K.get_variable_shape(x)
    shape = K.shape(x)
    total = K.cast(K.prod(shape[1:]), K.floatx())
    return K.reshape(K.sum(
        K.cast(x > 0.0, K.floatx()), axis=range(1, len(_shape))
    ), (-1, 1)) / total


def sparsity_output_shape(input_shape):
    shape = list(input_shape)
    return (shape[0], 1)


class Dense3D(Layer):

    """
    A 3D, trainable, dense tensor product layer
    """

    def __init__(self, first_dim,
                 last_dim,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense3D, self).__init__(**kwargs)
        self.first_dim = first_dim
        self.last_dim = last_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            shape=(self.first_dim, input_dim, self.last_dim),
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.first_dim, self.last_dim),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, mask=None):
        out = K.reshape(K.dot(inputs, self.kernel), (-1, self.first_dim, self.last_dim))
        if self.use_bias:
            out += self.bias
        return self.activation(out)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.first_dim, self.last_dim)

    def get_config(self):
        config = {
            'first_dim': self.first_dim,
            'last_dim': self.last_dim,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
