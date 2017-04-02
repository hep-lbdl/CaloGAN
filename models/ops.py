#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
file: ops.py
description: ancillary ops for [arXiv/1701.05927]
author: Luke de Oliveira (lukedeoliveira@lbl.gov)
"""

import keras.backend as K
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints, activations


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


class Dense3D(Layer):

    """
    A 3D, trainable, dense tensor product layer
    """

    def __init__(self, first_dim, last_dim, init='glorot_uniform',
                 activation=None, weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):

        self.init = initializers.get(init)
        self.activation = activations.get(activation)

        self.input_dim = input_dim
        self.first_dim = first_dim
        self.last_dim = last_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Dense3D, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(
            (self.first_dim, input_dim, self.last_dim),
            initializer=self.init,
            name='{}_W'.format(self.name),
            regularizer=self.W_regularizer,
            constraint=self.W_constraint
        )
        if self.bias:
            self.b = self.add_weight(
                (self.first_dim, self.last_dim),
                initializer='zero',
                name='{}_b'.format(self.name),
                regularizer=self.b_regularizer,
                constraint=self.b_constraint
            )
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x, mask=None):
        out = K.reshape(K.dot(x, self.W), (-1, self.first_dim, self.last_dim))
        if self.bias:
            out += self.b
        return self.activation(out)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.first_dim, self.last_dim)

    def get_config(self):
        config = {
            'first_dim': self.first_dim,
            'last_dim': self.last_dim,
            'init': self.init.__name__,
            'activation': self.activation.__name__,
            'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
            'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
            'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
            'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
            'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
            'bias': self.bias,
            'input_dim': self.input_dim
        }
        base_config = super(Dense3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
