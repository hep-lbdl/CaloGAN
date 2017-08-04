import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import (Dense, Reshape, Conv2D, LeakyReLU, BatchNormalization,
                          LocallyConnected2D, Activation, ZeroPadding2D,
                          Dropout, Lambda, Flatten, Input, AlphaDropout)

from keras.layers.merge import concatenate, multiply
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from functools import partial
import h5py


def make_roc(y, yhat, signal, background, outfile='plot.pdf'):
    import matplotlib
    matplotlib.use('PDF')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, classification_report

    print 'ACCURACY: %{}'.format(100 * ((yhat > 0.5) == y).mean())
    print classification_report(y, yhat > 0.5)

    fpr, tpr, _ = roc_curve(y, yhat)

    plt.figure(figsize=(10, 10))
    plt.plot(tpr, 1 / fpr, color='red', label='ROC Curve')
    plt.yscale('log')
    plt.grid('on', 'both')
    plt.xlabel('{} Efficiency'.format(signal))
    plt.ylabel('{} Background Rejection'.format(background))
    plt.legend(fontsize=30)
    plt.savefig(outfile)

    plt.figure(figsize=(10, 10))
    plt.plot(tpr, fpr, color='red', label='ROC Curve')
    plt.yscale('log')
    plt.grid('on', 'both')
    plt.xlabel('{} Efficiency'.format(signal))
    plt.ylabel('{} Background efficiency'.format(background))
    plt.legend(fontsize=30)
    plt.savefig('normal-' + outfile)


def load_calodata(fpath):
    with h5py.File(fpath, 'r') as h5:
        data = [h5['layer_{}'.format(i)][:] for i in xrange(3)]
    return data


def build_model(image):
    x = Conv2D(64, (2, 2), padding='same')(image)
    x = LeakyReLU()(x)

    x = ZeroPadding2D((1, 1))(x)
    x = LocallyConnected2D(16 * 4, (3, 3), padding='valid', strides=(1, 2))(x)
    # x = LeakyReLU()(x)
    x = Activation('selu')(x)
    x = AlphaDropout(0.05)(x)
    # x = BatchNormalization()(x)

    x = ZeroPadding2D((1, 1))(x)
    x = LocallyConnected2D(8 * 4, (2, 2), padding='valid')(x)
    # x = LeakyReLU()(x)
    x = Activation('selu')(x)
    x = AlphaDropout(0.05)(x)

    # x = BatchNormalization()(x)

    x = ZeroPadding2D((1, 1))(x)
    x = LocallyConnected2D(8 * 4, (2, 2), padding='valid', strides=(1, 2))(x)
    # x = LeakyReLU()(x)
    x = Activation('selu')(x)
    x = AlphaDropout(0.05)(x)

    # x = BatchNormalization()(x)

    x = Flatten()(x)

    return x

if __name__ == '__main__':

    CLASS_ONE = 'gamma'
    CLASS_TWO = 'eplus'

    concat = partial(np.concatenate, axis=0)

    c1 = load_calodata('{}.hdf5'.format(CLASS_ONE))
    c2 = load_calodata('{}.hdf5'.format(CLASS_TWO))
    data = map(concat, zip(c1, c2))

    labels = np.array([1] * c1[0].shape[0] + [0] * c2[0].shape[0])

    ix = np.array(range(len(labels)))
    np.random.shuffle(ix)

    nb_train = int(0.7 * len(ix))

    ix_train = ix[:nb_train]
    ix_test = ix[nb_train:]

    data_train = [np.expand_dims(d[ix_train], -1) for d in data]
    labels_train = labels[ix_train]

    data_test = [np.expand_dims(d[ix_test], -1) for d in data]
    labels_test = labels[ix_test]

    shapes = [d.shape[1:] for d in data_train]

    x = [Input(shape=sh) for sh in shapes]

    h = concatenate(map(build_model, x))

    h = Dense(128)(h)
    h = Activation('selu')(h)
    h = AlphaDropout(0.05)(h)

    y = Dense(1, activation='sigmoid')(h)

    net = Model(x, y)

    net.compile('adam', 'binary_crossentropy', metrics=['acc'])

    callbacks = [
        EarlyStopping(verbose=True, patience=7, monitor='val_acc'),
        ModelCheckpoint('{}vs{}-chkpt.h5'.format(CLASS_ONE, CLASS_TWO),
                        monitor='val_acc', verbose=True, save_best_only=True),
    ]

    try:
        net.fit(data_train, labels_train, callbacks=callbacks, verbose=True,
                validation_split=0.3, batch_size=64, epochs=100)
    except KeyboardInterrupt:
        print 'ending early'

    net.load_weights('{}vs{}-chkpt.h5'.format(CLASS_ONE, CLASS_TWO))
    net.save_weights('{}vs{}-final.h5'.format(CLASS_ONE, CLASS_TWO))

    with h5py.File('{}vs{}-split-indices.h5'.format(CLASS_ONE, CLASS_TWO), 'w') as h5:
        h5['train'] = ix_train
        h5['test'] = ix_test

    yhat = net.predict(data_test, verbose=True).ravel()

    make_roc(labels_test, yhat, signal=CLASS_ONE, background=CLASS_TWO,
             outfile='{}vs{}-roc.pdf'.format(CLASS_ONE, CLASS_TWO))
