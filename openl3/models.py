import os
import warnings
from openl3.openl3_exceptions import OpenL3Error

with warnings.catch_warnings():
    # Suppress TF and Keras warnings when importing
    warnings.simplefilter("ignore")

    from kapre.time_frequency import Spectrogram, Melspectrogram
    from keras.layers import (
        Input, Conv2D, BatchNormalization, MaxPooling2D,
        Flatten, Activation, Lambda)

    from keras.models import Model
    import keras.regularizers as regularizers
    from keras.utils.conv_utils import conv_output_length





PARAM_OVERRIDES = {
    'linear': {
        'n_dft': 512,
        'n_hop': 242
    },
    'mel128': {
        'n_dft': 2048,
        'n_hop': 242,
        'n_mels': 128
    },
    'mel256': {
        'n_dft': 2048,
        'n_hop': 242,
        'n_mels': 256
    },
}


def build_openl3_audio_network(
        input_repr='linear',
        n_dft=512, n_hop=242, sr=48000,
        n_mels=128, audio_window_dur=1,
        precomputed=False, input_shape=None):
    """
    Returns an uninitialized model object for a network with a Mel
    spectrogram input (with 128 frequency bins). No final pooling/flattening
    is performed.

    Returns
    -------
    model : keras.models.Model
        Model object.
    """

    weight_decay = 1e-5

    # CONV BLOCK 1
    n_filter_a_1 = 64
    filt_size_a_1 = 3, 3
    pool_size_a_1 = 2, 2

    # CONV BLOCK 2
    n_filter_a_2 = 128
    filt_size_a_2 = 3, 3
    pool_size_a_2 = 2, 2

    # CONV BLOCK 3
    n_filter_a_3 = 256
    filt_size_a_3 = 3, 3
    pool_size_a_3 = 2, 2

    # CONV BLOCK 4
    n_filter_a_4 = 512
    filt_size_a_4 = 3, 3



    if precomputed:
        # PRECOMPUTED INPUT SPECTROGRAMS
        if input_shape is None: # calculate input shape
            n_channels = n_dft // 2 + 1 if input_repr == 'linear' else n_mels

            n_spec_frames = conv_output_length(
                sr * audio_window_dur, n_dft, 'same', n_hop)

            input_shape = n_channels, n_spec_frames, 1

        x_a = y_a = Input(shape=input_shape, dtype='float32')
    else:
        x_a = Input(shape=(1, sr * audio_window_dur), dtype='float32')

        if input_repr == 'linear':
            # SPECTROGRAM PREPROCESSING
            # 257 x 199 x 1 (for defaults)
            y_a = Spectrogram(n_dft=n_dft, n_hop=n_hop, power_spectrogram=1.0,
                              return_decibel_spectrogram=True, padding='valid')(x_a)

        else:
            # MELSPECTROGRAM PREPROCESSING
            # {n_mels} x 199 x 1
            y_a = Melspectrogram(n_dft=n_dft, n_hop=n_hop, n_mels=n_mels,
                                 sr=sr, power_melgram=1.0, htk=True, # n_win=n_win,
                                 return_decibel_melgram=True, padding='same')(x_a)

    y_a = BatchNormalization()(y_a)

    # CONV BLOCK 1
    y_a = audio_conv_layer(
        y_a, n_filter_a_1, filt_size_a_1, weight_decay)

    y_a = audio_conv_layer(
        y_a, n_filter_a_1, filt_size_a_1, weight_decay, pool_size_a_1)

    # CONV BLOCK 2
    y_a = audio_conv_layer(
        y_a, n_filter_a_2, filt_size_a_2, weight_decay)

    y_a = audio_conv_layer(
        y_a, n_filter_a_2, filt_size_a_2, weight_decay, pool_size_a_2)

    # CONV BLOCK 3
    y_a = audio_conv_layer(
        y_a, n_filter_a_3, filt_size_a_3, weight_decay)

    y_a = audio_conv_layer(
        y_a, n_filter_a_3, filt_size_a_3, weight_decay, pool_size_a_3)

    # CONV BLOCK 4
    y_a = audio_conv_layer(
        y_a, n_filter_a_4, filt_size_a_4, weight_decay)

    y_a = Conv2D(n_filter_a_4, filt_size_a_4, padding='same',
                 kernel_initializer='he_normal', name='audio_embedding_layer',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)

    m = Model(inputs=x_a, outputs=y_a)
    print('out', m.output_shape)
    return m



def _get_factors(x):
    return ((i, int(x/i)) for i in range(int(x**0.5), 0, -1) if not x % i)

def calc_pool_size_embedding(output_shape, embedding_size):
    '''Calculate the pooling size for the final layer to get the desired embedding shape.'''
    _, shp_f, shp_t, n_filter_a_4 = output_shape

    # calculate the amount to pool along each axis
    scaling_factor = embedding_size / n_filter_a_4
    try:
        # we need integer factors to scale, otherwise it won't work.
        factors = _get_factors(scaling_factor)

        if shp_f > shp_t: # factors are returned from lowest to highest, so swap order.
            factors = ((b, a) for a, b in factors)

        # scaling factors
        s_f, s_t = next((scale_f, scale_t) for scale_f, scale_t in factors
                        if not shp_f % scale_f and not shp_t % scale_t)

    except StopIteration:
        raise OpenL3Error('Embedding of size {} could not be created from '
                          'a output channel size of {}'.format(
                          embedding_size, n_filter_a_4))

    return int(shp_f / s_f), int(shp_t / s_t)


def audio_conv_layer(y_a, n_filter, filt_size, weight_decay, pool_size=None):
    '''Builds a single convolutional block of the L3 network'''
    # CONV BLOCK
    y_a = Conv2D(n_filter, filt_size, padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=regularizers.l2(weight_decay))(y_a)
    y_a = BatchNormalization()(y_a)
    y_a = Activation('relu')(y_a)

    if pool_size: # optional pooling layer
        y_a = MaxPooling2D(pool_size=pool_size, strides=2)(y_a)

    return y_a



'''



Model Loading


'''


POOLINGS = {
    'linear': {
        6144: (8, 8),
        512: (32, 24),
    },
    'mel128': {
        6144: (4, 8),
        512: (16, 24),
    },
    'mel256': {
        6144: (8, 8),
        512: (32, 24),
    }
}



def load_embedding_model(input_repr, content_type, embedding_size, **kw):
    """
    Returns a model with the given characteristics. Loads the model
    if the model has not been loaded yet.

    Parameters
    ----------
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model.
    content_type : "music" or "env"
        Type of content used to train embedding.
    embedding_size : 6144 or 512
        Embedding dimensionality.

    Returns
    -------
    model : keras.models.Model
        Model object.
    """

    # Construct embedding model and load model weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        kw = dict(PARAM_OVERRIDES[input_repr], **kw)
        m = build_openl3_audio_network(input_repr=input_repr, **kw)

    weights_file = load_embedding_model_path(input_repr, content_type)
    m.load_weights(weights_file)

    # build embedding
    pool_size_emb = calc_pool_size_embedding(m.output_shape, embedding_size)
    y_a = MaxPooling2D(pool_size=pool_size_emb, padding='same')(m.output)
    y_a = Flatten()(y_a)

    m = Model(inputs=m.input, outputs=y_a)
    return m


def load_embedding_model_path(input_repr, content_type):
    """
    Returns the local path to the model weights file for the model
    with the given characteristics

    Parameters
    ----------
    input_repr : "linear", "mel128", or "mel256"
        Spectrogram representation used for model.
    content_type : "music" or "env"
        Type of content used to train embedding.

    Returns
    -------
    output_path : str
        Path to given model object
    """
    return os.path.join(os.path.dirname(__file__),
                        'openl3_audio_{}_{}.h5'.format(input_repr, content_type))


def _construct_linear_audio_network():
    """
    Returns an uninitialized model object for a network with a linear
    spectrogram input (With 257 frequency bins)

    Returns
    -------
    model : keras.models.Model
        Model object.
    """

    return build_openl3_audio_network(input_repr='linear', **PARAM_OVERRIDES['linear'])


def _construct_mel128_audio_network():
    """
    Returns an uninitialized model object for a network with a Mel
    spectrogram input (with 128 frequency bins).

    Returns
    -------
    model : keras.models.Model
        Model object.
    """

    return build_openl3_audio_network(input_repr='mel128', **PARAM_OVERRIDES['mel128'])


def _construct_mel256_audio_network():
    """
    Returns an uninitialized model object for a network with a Mel
    spectrogram input (with 256 frequency bins).

    Returns
    -------
    model : keras.models.Model
        Model object.
    """

    return build_openl3_audio_network(input_repr='mel256', **PARAM_OVERRIDES['mel256'])


MODELS = {
    'linear': _construct_linear_audio_network,
    'mel128': _construct_mel128_audio_network,
    'mel256': _construct_mel256_audio_network
}
