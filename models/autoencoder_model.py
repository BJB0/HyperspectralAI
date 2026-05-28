from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Reshape


def build_autoencoder(input_dim, latent_dim=10):

    inp = Input(shape=(input_dim,))

    x = Dense(256, activation='relu')(inp)
    x = Dense(128, activation='relu')(x)

    latent = Dense(latent_dim)(x)

    x = Dense(128, activation='relu')(latent)
    x = Dense(256, activation='relu')(x)

    out = Dense(input_dim)(x)

    autoencoder = Model(inp, out)

    encoder = Model(inp, latent)

    autoencoder.compile(
        optimizer='adam',
        loss='mse'
    )

    return autoencoder, encoder


def build_cnn_autoencoder(input_shape, latent_dim=10):

    inp = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    shape_before_flatten = x.shape[1:]

    x = Flatten()(x)
    latent = Dense(latent_dim, name='cnn_latent')(x)

    x = Dense(
        int(shape_before_flatten[0] * shape_before_flatten[1] * shape_before_flatten[2]),
        activation='relu'
    )(latent)
    x = Reshape(shape_before_flatten)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    out = Conv2D(input_shape[-1], (3, 3), activation='linear', padding='same')(x)

    autoencoder = Model(inp, out)

    encoder = Model(inp, latent)

    autoencoder.compile(
        optimizer='adam',
        loss='mse'
    )

    return autoencoder, encoder
