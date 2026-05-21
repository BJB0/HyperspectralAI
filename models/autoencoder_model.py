from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


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