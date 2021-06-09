import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense, LeakyReLU, Dropout, BatchNormalization


def build_generator(image_size, latent_code_length):
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)
    x = Input(latent_code_length, name='g_in')

    y = Dense(256, name='g_1', kernel_initializer=initializer)(x)
    y = BatchNormalization(name='g_1_bn')(y)
    y = LeakyReLU(name='g_1_act')(y)
    y = Dropout(0.1)(y)

    y = Dense(256, name='g_2', kernel_initializer=initializer)(y)
    y = BatchNormalization(name='g_2_bn')(y)
    y = LeakyReLU(name='g_2_act')(y)
    y = Dropout(0.1)(y)

    y = Dense(512, name='g_3', kernel_initializer=initializer)(y)
    y = BatchNormalization(name='g_3_bn')(y)
    y = LeakyReLU(name='g_3_act')(y)
    y = Dropout(0.1)(y)

    y = Dense(1024, name='g_4', kernel_initializer=initializer)(y)
    y = BatchNormalization(name='g_4_bn')(y)
    y = LeakyReLU(name='g_4_act')(y)
    y = Dropout(0.1)(y)

    y = Dense(image_size, activation='sigmoid', name='g_out', kernel_initializer=initializer)(y)

    return Model(x, y)


def build_encoder(image_size, latent_code_length):
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)
    x = Input(image_size)

    y = Dense(1024, name='e_1', kernel_initializer=initializer)(x)
    y = BatchNormalization(name='e_1_bn')(y)
    y = LeakyReLU(name='e_1_act')(y)
    y = Dropout(0.1)(y)

    y = Dense(512, name='e_2', kernel_initializer=initializer)(y)
    y = BatchNormalization(name='e_2_bn')(y)
    y = LeakyReLU(name='e_2_act')(y)
    y = Dropout(0.1)(y)

    y = Dense(256, name='e_3', kernel_initializer=initializer)(y)
    y = BatchNormalization(name='e_3_bn')(y)
    y = LeakyReLU(name='e_3_act')(y)
    y = Dropout(0.1)(y)

    y = Dense(256, name='e_4', kernel_initializer=initializer)(y)
    y = BatchNormalization(name='e_4_bn')(y)
    y = LeakyReLU(name='e_4_act')(y)
    y = Dropout(0.1)(y)

    y = Dense(latent_code_length, activation='sigmoid', name='e_out', kernel_initializer=initializer)(y)

    return Model(x, y)


def build_discriminator(image_size, latent_code_length):
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.02)
    act_feats = tf.keras.layers.Activation('sigmoid')

    x = Input(image_size, name='d_in_x')
    z = Input(latent_code_length, name='d_in_z')
    y = Concatenate(name='d_in')([x, z])

    y = Dense(1024, name='d_1', kernel_initializer=initializer)(y)
    y = BatchNormalization(name='d_1_bn')(y)
    y = LeakyReLU(name='d_1_act')(y)
    y = Dropout(0.1)(y)

    y = Dense(512, name='d_2', kernel_initializer=initializer)(y)
    y = BatchNormalization(name='d_2_bn')(y)
    y = LeakyReLU(name='d_2_act')(y)
    y = Dropout(0.1)(y)

    y = Dense(256, name='d_3', kernel_initializer=initializer)(y)
    y = BatchNormalization(name='d_3_bn')(y)
    y = LeakyReLU(name='d_3_act')(y)
    y = Dropout(0.1)(y)

    y = Dense(256, name='d_4', kernel_initializer=initializer)(y)
    y = BatchNormalization(name='d_4_bn')(y)
    y = act_feats(y)
    y = Dropout(0.1)(y)

    # intermediate feature
    out_features = y

    y = Dense(1, name='d_out', kernel_initializer=initializer)(y)

    return Model([x, z], [y, out_features])
