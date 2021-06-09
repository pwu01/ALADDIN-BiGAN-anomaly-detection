"""
By Peng Wu, peng.wu.14@ucl.ac.uk, 12/08/2020

Generative Adversarial Network (GAN) anomaly detection
This scripts builds Bidirectional Generative Adversarial Network (GAN) based anomaly detection

Key references:
-   Generative Adversarial Nets, https://arxiv.org/abs/1406.2661
-   Adversarially Learned Inference (ALI-GAN), https://arxiv.org/abs/1606.00704
-   Adversarial Feature Learning (BI-GAN), https://arxiv.org/pdf/1605.09782
-   A Survey on GANs for Anomaly Detection, https://arxiv.org/pdf/1906.11632
-   f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks,
        https://doi.org/10.1016/j.media.2019.01.010
"""


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle
import os
import time
from copy import deepcopy
import sys
from model import build_generator, build_encoder, build_discriminator


def validate(x, dis, gen, enc, kappa):

    x = tf.reshape(x, (1, -1))
    z = enc(x)
    x_gen = gen(z)

    # compute the L2 norm of the difference between the data and the generator output (reconstructed data)
    # NB: 960 is the data size
    score_ge_1 = tf.norm(x-x_gen, ord='euclidean') / 960
    # compute the anomaly score from feature layer of discriminator
    _, f1 = dis([x, z], training=False)
    _, f2 = dis([x_gen, z], training=False)
    # NB: 256 is the feature layer size
    score_ge_2 = tf.norm(f1-f2, ord='euclidean') * kappa / 256
    abnormal_score = score_ge_1 + score_ge_2

    return abnormal_score, score_ge_1, score_ge_2


def test_batch(batch, dis, gen, enc, d_size, kappa):
    x = batch
    z = enc(batch)
    x_gen = gen(z)

    # compute the L2 norm of the difference between the data and the generator output (reconstructed data)
    score_ge_2 = tf.norm(x - x_gen, ord='euclidean', axis=1) / d_size
    # compute the L2 norm from feature layer
    _, f1 = dis([x, z], training=False)
    _, f2 = dis([x_gen, z], training=False)

    score_ge_1 = tf.norm(f1-f2, ord='euclidean', axis=1) * kappa / 256  # NB: 256 is the feature layer size
    abnormal_score = score_ge_1 + score_ge_2

    return abnormal_score, score_ge_1, score_ge_2


def de_normalise_data(data, sensor_list, de_normal_info):

    min_max = de_normal_info
    no_sensors = len(sensor_list)

    data = data.reshape(data.shape[0], no_sensors, -1)

    for i in range(data.shape[0]):
        for j in range(no_sensors):
            value_max = min_max[sensor_list[j]]['max']
            value_min = min_max[sensor_list[j]]['min']
            data[i, j, :] = data[i, j, :] * (value_max-value_min) + value_min

    return data


def get_validation_data(original_data, sensor_list):
    """
    Single sensor failure
    Dual sensor failure
    Triple sensor failure
    :param original_data:
    :param sensor_list:
    :return:
    """
    no_data = original_data.shape[0]

    no_sensors = len(sensor_list)

    no_steps = original_data.shape[2]

    data_1_fail = deepcopy(original_data)

    data_2_fail = deepcopy(original_data)

    data_3_fail = deepcopy(original_data)

    idx = np.arange(no_sensors)

    for n in range(no_data):

        np.random.shuffle(idx)
        a = idx[0]
        b = idx[0:2]
        c = idx[0:3]

        data_1_fail[n, a, :] = np.zeros(no_steps)

        data_2_fail[n, b[0], :] = np.zeros(no_steps)
        data_2_fail[n, b[1], :] = np.zeros(no_steps)

        data_3_fail[n, c[0], :] = np.zeros(no_steps)
        data_3_fail[n, c[1], :] = np.zeros(no_steps)
        data_3_fail[n, c[2], :] = np.zeros(no_steps)

    return data_1_fail, data_2_fail, data_3_fail


def prepare_data(sensor_list):
    # training data
    filename = 'multi_vehicle' + '.pkl'
    data_all = pickle.load(open(os.path.join('training_data', filename), 'rb'))
    data_t0 = data_all['data']
    data_t0_cycle = data_all['cycle']
    data_t0_vehicle = data_all['vehicle']

    # validation data
    data_t1, data_t2, data_t3 = get_validation_data(data_t0, sensor_list)

    # reshape the data
    data_t0 = data_t0.reshape(data_t0.shape[0], data_t0.shape[1] * data_t0.shape[2]).astype('float32')
    data_t1 = data_t1.reshape(data_t1.shape[0], data_t1.shape[1] * data_t1.shape[2]).astype('float32')
    data_t2 = data_t2.reshape(data_t2.shape[0], data_t2.shape[1] * data_t2.shape[2]).astype('float32')
    data_t3 = data_t3.reshape(data_t3.shape[0], data_t3.shape[1] * data_t3.shape[2]).astype('float32')

    return data_t0, data_t0_cycle, data_t0_vehicle, data_t1, data_t2, data_t3


def batch_generator(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def application(data, study, batch_size, data_size, discriminator, generator, encoder, kappa):
    application_results = {}

    for vehicle, test_data in data.items():
        print(vehicle)
        total_no = len(test_data['data'])  # total record number

        application_results[vehicle] = {'r_v': np.zeros([total_no]),
                                        'r_1': np.zeros([total_no]),
                                        'r_2': np.zeros([total_no]),
                                        'cycle': np.zeros([total_no])}

        for batch_idx in batch_generator(np.arange(total_no), batch_size):

            batch = test_data['data'][batch_idx, :, :]
            batch = tf.convert_to_tensor(batch, dtype=tf.float32)
            batch = tf.reshape(batch, (-1, data_size))

            r_v_batch, r_v_1_batch, r_v_2_batch = test_batch(batch, discriminator, generator, encoder,
                                                             d_size=data_size, kappa=kappa)

            application_results[vehicle]['r_v'][batch_idx] = r_v_batch.numpy()
            application_results[vehicle]['r_1'][batch_idx] = r_v_1_batch.numpy()
            application_results[vehicle]['r_2'][batch_idx] = r_v_2_batch.numpy()
            application_results[vehicle]['cycle'][batch_idx] = test_data['cycle'][batch_idx]

    vessel_list = []
    for vehicle in data:
        vessel_list.append(vehicle)

    filename = 'test_results_' + study + '.pkl'
    if not os.path.exists('application_results'):
        os.makedirs('application_results')
    pickle.dump(application_results, open(os.path.join('application_results', filename), 'wb'))


def main(vehicle, train):
    np.random.seed(10)
    tf.random.set_seed(10)
    dts = np.array([5, 10, 30, 60, 120, 240])

    KAPPA = 2.
    CHECK_POINT = 100
    ITERS = 8000
    LATENT_CODE_LENGTH = 256
    BATCH_SIZE = 256

    # create logger
    writer = tf.summary.create_file_writer(os.path.join('logs', str(time.time())))

    # load sensor and de-normalisation info
    if vehicle == 'multi_vehicle':
        sensor_list_file = 'sensor_list_multi.pkl'
        de_normal_file = 'de_normal_multi.pkl'
    else:
        sensor_list_file = 'sensor_list.pkl'
        de_normal_file = 'de_normal.pkl'
    sensor_list = pickle.load(open(os.path.join('training_data', sensor_list_file), 'rb'))

    # load data for training and testing
    x_train, y_train, glider, x_test_1, x_test_2, x_test_3 = prepare_data(sensor_list)
    num_of_data = x_train.shape[0]
    data_size = x_train.shape[1]
    KAPPA = tf.constant(KAPPA)

    # run sensitivity study if not in training mode by loading existing models
    if not train:
        generator = tf.keras.models.load_model('saved_model/generator.h5', compile=False)
        encoder = tf.keras.models.load_model('saved_model/encoder.h5', compile=False)
        discriminator = tf.keras.models.load_model('saved_model/discriminator.h5', compile=False)
        # load base line
        file_name = 'test_data_all.pkl'
        study = 'base_line'
        test_data_all = pickle.load(open(os.path.join('test_data', file_name), 'rb'))
        application(test_data_all, study, BATCH_SIZE, data_size, discriminator, generator, encoder, KAPPA)

        # group a
        i = 0
        n = len(sensor_list) * len(dts)
        for s in sensor_list:
            for dt in dts:
                i += 1
                print('Group A', i, 'out of', n, 'studies')
                study = s + '_' + str(dt)
                file_name = 'test_data_all_' + study + '.pkl'
                test_data_all = pickle.load(open(os.path.join('test_data', file_name), 'rb'))
                application(test_data_all, study, BATCH_SIZE, data_size, discriminator, generator, encoder, KAPPA)

        # group b
        i = 0
        n = len(dts)
        for dt in dts:
            i += 1
            print('Group B:', i, 'out of', n, 'studies', 'Resampling dt = ', dt)
            study = str(dt)
            file_name = 'test_data_group_b_' + study + '.pkl'
            test_data_all = pickle.load(open(os.path.join('test_data', file_name), 'rb'))
            application(test_data_all, study, BATCH_SIZE, data_size, discriminator, generator, encoder, KAPPA)

        sys.exit()

    # build models
    generator = build_generator(data_size, LATENT_CODE_LENGTH)
    encoder = build_encoder(data_size, LATENT_CODE_LENGTH)
    discriminator = build_discriminator(data_size, LATENT_CODE_LENGTH)

    # set optimizers
    d_optimizer = Adam(learning_rate=0.00001)
    ge_optimizer = Adam(learning_rate=0.00001)

    @tf.function
    def train_step_ge(x, kappa):

        with tf.GradientTape() as tape:
            z_gen = encoder(x, training=True)
            x_gen = generator(z_gen, training=True)

            _, f_real = discriminator([x, z_gen], training=False)
            _, f_fake = discriminator([x_gen, z_gen], training=False)

            loss_1 = tf.reduce_mean(tf.norm(f_real - f_fake, ord='euclidean', axis=1)) * kappa / 256
            loss_2 = tf.reduce_mean(tf.norm(x - x_gen, ord='euclidean', axis=1)) / data_size

            loss = loss_1 + loss_2

        ge_gradients = tape.gradient(loss, generator.trainable_variables + encoder.trainable_variables)
        ge_optimizer.apply_gradients(zip(ge_gradients, generator.trainable_variables + encoder.trainable_variables))

        return loss_1, loss_2

    @tf.function
    def train_step_dis(x):

        with tf.GradientTape() as tape:
            z = tf.random.truncated_normal([BATCH_SIZE, LATENT_CODE_LENGTH], mean=0., stddev=1.)
            z_gen = encoder(x, training=False)
            x_gen = generator(z, training=False)
            l_encoder, inter_layer_inp = discriminator([x, z_gen], training=True)
            l_generator, inter_layer_rct = discriminator([x_gen, z], training=True)
            loss_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder),
                                                                              logits=l_encoder))
            loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator),
                                                                              logits=l_generator))
            loss = loss_enc + loss_gen

        d_gradients = tape.gradient(target=loss,
                                    sources=discriminator.trainable_variables
                                            + generator.trainable_variables
                                            + encoder.trainable_variables)
        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables
                                                     + generator.trainable_variables
                                                     + encoder.trainable_variables))

        return loss

    t_idx, all_idx, ge_idx = 0, 0, 0

    start_time = time.time()
    for i in range(ITERS):
        real_data = x_train[np.random.permutation(num_of_data)[:BATCH_SIZE]]

        real_data = tf.convert_to_tensor(real_data)

        # train all
        for _ in range(10):
            all_loss = train_step_dis(real_data)
            with writer.as_default():
                tf.summary.scalar('all_loss', all_loss, step=all_idx)
            all_idx += 1

        # train generator and encoder
        ge_loss_1, ge_loss_2 = train_step_ge(real_data, kappa=KAPPA)

        ge_idx += 1
        with writer.as_default():
            tf.summary.scalar('ge_loss_1', ge_loss_1, step=ge_idx)
            tf.summary.scalar('ge_loss_2', ge_loss_2, step=ge_idx)
        print("\r[{}/{}]  ge_loss_1: {:.4}, ge_loss_2: {:.4}"
              .format(ge_idx, ITERS, ge_loss_1, ge_loss_2), end="")

        if (ge_idx+1) % CHECK_POINT == 0:
            # Save the entire model as a SavedModel.
            if not os.path.exists('saved_model'):
                os.makedirs('saved_model')
            generator.save('saved_model/generator.h5')
            encoder.save('saved_model/encoder.h5')
            discriminator.save('saved_model/discriminator.h5')

            for t in range(10):
                cycle = np.random.randint(0, num_of_data)
                t_patch_0 = tf.convert_to_tensor(x_train[cycle])
                t_patch_1 = tf.convert_to_tensor(x_test_1[cycle])
                t_patch_2 = tf.convert_to_tensor(x_test_2[cycle])
                t_patch_3 = tf.convert_to_tensor(x_test_3[cycle])

                r0, s_0_1, s_0_2 = validate(t_patch_0, discriminator, generator, encoder, sensor_list)
                r1, s_1_1, s_1_2 = validate(t_patch_1, discriminator, generator, encoder, sensor_list)
                r2, s_2_1, s_2_2 = validate(t_patch_2, discriminator, generator, encoder, sensor_list)
                r3, s_3_1, s_3_2 = validate(t_patch_3, discriminator, generator, encoder, sensor_list)

                with writer.as_default():
                    tf.summary.scalar('Validate_0_sensor_abnormal', r0, step=t_idx)
                    tf.summary.scalar('Validate_0_sensor_abnormal', s_0_1, step=t_idx)
                    tf.summary.scalar('Validate_0_sensor_abnormal', s_0_2, step=t_idx)

                    tf.summary.scalar('Validate_1_sensors_abnormal', r1, step=t_idx)
                    tf.summary.scalar('Validate_1_sensors_abnormal', s_1_1, step=t_idx)
                    tf.summary.scalar('Validate_1_sensors_abnormal', s_1_2, step=t_idx)

                    tf.summary.scalar('Validate_2_sensors_abnormal', r2, step=t_idx)
                    tf.summary.scalar('Validate_2_sensors_abnormal', s_2_1, step=t_idx)
                    tf.summary.scalar('Validate_2_sensors_abnormal', s_2_2, step=t_idx)

                    tf.summary.scalar('Validate_3_sensors_abnormal', r3, step=t_idx)
                    tf.summary.scalar('Validate_3_sensors_abnormal', s_3_1, step=t_idx)
                    tf.summary.scalar('Validate_3_sensors_abnormal', s_3_2, step=t_idx)
                t_idx += 1

    # print training time
    if train:
        training_time = time.time() - start_time
        print('Training time:', training_time)


if __name__ == "__main__":
    vessel = 'multi_vehicle'
    main(vessel, train=False)
