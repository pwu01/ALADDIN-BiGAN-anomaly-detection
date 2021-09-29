import tensorflow as tf
import pickle
from copy import deepcopy
import numpy as np
import os


def test_batch(batch, dis, gen, enc, d_size, f_size, kappa):
    x = batch
    z = enc(batch)
    x_gen = gen(z)

    # compute the L2 norm of the difference between the data and the generator output (reconstructed data)
    score_ge_2 = tf.norm(x - x_gen, ord='euclidean', axis=1) / d_size
    # compute the L2 norm from feature layer
    _, f1 = dis([x, z], training=False)
    _, f2 = dis([x_gen, z], training=False)

    score_ge_1 = tf.norm(f1-f2, ord='euclidean', axis=1) * kappa / f_size
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


def prepare_data(vehicle):
    # load sensor list
    if vehicle == 'multi_vehicle':
        sensor_list_file = 'sensor_list_multi.pkl'
    else:
        sensor_list_file = 'sensor_list.pkl'
    sensor_list = pickle.load(open(os.path.join('training_data', sensor_list_file), 'rb'))
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
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


def application(data, study, bigan):
    batch_size = bigan.batch_size
    data_size = bigan.data_size
    discriminator = bigan.discriminator
    generator = bigan.generator
    encoder = bigan.encoder
    kappa = bigan.kappa
    f_size = bigan.feature_size

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
                                                             d_size=data_size, f_size=f_size, kappa=kappa)

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
