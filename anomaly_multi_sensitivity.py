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
import time
import pickle
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import build_generator, build_encoder, build_discriminator
from utilities import prepare_data, application


def get_args():
    """
    Define hyper parameters
    """
    parser = argparse.ArgumentParser(description='BiGANAnomaly')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--feature_size', default=256, type=int)
    parser.add_argument('--latent_size', default=256, type=int)
    parser.add_argument('--kappa', default=2., type=float)
    parser.add_argument('--ge_learn_rate', default=0.0001, type=float)
    parser.add_argument('--d_learn_rate', default=0.0001, type=float)
    parser.add_argument('--num_updates', default=8000, type=int)
    parser.add_argument('--check_point', default=100, type=int)
    parser.add_argument('--interval', default=10, type=int)

    args = parser.parse_args()
    return args


class BiGANAnomaly:
    def __init__(self,
                 vehicle,
                 train,
                 batch_size,
                 data_size,
                 feature_size,
                 latent_size,
                 kappa,
                 ge_learn_rate,
                 d_learn_rate):

        self.batch_size = batch_size
        self.data_size = data_size
        self.latent_size = latent_size
        self.feature_size = feature_size
        self.kappa = tf.constant(kappa)

        # load sensor list and de-normalisation info
        if vehicle == 'multi_vehicle':
            sensor_list_file = 'sensor_list_multi.pkl'
            de_normal_file = 'de_normal_multi.pkl'
        else:
            sensor_list_file = 'sensor_list.pkl'
            de_normal_file = 'de_normal.pkl'
        self.sensor_list = pickle.load(open(os.path.join('training_data', sensor_list_file), 'rb'))
        self.de_normal_info = pickle.load(open(os.path.join('training_data', de_normal_file), 'rb'))

        # initialise the neural networks: train from scratch or load pre-trained model
        if train is True:
            self.encoder = build_encoder(data_size, latent_size)
            self.generator = build_generator(data_size, latent_size)
            self.discriminator = build_discriminator(data_size, latent_size)
        else:
            self.encoder = tf.keras.models.load_model('saved_model/encoder.h5', compile=False)
            self.generator = tf.keras.models.load_model('saved_model/generator.h5', compile=False)
            self.discriminator = tf.keras.models.load_model('saved_model/discriminator.h5', compile=False)

        # initialise a logger if training is true
        self.writer = tf.summary.create_file_writer(os.path.join('logs', str(time.time())))

        # initialise optimisers
        self.ge_optimizer = Adam(learning_rate=ge_learn_rate)
        self.d_optimizer = Adam(learning_rate=d_learn_rate)

        # initialise counters
        self.ver_idx = 0  # verification counter
        self.all_idx = 0  # counter of G, E, D updated together
        self.ge_idx = 0  # counter of only G & E updated

    @tf.function
    def train_step_ge(self, x):
        # update the Generator and Encoder
        with tf.GradientTape() as tape:
            z_gen = self.encoder(x, training=True)
            x_gen = self.generator(z_gen, training=True)

            _, f_real = self.discriminator([x, z_gen], training=False)
            _, f_fake = self.discriminator([x_gen, z_gen], training=False)

            loss_1 = tf.reduce_mean(tf.norm(f_real - f_fake, ord='euclidean', axis=1)) * self.kappa / self.feature_size
            loss_2 = tf.reduce_mean(tf.norm(x - x_gen, ord='euclidean', axis=1)) / self.data_size

            loss = loss_1 + loss_2

        ge_gradients = tape.gradient(loss, self.generator.trainable_variables + self.encoder.trainable_variables)
        self.ge_optimizer.apply_gradients(zip(ge_gradients,
                                              self.generator.trainable_variables + self.encoder.trainable_variables))

        return loss_1, loss_2

    @tf.function
    def train_step_dis(self, x):
        with tf.GradientTape() as tape:
            z = tf.random.truncated_normal([self.batch_size, self.latent_size], mean=0., stddev=1.)
            z_gen = self.encoder(x, training=False)
            x_gen = self.generator(z, training=False)
            l_encoder, inter_layer_inp = self.discriminator([x, z_gen], training=True)
            l_generator, inter_layer_rct = self.discriminator([x_gen, z], training=True)
            loss_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder),
                                                                              logits=l_encoder))
            loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator),
                                                                              logits=l_generator))
            loss = loss_enc + loss_gen

        d_gradients = tape.gradient(target=loss,
                                    sources=self.discriminator.trainable_variables +
                                    self.generator.trainable_variables +
                                    self.encoder.trainable_variables)

        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables +
                                             self.generator.trainable_variables +
                                             self.encoder.trainable_variables))

        return loss

    def test(self):
        # resampling interval settings for the sensitivity study
        dts = np.array([5, 10, 30, 60, 120, 240])

        # load base line
        file_name = 'test_data_all.pkl'
        study = 'base_line'
        test_data_all = pickle.load(open(os.path.join('test_data', file_name), 'rb'))
        application(test_data_all, study, self)

        # sensitivity study group a
        i = 0
        n = len(self.sensor_list) * len(dts)
        for s in self.sensor_list:
            for dt in dts:
                i += 1
                print('Group A', i, 'out of', n, 'studies')
                study = s + '_' + str(dt)
                file_name = 'test_data_all_' + study + '.pkl'
                test_data_all = pickle.load(open(os.path.join('test_data', file_name), 'rb'))
                application(test_data_all, study, self)

        # sensitivity study group b
        i = 0
        n = len(dts)
        for dt in dts:
            i += 1
            print('Group B:', i, 'out of', n, 'studies', 'Resampling dt = ', dt)
            study = str(dt)
            file_name = 'test_data_group_b_' + study + '.pkl'
            test_data_all = pickle.load(open(os.path.join('test_data', file_name), 'rb'))
            application(test_data_all, study, self)

    def verification_batch(self, batch, d_size, f_size, kappa):
        # periodical verification of the model in training
        x = batch
        z = self.encoder(batch)
        x_gen = self.generator(z)

        # compute the L2 norm of the difference between the data and the generator output (reconstructed data)
        score_ge_2 = tf.norm(x - x_gen, ord='euclidean', axis=1) / d_size
        # compute the L2 norm from feature layer
        _, f1 = self.discriminator([x, z], training=False)
        _, f2 = self.discriminator([x_gen, z], training=False)

        score_ge_1 = tf.norm(f1 - f2, ord='euclidean', axis=1) * kappa / f_size
        abnormal_score = score_ge_1 + score_ge_2

        return abnormal_score, score_ge_1, score_ge_2

    def train_fcn(self, args, x_train, x_ver_1, x_ver_2, x_ver_3):
        """
        Training function
        """
        start_time = time.time()
        num_of_data = x_train.shape[0]
        data_size = x_train.shape[1]
        for i in range(args.num_updates):
            real_data = x_train[np.random.permutation(num_of_data)[:args.batch_size]]
            real_data = tf.convert_to_tensor(real_data)
            for _ in range(args.interval):
                all_loss = self.train_step_dis(real_data)
                with self.writer.as_default():
                    tf.summary.scalar('all_loss', all_loss, step=self.all_idx)
                self.all_idx += 1

            # train generator and encoder
            ge_loss_1, ge_loss_2 = self.train_step_ge(real_data)

            self.ge_idx += 1
            with self.writer.as_default():
                tf.summary.scalar('ge_loss_1', ge_loss_1, step=self.ge_idx)
                tf.summary.scalar('ge_loss_2', ge_loss_2, step=self.ge_idx)
            print("\r[{}/{}]  ge_loss_1: {:.4}, ge_loss_2: {:.4}"
                  .format(self.ge_idx, args.num_updates, ge_loss_1, ge_loss_2), end="")

            if (self.ge_idx + 1) % args.check_point == 0:
                # Save the entire model as a SavedModel.
                if not os.path.exists('saved_model'):
                    os.makedirs('saved_model')
                self.generator.save('saved_model/generator.h5')
                self.encoder.save('saved_model/encoder.h5')
                self.discriminator.save('saved_model/discriminator.h5')

                # verification in training
                for t in range(10):
                    cycle = np.random.randint(0, num_of_data)
                    t_patch_0 = tf.convert_to_tensor(x_train[cycle])
                    t_patch_1 = tf.convert_to_tensor(x_ver_1[cycle])
                    t_patch_2 = tf.convert_to_tensor(x_ver_2[cycle])
                    t_patch_3 = tf.convert_to_tensor(x_ver_3[cycle])

                    r0, s_0_1, s_0_2 = self.verification_batch(t_patch_0, data_size, args.feature_size, args.kappa)
                    r1, s_1_1, s_1_2 = self.verification_batch(t_patch_1, data_size, args.feature_size, args.kappa)
                    r2, s_2_1, s_2_2 = self.verification_batch(t_patch_2, data_size, args.feature_size, args.kappa)
                    r3, s_3_1, s_3_2 = self.verification_batch(t_patch_3, data_size, args.feature_size, args.kappa)

                    with self.writer.as_default():
                        tf.summary.scalar('Validate_0_sensor_abnormal', r0, step=self.ver_idx)
                        tf.summary.scalar('Validate_0_sensor_abnormal', s_0_1, step=self.ver_idx)
                        tf.summary.scalar('Validate_0_sensor_abnormal', s_0_2, step=self.ver_idx)

                        tf.summary.scalar('Validate_1_sensors_abnormal', r1, step=self.ver_idx)
                        tf.summary.scalar('Validate_1_sensors_abnormal', s_1_1, step=self.ver_idx)
                        tf.summary.scalar('Validate_1_sensors_abnormal', s_1_2, step=self.ver_idx)

                        tf.summary.scalar('Validate_2_sensors_abnormal', r2, step=self.ver_idx)
                        tf.summary.scalar('Validate_2_sensors_abnormal', s_2_1, step=self.ver_idx)
                        tf.summary.scalar('Validate_2_sensors_abnormal', s_2_2, step=self.ver_idx)

                        tf.summary.scalar('Validate_3_sensors_abnormal', r3, step=self.ver_idx)
                        tf.summary.scalar('Validate_3_sensors_abnormal', s_3_1, step=self.ver_idx)
                        tf.summary.scalar('Validate_3_sensors_abnormal', s_3_2, step=self.ver_idx)
                    self.ver_idx += 1
        training_time = time.time() - start_time
        print('Training time:', training_time)


def main(vehicle, train):
    # set random seeds for reproducibility
    np.random.seed(10)
    tf.random.set_seed(10)

    # get hyper parameters
    args = get_args()

    # load data for training and testing
    x_train, _, _, x_ver_1, x_ver_2, x_ver_3 = prepare_data(vehicle)
    data_size = x_train.shape[1]

    # initialise the BiGan class
    bigan = BiGANAnomaly(vehicle,
                         train,
                         batch_size=args.batch_size,
                         data_size=data_size,
                         feature_size=args.feature_size,
                         latent_size=args.latent_size,
                         kappa=args.kappa,
                         ge_learn_rate=args.ge_learn_rate,
                         d_learn_rate=args.d_learn_rate)

    # train the model if train is True, otherwise load pre-trained model and test
    if train:
        bigan.train_fcn(args, x_train, x_ver_1, x_ver_2, x_ver_3)
    else:
        bigan.test()


if __name__ == "__main__":
    vessel = 'multi_vehicle'
    main(vessel, train=False)
