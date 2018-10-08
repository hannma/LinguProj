import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from data_util import load_wav, log_specgram, list_all_files
from tensorflow.nn.rnn_cell import GRUCell, MultiRNNCell

class RNNModel:
    def __init__(self, batch_size=1):
        self.input_ph = tf.placeholder(tf.float32, shape=[batch_size, 221, 250])
        self.label_ph = tf.placeholder(tf.int32, shape=[batch_size,])
        self.batch_size=batch_size
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = tf.train.Saver()
    

    def preprocess(self):
        spectrogram = tf.expand_dims(self.input_ph, -1)
        tf.summary.image('spectrogram', spectrogram)
        print(spectrogram)

        one_hot = tf.one_hot(self.label_ph, 8, on_value=1.0, off_value=0.0)
        return spectrogram, one_hot

    # 1) build convolutional net
    def model(self, train_x, epochs=1, batch_size=1):
        print('trainx', train_x)

        '''
        with tf.variable_scope('CNN'):
            conv_net = tf.layers.conv2d(train_x, filters=64, kernel_size=(7, 7), strides=(1, 1), 
                                padding='SAME', activation=tf.nn.relu, name='Conv1')
            print(conv_net)

            conv_net = tf.layers.conv2d(conv_net, filters=128, kernel_size=(5, 5), strides=(2, 2), 
                                padding='SAME', name='Conv2')
            conv_net = tf.layers.batch_normalization(conv_net)
            conv_net = tf.nn.relu(conv_net)

            print(conv_net)

            conv_net = tf.layers.conv2d(conv_net, filters=256, kernel_size=(5, 5), strides=(2, 2), 
                                padding='SAME', name='Conv3')
            conv_net = tf.layers.batch_normalization(conv_net)
            conv_net = tf.nn.relu(conv_net)

            print(conv_net)

            conv_net = tf.layers.conv2d(conv_net, filters=512, kernel_size=(5, 5), strides=(2, 2), 
                                padding='SAME', name='Conv4')
            conv_net = tf.layers.batch_normalization(conv_net)
            conv_net = tf.nn.relu(conv_net)

            print(conv_net)

            conv_net = tf.layers.conv2d(conv_net, filters=1, kernel_size=(7, 7), strides=(2, 2), 
                                padding='SAME', name='Conv5')

            print(conv_net)

            conv_out = tf.contrib.layers.flatten(conv_net)
            conv_out = tf.layers.dense(conv_out, 250)
            print(conv_out)
        '''

        with tf.variable_scope('RNN'):

            #rnn_inp = tf.expand_dims(conv_out, -1)
            #print(rnn_inp)

            num_units = [64, 128, 256]
            cell = MultiRNNCell([GRUCell(n) for n in num_units])
            out, state = tf.nn.dynamic_rnn(cell, tf.squeeze(train_x, [-1]), dtype=tf.float32)

            fc_out = tf.layers.dense(state[-1], 1024, activation=tf.sigmoid)
            fc_out = tf.layers.dense(fc_out, 8, activation=tf.sigmoid)
            print(fc_out)
            return fc_out

    # 2) compare outcome with true labels
    def loss(self, predictions, train_y):
        raw_loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_y, logits=predictions)
        loss = tf.reduce_mean(raw_loss)
        tf.summary.scalar('loss', loss)

        return loss

    # 3) calculate gradient and optimize --> adapts weights
    def optimize(self, loss, learning_rate=1e-4):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

        return optimizer, self.global_step

    # 4) combine 1), 2) and 3)
    def build(self):
        train_x, train_y = self.preprocess()
        predictions = self.model(train_x)
        loss = self.loss(predictions, train_y)
        optimize, global_step = self.optimize(loss)
        
        summaries_op = tf.summary.merge_all()

        return optimize, loss, global_step, summaries_op



if __name__ == "__main__":

    if os.path.isdir('./records/summaries'):
        while True:
            user_input = input('Delete summaries? y/[n]\n')
            if user_input.lower() in ['y', 'yes', '1', 'true']:
                print('Deleting summary directory')
                os.system('rm -r ./records/summaries')
                break
            else:
                print('Keeping summary files')
                break


    batchsize = 3

    model = RNNModel(batchsize)
    optimize_op, loss_op, step_op, summ_op = model.build()

    epochs = 10
    N_SAMPLES = 1440


    ravdess = np.load('ravdess.npz')
    train_data = ravdess['data']
    train_data = np.reshape(train_data, (1440, 221, 250))
    train_labels = ravdess['labels']


    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('./records/summaries', sess.graph)

        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            shuffle_idx = np.random.permutation(N_SAMPLES)
            train_data = train_data[shuffle_idx]
            train_labels = train_labels[shuffle_idx]

            for smpl_nr in range(N_SAMPLES // batchsize) :

                train_x = train_data[smpl_nr:smpl_nr + batchsize]
                train_y = train_labels[smpl_nr:smpl_nr + batchsize]

                _, np_loss, step, summaries = sess.run([optimize_op, loss_op, step_op, summ_op], feed_dict={model.input_ph:train_x,
                                                            model.label_ph:train_y})

                sys.stdout.write('\rIteration {} : loss = {}'.format(step, np_loss))
                sys.stdout.flush()
                if step % 20 == 0:
                    save_path = model.saver.save(sess, './records/checkpoints/model')
                    summary_writer.add_summary(summaries, step)
                    print(' +++ Saved model to {} and wrote summaries. +++ '.format(save_path))
                

