import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from data_util import load_wav, log_specgram, list_all_files

class ConvModel:
    def __init__(self, batch_size=1):
        self.input_ph = tf.placeholder(tf.float32, shape=[250, 221])
        self.label_ph = tf.placeholder(tf.int32, shape=[1,])
        self.batch_size=batch_size
        self.global_step = tf.Variable(0, trainable=False)
        self.saver = tf.train.Saver()
    

    def preprocess(self):
        with tf.name_scope('preprocessing'):
            spectrogram = tf.expand_dims(self.input_ph,0)
            spectrogram = tf.transpose(tf.expand_dims(spectrogram,-1), [0,2,1,3])
            tf.summary.image('Spectrogram', spectrogram)
            tf.summary.histogram('Spec_Hist', spectrogram)
            print(spectrogram)

            one_hot = tf.one_hot(self.label_ph, 8, on_value=1.0, off_value=0.0)
            return spectrogram, one_hot

    # 1) build convolutional net
    def model(self, train_x, epochs=1, batch_size=1):
        print('\nTHE MODEL:')
        net = tf.layers.conv2d(train_x, filters=32, kernel_size=(7, 7), strides=(1, 1), 
                               padding='SAME', activation=tf.nn.relu, name='Conv1')
        print(net)
        net = tf.layers.conv2d(net, filters=32, kernel_size=(5, 5), strides=(2, 2),
                               padding='SAME', activation=tf.nn.relu, name='Conv2')
        print(net)
        net = tf.layers.conv2d(net, filters=64, kernel_size=(5, 5), strides=(1, 1),
                               padding='SAME', activation=tf.nn.relu, name='Conv3')
        print(net)

        net = tf.layers.conv2d(net, filters=64,kernel_size=(3, 3), strides=(2, 2), 
                               padding='SAME', activation=tf.nn.relu, name='Conv4')
        print(net)
        net = tf.layers.conv2d(net, filters=128,kernel_size=(3, 3), strides=(2, 2), 
                               padding='SAME', activation=tf.nn.relu, name='Conv5')
        print(net)
        # -1: choose whatever you want
        #  8: 8 emotion classes
        #net = tf.reshape(net, [-1, 8])

        net = tf.contrib.layers.flatten(net)
        print(net)
        # create fully connected layer (= dense layer) with 1024 nodes
        with tf.name_scope('Dense1'):
            net = tf.layers.dense(net, 1024)
            print(net)
            net = tf.layers.dropout(net, rate=0.5, training=True)
            print(net)
        # now: net contains 8 predictions
        net = tf.layers.dense(net, 8, name='Dense2')
        print(net)

        # Result of a forward pass
        return net

    # 2) compare outcome with true labels
    def loss(self, predictions, train_y):
        with tf.name_scope('loss'):
            raw_loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_y, logits=predictions)
            loss = tf.reduce_mean(raw_loss)
            tf.summary.scalar('loss', loss)

            return loss

    # 3) calculate gradient and optimize --> adapts weights
    def optimize(self, loss, learning_rate=1e-5):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

            return optimizer, self.global_step

    # 4) combine 1), 2) and 3)
    def build(self):
        train_x, train_y = self.preprocess()
        predictions = self.model(train_x)
        loss = self.loss(predictions, train_y)
        optimize, global_step = self.optimize(loss)
        
        summaries_op = tf.summary.merge_all()

        return optimize, loss, global_step, summaries_op, predictions, train_y



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


    model = ConvModel()
    optimize_op, loss_op, step_op, summ_op, pred_op, one_hot_op = model.build()
    softmax_op = tf.nn.softmax(pred_op)

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
            #shuffle_idx = np.random.permutation(N_SAMPLES)
            #train_data = train_data[shuffle_idx]
            #train_labels = train_labels[shuffle_idx]

            for smpl_nr in range(N_SAMPLES):

                train_x = train_data[smpl_nr].T
                train_y = train_labels[smpl_nr]

                _, np_loss, step, summaries, np_pred, np_onehot = sess.run([optimize_op, loss_op, step_op, summ_op, softmax_op, one_hot_op], feed_dict={model.input_ph:train_x,
                                                            model.label_ph:[train_y]})

                sys.stdout.write('\rIteration {} : loss = {}'.format(step, np_loss))
                sys.stdout.flush()
                
                if step % 25 == 0:
                    save_path = model.saver.save(sess, './records/checkpoints/model')
                    summary_writer.add_summary(summaries, step)
                    print(' +++ Saved model to {} and wrote summaries. +++ '.format(save_path))
                

