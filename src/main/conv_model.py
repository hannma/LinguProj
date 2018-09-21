import tensorflow as tf
#import keras
import numpy as np
import os
#from keras.models import Sequential
#from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from data_util import load_wav, log_specgram, list_all_files

class ConvModel:
    def __init__(self, batch_size=1):
        self.input_ph = tf.placeholder(tf.float32, shape=[None, None])
        self.label_ph = tf.placeholder(tf.int32, shape=[1,])
        self.batch_size=batch_size
        self.global_step = tf.Variable(0, trainable=False)
    

    def preprocess(self):
        spectrogram = tf.expand_dims(self.input_ph,0)
        spectrogram = tf.transpose(tf.expand_dims(spectrogram,-1), [0,2,1,3])
        tf.summary.image('Spectrogram', spectrogram)
        print(spectrogram)
        cropped_spectrogram = tf.image.resize_image_with_crop_or_pad(spectrogram, target_height=221, target_width=250)
        print(cropped_spectrogram)
        tf.summary.image('Cropped_Spec', cropped_spectrogram)

        one_hot = tf.one_hot(self.label_ph, 8, on_value=1.0, off_value=0.0)
        return cropped_spectrogram, one_hot

    # 1) build convolutional net
    def model(self, train_x, epochs=1, batch_size=1):
        print('\nTHE MODEL:')
        net = tf.layers.conv2d(train_x, filters=32, kernel_size=(7, 7), strides=(1, 1), 
                               padding='SAME', activation=tf.nn.relu)
        print(net)
        net = tf.layers.conv2d(net, filters=32, kernel_size=(3, 3), strides=(2, 2),
                               padding='SAME', activation=tf.nn.relu)
        print(net)
        net = tf.layers.max_pooling2d(net, pool_size=(2,2), strides=(2, 2))
        print(net)

        net = tf.layers.conv2d(net, filters=64,kernel_size=(3, 3), strides=(1, 1), 
                               padding='SAME', activation=tf.nn.relu)
        print(net)
        net = tf.layers.max_pooling2d(net, pool_size=(2,2), strides=(2, 2))
        print(net)
        # -1: choose whatever you want
        #  8: 8 emotion classes
        #net = tf.reshape(net, [-1, 8])

        net = tf.contrib.layers.flatten(net)
        print(net)
        # create fully connected layer (= dense layer) with 1024 nodes
        net = tf.layers.dense(net, 1024)
        print(net)
        net = tf.layers.dropout(net, rate=0.5, training=True)
        print(net)
        # now: net contains 8 predictions
        net = tf.layers.dense(net, 8)
        print(net)
        input('...')

        return net

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




        '''model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3),
                         activation='relu',
                         input_shape=[28, 28, 1]))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='softmax'))
        print(model)
        input('model')

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accurarcy'])

        model.fit(data, labels,
                  epochs=epochs,
                  bacht_size=batch_size,
                  verbose=1)'''
            
        


        


if __name__ == "__main__":
    list_files, _ = list_all_files()

    model = ConvModel()
    optimize_op, loss_op, step_op, summ_op = model.build()

    epochs = 10

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter('./records/summaries', sess.graph)

        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for file in list_files:
                #extract feature number by extracting emotion-label out of file-name which is the 3rd one
                #then drop zeros
                emotion_number = int(file.split('/')[-1].split('-')[2].lstrip('0'))

                samples, sr = load_wav(file)
                freqs, time, spectrogram = log_specgram(samples, sr)
                #print(np.shape(spectrogram))
                _, np_loss, step, summaries = sess.run([optimize_op, loss_op, step_op, summ_op], feed_dict={model.input_ph:spectrogram,
                                                            model.label_ph:[emotion_number]})

                print('iteration {} : loss = {}'.format(step, np_loss))
                if step % 10 == 0:
                    summary_writer.add_summary(summaries, step)
                

