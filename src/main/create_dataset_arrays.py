import tensorflow as tf
#import keras
import numpy as np
import os
import sys
#from keras.models import Sequential
#from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
from data_util import load_wav, log_specgram, list_all_files

class DataModel:
    def __init__(self, batch_size=1):
        self.input_ph = tf.placeholder(tf.float32, shape=[None, None])
    

    def preprocess(self):
        spectrogram = tf.expand_dims(self.input_ph,0)
        spectrogramT = tf.transpose(tf.expand_dims(spectrogram,-1), [0,2,1,3])
        print(spectrogramT)
        spectrogram_added = spectrogramT + 100.
        cropped_spectrogram = tf.image.resize_image_with_crop_or_pad(spectrogram_added, target_height=221, target_width=250)
        print(cropped_spectrogram)
        cropped_spectrogram = tf.squeeze(cropped_spectrogram)
        print(cropped_spectrogram)
        # flattened_spectrogram = tf.reshape(cropped_spectrogram, [-1])
        # print(flattened_spectrogram)
        
        return cropped_spectrogram #flattened_spectrogram

    


if __name__ == "__main__":
    list_files, _ = list_all_files()

    model = DataModel()
    spec_op = model.preprocess()



    with tf.Session() as sess:

        dataset = []
        labels = []
        for i, file in enumerate(list_files):
            #extract feature number by extracting emotion-label out of file-name which is the 3rd one
            #then drop zeros

            samples, sr = load_wav(file)
            freqs, time, spectrogram = log_specgram(samples, sr)
            
            np_flattened_spec = sess.run(spec_op, feed_dict={model.input_ph:spectrogram})
            dataset.append(list(np_flattened_spec))
            emotion_number = int(file.split('/')[-1].split('-')[2].lstrip('0'))
            labels.append(emotion_number-1)
            sys.stdout.write('\rDone with file number {}'.format(i+1))
            sys.stdout.flush()

        np_dataset = np.asarray(dataset)
        np_labels = np.asarray(labels)

        data_min = np.min(np_dataset)
        print('Min:', data_min)
        data_max = np.max(np_dataset)
        print('Max:', data_max)

        normalized_data = (np_dataset - data_min) / (data_max - data_min)

        np.savez('ravdess', data=normalized_data, labels=np_labels)
        meaned_dataset = np.mean(np_dataset, 0)
        np.save('ravdess_mean', meaned_dataset)
        print(meaned_dataset)
        
               
                    

