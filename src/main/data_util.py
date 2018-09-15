"""
Utilitiess to handle the RAVDESS dataset (https://smartlaboratory.org/ravdess/)

The dataset is structured as follows:

Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
Vocal channel (01 = speech, 02 = song).
Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the ‘neutral’ emotion.
Statement (01 = “Kids are talking by the door”, 02 = “Dogs are sitting by the door”).
Repetition (01 = 1st repetition, 02 = 2nd repetition).
Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

Filename example: 03-01-06-01-02-01-12.wav
"""


import os
import glob
import logging
logging.basicConfig(filename='../../log/events.log',level=logging.DEBUG)


DATA_DIR_RELPATH = '../../data' 

def list_all_files(path='../../data/Audio_Speech_Actors_01-24/',
                   write_file=True):
    
    if not os.path.isdir(path):
        msg = 'The specified directory does not exist: {}'
        raise FileNotFoundError(msg=msg.format(path))

    relevant_files_regex = 'Actor_*/03-01-*.wav'
    relevant_files_list = glob.glob(path + relevant_files_regex)

    dataset_fp = os.path.join(DATA_DIR_RELPATH, 'ravdess.dataset')

    if write_file and not os.path.isfile(dataset_fp):
        with open(dataset_fp, 'w') as f:
            for i in relevant_files_list:
                print(i, file=f)
        logging.info('Wrote filenames to file {}'.format(dataset_fp))

    return relevant_files_list