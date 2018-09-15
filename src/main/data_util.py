import os
import glob

DATA_DIR = '../../data/Audio_Speech_Actors_01-24/'
assert os.path.isdir(DATA_DIR)

actor_folders = glob.glob(DATA_DIR + 'Actor_*')

