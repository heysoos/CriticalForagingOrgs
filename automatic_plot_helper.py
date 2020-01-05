from os import listdir
from os.path import isfile, join
import os
import sys
import numpy as np
import pickle

def detect_all_isings(sim_name):
    '''
    Creates iter_list
    Automatically detects the ising generations in the isings folder
    '''
    curdir = os.getcwd()
    mypath = curdir + '/save/{}/isings/'.format(sim_name)
    all_isings = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith('isings.pickle')]
    gen_nums = []
    for name in all_isings:
        i_begin = name.find('[') + 1
        i_end = name.find(']')
        gen_nums.append(int(name[i_begin:i_end]))
    gen_nums = np.sort(gen_nums)
    return gen_nums

def load_settings(loadfile):
    '''
    load settings from loadfile
    :param loadfile: simulation name
    '''
    curdir = os.getcwd()
    load_settings = '/save/{}/settings.pickle'.format(loadfile)
    settings = pickle.load(open(curdir + load_settings, 'rb'))
    return settings

def load_isings(loadfile):
    '''
    Load all isings pickle files and return them as list
    :param loadfile : simulation name
    '''
    iter_list = detect_all_isings(loadfile)
    settings = load_settings(loadfile)
    numAgents = settings['pop_size']
    isings_list = []
    for ii, iter in enumerate(iter_list):
        filename = 'save/' + loadfile + '/isings/gen[' + str(iter) + ']-isings.pickle'
        startstr = 'Loading simulation:' + filename
        print(startstr)

        try:
            isings = pickle.load(open(filename, 'rb'))
        except Exception:
            print("Error while loading %s. Skipped file" % filename)
            # Leads to the previous datapoint being drawn twice!!

        isings_list.append(isings)
    return isings_list
