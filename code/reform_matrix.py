import numpy as np
from midi_to_statematrix import midiToNoteStateMatrix

import os
##### lowbound pitch :24
##### upperbound pitch: 102
PITCH_RANGE = 78
path = './dev/'


def append_zeros(d, max_len, num_file):
    result = np.zeros((num_file, max_len,2*PITCH_RANGE))
    ctr = 0
    for key, lst in d.iteritems():
        old_array = np.array(lst)
        result[ctr,0:old_array.shape[0],:] = old_array
        ctr += 1
    return result        

def main():
    entire_dict = {}
    max_len = 0
    file_count = 0
    for fname in os.listdir(path):
        file_count += 1
        name = fname[:-4]
        reformed_list = []
        state_matrix = midiToNoteStateMatrix(os.path.join(path, fname))
        print "Loaded {}".format(name)
        for state in state_matrix:
            sub_list = []
            for i in range(PITCH_RANGE):
                sub_list.append(state[i][0])
                sub_list.append(state[i][1])
            reformed_list.append(sub_list)
        print (np.array(reformed_list).shape[0])
        if np.array(reformed_list).shape[0] > max_len:
            max_len = np.array(reformed_list).shape[0]
        entire_dict[name] = reformed_list
    print (max_len)
    reformed_matrix = append_zeros(entire_dict, max_len, file_count)
    np.save('dev_set', reformed_matrix)
    print "Successfully build training set with {} examples".format(file_count)

if __name__ == "__main__":
    main()
