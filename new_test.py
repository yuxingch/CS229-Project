import tensorflow as tf
import numpy as np
from magenta.music import note_sequence_io
from magenta.protobuf import music_pb2


def process_tempo_index(qpm_dict, keys):
    timestart_to_qpm_idx = {}
    prev_idx = 0
    for i in range(len(keys)-1):
        start_time = keys[i]
        end_time = keys[i+1]
        idx_inc = (end_time - start_time) * qpm_dict[start_time]
        if abs(idx_inc - round(idx_inc)) > 0.01:
            print(idx_inc)
        timestart_to_qpm_idx[start_time] = (qpm_dict[start_time], int(round(prev_idx)))
        prev_idx += idx_inc
    end_idx = prev_idx
    timestart_to_qpm_idx[keys[-1]] = (qpm_dict[keys[-1]], int(round(end_idx)))

    return timestart_to_qpm_idx, int(round(end_idx))

def store_in_array(file):
    '''
    Data structure: 
        First store the pitches in channel_0 in a list of lists. Each sublist represents a time track.
        Then store the pitches in channel_1 in a list of lists. Each sublist represents a time track.
        Finally, convert each of them to a 2D numpy array of shape (H, W). Here we set 
            W corresponds to the length.
            H corresponds to the max num of different pitches at a time.
        *** REMINDER: SET H AS CONSTANT 5 ***
    Rules:
        0. # of repeated pitch is determined by (end_time-start_time)/(1/qpm)
        1. If more than one note occurs at the same time: record in different time track.
        2. Gap: set the pitch value as 0.
    '''
    sequence = music_pb2.NoteSequence()
    for i, sequence in enumerate(note_sequence_io.note_sequence_record_iterator(file)):
        '''
        * create a list of tuples, which stores the values of time and tempo:
            the first element in each tuple: tempo.time
            the second element in each tuple: tempo.qpm
        ** let curr_tempo_ptr point to the first element in the list
        '''
        total_time = sequence.total_time
        #print(total_time)
        
        # use a dictionray to store the relationship between time and qpm
        # use a numpy array to store the keys
        time_qpm = []
        keys = []
        for tempo in sequence.tempos:
            time_qpm.append((tempo.time,tempo.qpm))
            keys.append(tempo.time)
        qpm_keys = np.array(keys)
        #key_len = len(keys)
        qpm_dict = dict(time_qpm)

        # construct dictionary and two channel arrays
        my_dict, end_len = process_tempo_index(qpm_dict, keys)
        channel_0 = np.zeros((5,end_len))
        channel_1 = np.zeros((5,end_len))
        channel_flag = 0
        # construct an array to keep track of available slots
        available_slot = np.zeros((end_len,))
        '''
        # two channels
        channel_0 = [[] for i in range(5)]
        channel_1 = [[] for i in range(5)]
        '''
        prev_end_time = 0
        prev_start_time = 0
        count = 0
        for note in sequence.notes:
            print(count)
            count += 1
            # get current pitch value, start time, and end time
            curr_pitch = note.pitch
            curr_start_time = note.start_time
            curr_end_time = note.end_time

            # switch the channel if needed
            if prev_start_time > curr_start_time and prev_end_time == total_time:
                channel_flag = 1
                # clean up the tracking status
                prev_end_time = 0
                available_slot = np.zeros((end_len,))
            '''
            gap = curr_start_time - prev_end_time
            # if there is a gap, fill the gap with 0
            if gap > 0:
                key_index = (np.abs(qpm_keys - prev_end_time)).argmin()
                if qpm_keys[key_index] > prev_end_time:
                    key_index = key_index - 1
                # sum over intervals
                raw_gap_n = 0
                lower_bound = prev_end_time
                while curr_start_time > qpm_keys[key_index+1]:
                    upper_bound = qpm_keys[key_index+1]
                    gap_qpm = qpm_dict[qpm_keys[key_index]]
                    #gap_n += int(round((upper_bound - lower_bound) * gap_qpm))
                    raw_gap_n += (upper_bound - lower_bound) * gap_qpm
                    lower_bound = upper_bound
                    key_index += 1
                gap_qpm = qpm_dict[qpm_keys[key_index]]
                raw_gap_n += (curr_start_time - lower_bound)*gap_qpm
                gap_n = int(round(prev_index+raw_gap_n) - round(prev_index))
                prev_index += raw_gap_n

                # `start_time` has been changed, which is now equal to prev_end_time
                tracking_start = prev_end_time
                max_prev_len = np.amax(tracking_index)
                for j in range(5):
                    #channel_0[j] = channel_0[j] + [0] * (max_prev_len - tracking_index[j])
                    pad = [0 for _ in range(int(max_prev_len - tracking_index[j]))]
                    if channel_flag == 0:
                        channel_0[j].extend(pad)
                    else:
                        channel_1[j].extend(pad)
                    tracking_index[j] = max_prev_len
                # append new pitch, which is 0 here, to each row
                new_pitches = [0 for _ in range(gap_n)]
                if channel_flag == 0:
                    channel_0[0].extend(new_pitches)
                else:
                    channel_1[0].extend(new_pitches)
                tracking_index[0] = tracking_index[0]+gap_n
                '''
                # update the tempo pointer (it must be incremented by 1 if it is not the last one)
                #if curr_tempo_ptr + 1 < len_time_qpm:
                #    curr_tempo_ptr = curr_tempo_ptr + 1

            prev_end_time = curr_end_time
            prev_start_time = curr_start_time
            
            '''
            # get qpm
            key_index = (np.abs(qpm_keys-curr_start_time)).argmin()
            if qpm_keys[key_index] > curr_start_time:
                key_index = key_index - 1
            # sum over intervals
            n = 0
            raw_n = 0
            lower_bound = curr_start_time
            while curr_end_time > qpm_keys[key_index+1]:
                upper_bound = qpm_keys[key_index+1]
                curr_qpm = qpm_dict[qpm_keys[key_index]]
                raw_n += (upper_bound - lower_bound) * curr_qpm
                #n += int(round((upper_bound - lower_bound) * curr_qpm))
                lower_bound = upper_bound
                key_index += 1
            curr_qpm = qpm_dict[qpm_keys[key_index]]
            raw_n += (curr_end_time - lower_bound)*curr_qpm
            n = int(round(prev_index + raw_n) - round(prev_index))
            prev_index += raw_n
            '''

            # get the starting and ending key indices
            start_ind = (np.abs(qpm_keys-curr_start_time)).argmin()
            if (qpm_keys[start_ind]) > curr_start_time:
                start_ind = start_ind - 1
            end_ind = (np.abs(qpm_keys-curr_end_time)).argmin()
            if (qpm_keys[end_ind]) > curr_end_time:
                end_ind = end_ind - 1
            # get the starting and ending position
            start_key = qpm_keys[start_ind]
            start_base = my_dict[start_key][1]
            start_pos = int(start_base + round((curr_start_time - start_key)*qpm_dict[start_key]))
            end_key = qpm_keys[end_ind]
            end_base = my_dict[end_key][1]
            end_pos = int(end_base + round((curr_end_time - end_key)*qpm_dict[end_key]))

            # update values
            for idx in range(start_pos, min(end_pos,end_len)):
                row = int(available_slot[idx])
                if channel_flag == 0:
                    channel_0[row,idx] = curr_pitch
                    available_slot[idx] = row + 1
                else:
                    channel_1[row,idx] = curr_pitch
                    available_slot[idx] = row + 1
            '''
            # if start_time is unchanged for this note
            if curr_start_time == tracking_start:
                row_range = 5
                for i in range(n):
                    row = np.argmin(tracking_index[:row_range])
                    ids = tracking_index[row]
                    if channel_flag == 0:
                        channel_0[row].append(curr_pitch)
                    else:
                        channel_1[row].append(curr_pitch)
                    tracking_index[row] = ids + 1
                    row_range = row + 1
            # if start_time is changed
            else:
                # padding the old lists
                tracking_start = curr_start_time
                max_prev_len = np.amax(tracking_index)
                for j in range(5):
                    pad = [0 for _ in range(int(max_prev_len - tracking_index[j]))]
                    if channel_flag == 0:
                        channel_0[j].extend(pad)
                    else:
                        channel_1[j].extend(pad)
                    tracking_index[j] = max_prev_len
                # append new pitch to the first row
                new_pitches = [curr_pitch for _ in range(n)]
                if channel_flag == 0:
                    channel_0[0].extend(new_pitches)
                else:
                    channel_1[0].extend(new_pitches)
                tracking_index[0] = tracking_index[0] + n

            if (curr_start_time > 6.7588 and curr_start_time < 6.759 and channel_flag == 1):
                print (tracking_index, prev_end_time, channel_flag, raw_n,n,prev_index)
            if (curr_start_time > 6.7588 and curr_start_time < 7.88 and channel_flag == 0):
                print (tracking_index, prev_end_time, channel_flag, raw_n,n,prev_index)
            # update the pointer
            #if curr_tempo_ptr+1 < len_time_qpm:
                #next_time = time_qpm[curr_tempo_ptr+1][0]
                #if next_time <= curr_end_time:
                    #curr_tempo_ptr = curr_tempo_ptr + 1
            '''
    '''
    # check if we need extra padding
    channel0_len = np.zeros((5,))
    channel1_len = np.zeros((5,))
    for i in range(5):
        channel0_len[i] = len(channel_0[i])
        channel1_len[i] = len(channel_1[i])
    max0 = np.amax(channel0_len)
    max1 = np.amax(channel1_len)
    for j in range(5):
        pad0 = [0 for _ in range(int(max0 - channel0_len[j]))]
        pad1 = [0 for _ in range(int(max1 - channel1_len[j]))]
        channel_0[j].extend(pad0)
        channel_1[j].extend(pad1)
    
    print(len(channel_0[0]))
    print(len(channel_0[1]))
    print(len(channel_0[2]))
    print(len(channel_0[3]))
    print(len(channel_0[4]))
    print(len(channel_1[0]))
    print(len(channel_1[1]))
    print(len(channel_1[2]))
    print(len(channel_1[3]))
    print(len(channel_1[4]))
    '''
    '''
    # convert to 2d arrays 
    result_0 = np.array([np.array(track) for track in channel_0])
    result_1 = np.array([np.array(track) for track in channel_1])
    return result_0, result_1
    '''
    print(channel_0.shape,channel_1.shape)
    return channel_0, channel_1
            

def write_to_file(file):
    fp = open("pitch_mond2.txt","w+")
    ft = open("tempo_mond2.txt","w+")
    fp.write('pitch,start_time,end_time,time_diff,note_len\n')
    sequence = music_pb2.NoteSequence()
    for i, sequence in enumerate(note_sequence_io.note_sequence_record_iterator(file)):
        for tempo in sequence.tempos:
            ft.write('time:{}\n'.format(tempo.time))
            ft.write('qpm:{}\n'.format(tempo.qpm))
        for note in sequence.notes:
            fp.write('{},{},{},{},{}/{}\n'.format(note.pitch,note.start_time,note.end_time,note.end_time - note.start_time,note.numerator,note.denominator))

def main():
    file_name = "notes.tfrecord"
    #write_to_file(file_name)
    channel0, channel1 = store_in_array(file_name)
    np.save('channel0',channel0)
    np.save('channel1',channel1)

if __name__ == "__main__":
    main()
