import tensorflow as tf
import numpy as np
from magenta.music import note_sequence_io
from magenta.protobuf import music_pb2

'''
sequences = []
for i in range(4):
    sequence = music_pb2.NoteSequence()
    sequence.id = str(i)
    sequence.notes.add().pitch = i
    sequences.append(sequence)
'''


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
        key_len = len(keys)
        qpm_dict = dict(time_qpm)

        '''
        time_qpm = []
        for tempo in sequence.tempos:
            time_qpm.append((tempo.time,tempo.qpm))
        curr_tempo_ptr = 0
        len_time_qpm = len(time_qpm)
        '''

        # two channels
        channel_0 = [[] for i in range(5)]
        channel_1 = [[] for i in range(5)]
        channel_flag = 0
        
        # keep track of start_time and the index of each list
        tracking_index = np.zeros((5,))
        tracking_start = 0

        prev_end_time = 0
        prev_start_time = 0
        prev_index = 0
        #counter = 0
        for note in sequence.notes:
            #print (counter)
            #counter = counter + 1
            # get current pitch value, start time, and end time
            curr_pitch = note.pitch
            #print(curr_pitch)
            curr_start_time = note.start_time
            curr_end_time = note.end_time
            duration = curr_end_time - curr_start_time

            # switch the channel if needed
            if prev_start_time > curr_start_time and prev_end_time == total_time:
                #print ('switched channel',counter)
                channel_flag = 1
                # move the pointer back to the beginning
                #curr_tempo_ptr = 0
                # clean up the tracking status
                tracking_index = np.zeros((5,))
                tracking_start = 0
                prev_end_time = 0
                prev_index = 0

            #f (prev_end_time > 6.75 and prev_end_time < 6.76):
                #print(prev_end_time, tracking_index, channel_flag)

            gap = curr_start_time - prev_end_time
            # if there is a gap, fill the gap with 0
            if gap > 0:
                #print(0)
                #print ('there is a gap')
                #gap_qpm = time_qpm[curr_tempo_ptr][1]
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
                #gap_n += int(round((curr_start_time - lower_bound)*gap_qpm))
                '''
                # get qpm
                key_index = (np.abs(qpm_keys-prev_end_time)).argmin()
                if qpm_keys[key_index] > prev_end_time:
                    key_index = key_index - 1
                gap_qpm = qpm_dict[qpm_keys[key_index]]
                #print(gap_qpm)
                gap_n = int(round(gap * gap_qpm))
                '''
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
                for j in range(5):
                    if channel_flag == 0:
                        channel_0[j].extend(new_pitches)
                    else:
                        channel_1[j].extend(new_pitches)
                    tracking_index[j] = tracking_index[j]+gap_n
                '''
                # update the tempo pointer (it must be incremented by 1 if it is not the last one)
                #if curr_tempo_ptr + 1 < len_time_qpm:
                #    curr_tempo_ptr = curr_tempo_ptr + 1

            prev_end_time = curr_end_time
            prev_start_time = curr_start_time

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
            #n += int(round((curr_end_time - lower_bound)*curr_qpm))
            # compute how many times we want it to repeat
            #curr_qpm = time_qpm[curr_tempo_ptr][1]
            #n = int(round(duration * curr_qpm))

            #if (prev_end_time > 6.7588 and prev_end_time < 6.7589):
                #print (tracking_index, prev_end_time, channel_flag, raw_n)

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
    
    
    # convert to 2d arrays 
    result_0 = np.array([np.array(track) for track in channel_0])
    result_1 = np.array([np.array(track) for track in channel_1])
    return result_0, result_1
            

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
