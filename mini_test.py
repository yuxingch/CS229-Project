import tensorflow as tf
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


def main():
    fp = open("pitch.txt","w+")
    fp.write('pitch,start_time,end_time,time_diff,note_len\n')
    file_name = "notes.tfrecord"
    sequence = music_pb2.NoteSequence()
    for i, sequence in enumerate(note_sequence_io.note_sequence_record_iterator(file_name)):
	for note in sequence.notes:
            fp.write('{},{},{},{},{}/{}\n'.format(note.pitch,note.start_time,note.end_time,note.end_time - note.start_time,note.numerator,note.denominator))
	
if __name__ == "__main__":
    main()
