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
    file_name = "notes.tfrecord"
    sequence = music_pb2.NoteSequence()
    for i, sequence in enumerate(note_sequence_io.note_sequence_record_iterator(file_name)):
        print(i)
        print(sequence.id)
        print(sequence.filename)
        for note in sequence.notes:
            print(note.pitch)

if __name__ == "__main__":
    main()
