from joblib import load
import soundfile as sf
import librosa
import os


if __name__ == '__main__':

    output_array = {'de': 'German', 'en': 'English', 'es': 'Espanol'}

    # import file
    path = os.path.join(os.getcwd(), 'Dataset/test')
    filename = 'de_f_63f5b79c76cf5a1a4bbd1c40f54b166e.fragment73.flac'
    flac, samplerate = sf.read(os.path.join(path, filename))

    model = load('model/model.joblib')
    print('Nama File\t:', filename)
    print('Kelas\t\t:', output_array[model.predict(librosa.feature.zero_crossing_rate(flac))[0]])
