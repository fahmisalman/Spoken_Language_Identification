from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from joblib import dump

from keras import Sequential, models
from keras.layers import LSTM, Dense, Flatten, GRU

import soundfile as sf
import pandas as pd
import numpy as np
import librosa
import os


def one_hot_encode(x):
    return label_binarizer.transform(x)


def train(x, y, epoch=20, hidden=256):
    model = Sequential()
    model.add(GRU(units=hidden, input_shape=(x.shape[1], 1), return_sequences=True))
    model.add(GRU(units=128))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(x, y, epochs=epoch, batch_size=64, verbose=1)
    return model


def evaluate(x, y, model):
    mse = model.evaluate(x, y, verbose=0)
    return mse


def save_model(model, s):
    model_json = model.to_json()
    with open("model/%s.json" % s, "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model/%s.h5" % s)


def load_model(s):
    model_json = open('model/%s.json' % s, 'r').read()
    model = models.model_from_json(model_json)
    model.load_weights("model/%s.h5" % s)
    return model


def predict(x, model):
    return model.predict(x)


if __name__ == '__main__':

    # Load Training Data
    print('Data Training')
    train_path = 'dataset/train/'

    series = []
    length = []
    for filename in os.listdir(train_path):
        flac, samplerate = sf.read(train_path + filename)
        series.append(flac)
        length.append(samplerate)

    label = []
    for filename in os.listdir(train_path):
        label.append(filename[:2])

    data = {'series': series,
            'languange': label}

    df = pd.DataFrame(data)
    print(df.head())

    # Load Testing Data
    print('\nData Testing')
    test_path = 'dataset/test/'

    series = []
    length = []
    for filename in os.listdir(test_path):
        flac, samplerate = sf.read(test_path + filename)
        series.append(flac)
        length.append(samplerate)

    label = []
    for filename in os.listdir(test_path):
        label.append(filename[:2])

    data_test = {'series': series,
                 'languange': label}

    df_test = pd.DataFrame(data_test)
    print(df_test.head())

    # Data Preprocessing
    print('Data preprocessing . . .')
    x = np.array(df['series'])
    x_t = np.array(df_test['series'])
    y_train = df['languange']
    y_test = df_test['languange']

    # Convert Pandas DataFrame into Numpy array
    x_train = np.zeros((len(x), 431))
    for i in range(len(x_train)):
        a = librosa.feature.zero_crossing_rate(x[i])
        x_train[i] = a

    x_test = np.zeros((len(x_t), 431))
    for i in range(len(x_test)):
        a = librosa.feature.zero_crossing_rate(x_t[i])
        x_test[i] = a

    # Convert label into one hot encoding
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(y_train)
    y_train_onehot = one_hot_encode(y_train)

    label_binarizer.fit(y_test)
    y_test_onehot = one_hot_encode(y_test)

    print('------------------------------\nClassification')
    print('Naive Bayes')
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    print('Train:', clf.score(x_train, y_train))
    print('Test :', clf.score(x_test, y_test))

    print('\nNeural Network')
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(10), random_state=1)

    clf.fit(x_train, y_train_onehot)
    print('Train:', clf.score(x_train, y_train_onehot))
    print('Test :', clf.score(x_test, y_test_onehot))

    print('\nk-Nearest Neighbors')
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(x_train, y_train)
    print('Train:', clf.score(x_train, y_train))
    print('Test :', clf.score(x_test, y_test))

    print('\nSupport Vector Machine')
    clf = SVC()
    clf.fit(x_train, y_train)
    print('Train:', clf.score(x_train, y_train))
    print('Test :', clf.score(x_test, y_test))

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    model = train(x_train, y_train_onehot, epoch=100)
    print(evaluate(x_test, y_test_onehot, model))
    save_model(model, 'model')