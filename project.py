import argparse
import os.path
import wave
from array import array
from sys import byteorder
from tkinter import *

import librosa
import numpy as np
import pandas as pd
import pyaudio
import pyttsx3 as tts
import speech_recognition as sr
from PIL import ImageTk
from keras import Sequential
from keras.layers import Dense, Dropout
from librosa import feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

root = Tk()
root.geometry('1000x970+100+30')
root.title('Voisy The Voice Based Gender Recognizer')
root.config(bg='light blue')

centreFrame = Frame(root)
centreFrame.pack()

textarea = Text(centreFrame, font=('times new roman', 20, 'bold'), height=10,wrap='word')
textarea.pack(side=LEFT)
def load_data():
    # loading dataset
    df = pd.read_csv('voice.csv')
    df.label = [1 if each == 'female' else 0 for each in df.label]
    y = df.label
    x = df.drop(['label'], axis='columns')
    np.save('features.npy', x)
    np.save('labels.npy', y)
    return x, y


def create_model(vector_length=128):
    model = Sequential()
    model.add(Dense(256, input_shape=(vector_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    # one output neuron with sigmoid activation function, 0 means female, 1 means male
    model.add(Dense(1, activation="sigmoid"))
    # using binary crossentropy as it's male/female classification (binary)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    # print summary of the model
    #model.summary()
    return model


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
    algo_names = []
    algo_scores = []

    rand_forest = RandomForestClassifier(random_state=50)
    rand_forest.fit(x_train, y_train)
    algo_names.append('Random Forest')
    algo_scores.append(rand_forest.score(x_test, y_test))
    print(algo_names, algo_scores)


record_seconds = 15
threshold = 500
wave_file = "demo.wav"
chunk_size = 1024
for_mat = pyaudio.paInt16
rate = 48000
silence = 30


def is_silent(snd_data):
    return max(snd_data) < threshold


def normalize(snd_data):
    maximun = 16384
    times = float(maximun) / max(abs(i) for i in snd_data)
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))

    return r


def trim(snd_data):
    def _trim(snd_data):
        snd_started = False
        r = array('h')
        for i in snd_data:
            if not snd_started and abs(i) > threshold:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)

        return r

    snd_data = _trim(snd_data)

    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    r = array('h', [0 for i in range(int(seconds * rate))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds * rate))])
    return r


def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=for_mat, channels=1, rate=rate, input=True, output=True, frames_per_buffer=chunk_size)
    num_silent = 0
    snd_started = False
    r = array('h')

    print("Recording...")

    for i in range(0, int(rate / chunk_size * record_seconds)):
        snd_data = stream.read(chunk_size)
        frames.append(snd_data)
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)
        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > silence:
            break

    print("Done recording...")

    sample_width = p.get_sample_size(for_mat)
    stream.stop_stream()
    stream.close()
    p.terminate()
    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_to_file(path):
    sample_width, data = record()
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


def extract_features(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    mel = kwargs.get("mel")

    x, sample_rate = librosa.core.load(file_name)
    result = np.array([])

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=x, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result

rec = sr.Recognizer()
sys = tts.init()
frames = []
x, y = load_data()
data = split_data(x, y)
model = create_model()
model.save("model.h5")


parser = argparse.ArgumentParser(description="Gender Recognition Using Voice Input From User")
parser.add_argument("-f", "--f", help="the path file preferred to be in WAV format")
args = parser.parse_args()
file = args.__init__()
model.load_weights("model.h5")

if not file or not os.path.isfile("demo.wav"):
    print("Please Talk")
    file = "demo.wav"


record_to_file(file)
features = extract_features(file, mel=True).reshape(1, -1)
male_prob = model.predict(features)[0][0]
female_prob = 1 - male_prob
gender = "male" if male_prob > female_prob else "female"

print("Result:", gender)


if gender =='female':
    voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
    sys.setProperty('voice', voice_id)
    genpic=ImageTk.PhotoImage(file='female.png')
    logoPicLabel = Label(root, image=genpic, bg='deep pink')
    logoPicLabel.pack()
else :
    genpic = ImageTk.PhotoImage(file='male.png')
    logoPicLabel = Label(root, image=genpic, bg='deep pink')
    logoPicLabel.pack()

print(f"Probabilities:      Male:{male_prob*100:.2f}%      Female:{female_prob*100:.2f}%")

audio = wave.open(file, "rb")
audio_frames = audio.readframes(-1)
smple_rate = audio.getframerate()
smple_width = audio.getsampwidth()
spt = sr.AudioData(audio_frames, smple_rate, smple_width)

text_from_speech = rec.recognize_google(spt, language='en-IN')
audio.close()
print(text_from_speech)

textarea.insert(END, "Hi I am Voisy\n You are a  :" + gender)
textarea.insert(END,'\nYou Said: \n'+ text_from_speech)
sys.say(text_from_speech)
sys.runAndWait()

root.mainloop()




