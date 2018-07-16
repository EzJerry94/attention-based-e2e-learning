import numpy as np
import wave
import pylab as pl

def preprocess_stats(file, csv_name):
    lines = np.loadtxt(file,dtype='str',usecols=(0,2,3,4))
    print(lines)
    print(lines.shape)
    print(type(lines))
    for sample in lines:
        #arousal; valence; dominance
        arousal = sample[1]
        valence = sample[2]
        dominance = sample[3]
        sample[1] = change_stats_to_int(arousal)
        sample[2] = change_stats_to_int(valence)
        sample[3] = change_stats_to_int(dominance)
    lines[:, 1:].astype(int)
    print(lines)
    print(lines.shape)
    print(type(lines))
    np.savetxt('./data/'+csv_name, lines, fmt='%s %s %s %s')

def change_stats_to_int(attribute):
    if attribute == 'pos':
        attribute = 1
    elif attribute == 'neg':
        attribute = 2
    elif attribute == 'neu':
        attribute = 3
    return attribute

def show_wav(file):
    f = wave.open(file, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print(params)
    str_data = f.readframes(nframes)
    f.close()

    wave_data = np.fromstring(str_data, dtype=np.short)
    print(wave_data)
    time = np.arange(0, nframes) * (1.0 / framerate)
    pl.plot(time, wave_data)
    pl.show()