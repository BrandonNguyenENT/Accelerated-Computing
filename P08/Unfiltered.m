file = 'FFTW_P08_out.wav';
[y,Fs] = audioread(file);
spectrogram(y,128,120,128,Fs)