import matplotlib.pyplot as plt
import streamlit as st
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset , random_split
import pandas as pd
import numpy as np
import pickle

class Config:
    def __init__(self,mode='conv',nfilt=26,nfeat=13,nfft=512,rate=16000):
        self.mode=mode
        self.nfilt=nfilt
        self.nfeat=nfeat
        self.rate=rate
        self.nfft=nfft
        self.step=int(rate/10)
        self.min=float('inf')   
        self.max=-float('inf')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def envelope(y,rate,threshold):
    mask=[]
    y=pd.Series(y).apply(np.abs)
    y_mean=y.rolling(window=int(rate/10),min_periods=1,center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else :
            mask.append(False)
    return mask

def calc_fft(y,rate):
    n=len(y)
    freq=np.fft.rfftfreq(n,d=1/rate)
    Y=abs(np.fft.rfft(y)/n)
    return(Y,freq)


class AudioClassifiers(nn.Module):
    def __init__(self,inputsize=13, *args, **kwargs) :
        super(AudioClassifiers,self).__init__(*args, **kwargs)
        self.Conv2D1=nn.Conv2d(inputsize,16,3,1,2).to(device=device)
        self.Relu1 = nn.ReLU().to(device=device).to(device).to(device=device)
        self.Conv2D2=nn.Conv2d(16,32,3,1,2).to(device=device)
        self.Relu2 = nn.ReLU().to(device=device)
        # self.Conv2D3=nn.Conv2d(32,64,3,1,2).to(device=device)
        # self.Relu3 = nn.ReLU().to(device=device)
        # self.Conv2D4=nn.Conv2d(64,128,3,1,2).to(device=device)
        # self.Relu4 = nn.ReLU().to(device=device)
        self.Maxpool=nn.MaxPool2d(2).to(device=device)
        self.Dropout=nn.Dropout(0.5).to(device=device)
        self.flatten=nn.Flatten(start_dim=1).to(device=device)
        # self.Denseex1=nn.Linear(17408,8192).to(device=device)
        # self.Reluex1 = nn.ReLU().to(device=device)
        # self.Denseex2=nn.Linear(8192,2048).to(device=device)
        # self.Reluex2 = nn.ReLU().to(device=device)
        self.Dense1=nn.Linear(2048,128).to(device=device)
        self.Relu5 = nn.ReLU().to(device=device)
        self.Dense2=nn.Linear(128,64).to(device=device)
        self.Relu6 = nn.ReLU().to(device=device)
        self.Dense3=nn.Linear(64,32).to(device=device)
        self.Relu7 = nn.ReLU().to(device=device)
        self.Dense4=nn.Linear(32,2).to(device=device)
        self.softmax = nn.Softmax(dim=1).to(device=device)
    def forward(self,msk,batch_size=1):
        msk=self.Conv2D1(msk)
        msk=self.Relu1(msk)
        msk=self.Conv2D2(msk)
        msk=self.Relu2(msk)
        # msk=self.Conv2D3(msk)
        # msk=self.Relu3(msk)
        # msk=self.Conv2D4(msk)
        # msk=self.Relu4(msk)
        msk=self.Maxpool(msk)

        msk=self.Dropout(msk)
        msk=self.flatten(msk)
        # print(msk.size())


        # msk=self.Denseex1(msk)
        # msk=self.Reluex1(msk)
        # msk=self.Denseex2(msk)
        # msk=self.Reluex2(msk)        
        msk=self.Dense1(msk)
        msk=self.Relu5(msk)       
        msk=self.Dense2(msk)
        msk=self.Relu6(msk)  
        msk=self.Dense3(msk)
        msk=self.Relu7(msk)
        msk=self.Dense4(msk)
        out=self.softmax(msk)
        return out      

model =AudioClassifiers()
model.load_state_dict(torch.load("audio_classifier_best.pth"))
model.eval()

with open("config.pkl", 'rb') as file:
    config = pickle.load(file)

pad2d = lambda a, i: a[:, 0: i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0],i - a.shape[1]),dtype=np.float32)))
def build_rand_feat(rate,wav):
    _min=config.min
    _max=config.max
    X=[]
    X_sample=librosa.feature.mfcc(y=wav,sr=rate,n_mels=config.nfilt,n_fft=config.nfft,n_mfcc=config.nfeat)
    X_sample=pad2d(X_sample,60)
    X_sample=librosa.power_to_db(X_sample)
    X.append(X_sample)
    X=np.array(X)
    X=(X-_min)/(_max-_min)
    if config.mode == 'conv':
        X=X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    elif config.mode == 'time':
        X=X.reshape(X.shape[0],X.shape[1],X.shape[2])
    return X

# Set Streamlit app title
st.title("Audio Analysis App")

# Upload audio file
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"],accept_multiple_files=True)
if audio_file:
    # Load audio data using Librosa
    for i in audio_file:
        signal, rate = librosa.load(i)
        st.header(i.name)
        st.audio(signal,sample_rate=rate)
        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(5, 10))
        mask = envelope(signal,rate,0.0005)
        signal=signal[mask]
        fft=calc_fft(signal,rate)
        logfbank = librosa.feature.melspectrogram(y=signal,sr=rate,n_mels=26,n_fft=512)
        fbank=librosa.power_to_db(logfbank)
        temp=librosa.feature.mfcc(y=signal,sr=rate,n_mels=128,n_fft=512,n_mfcc=13)
        mfcc=librosa.power_to_db(temp)
        ax[0].plot(signal)
        Y,freq=fft
        ax[1].plot(freq,Y)
        ax[2].imshow(fbank,cmap='hot',interpolation='nearest')
        ax[3].imshow(mfcc,cmap='hot',interpolation='nearest')
        # Display audio waveform plot
        
        st.pyplot(fig)
        
        sample=build_rand_feat(rate,signal)
        sample=torch.tensor(sample,dtype=torch.float32).to(device=device)
        output=model(sample)
        output.cpu()
        _, predicted = torch.max(output, 1)
        predicted=predicted.cpu()
        hsl=predicted.numpy()
        hslpre=hsl.sum()
        hslpre=hslpre/hsl.shape[0]
        hash=["happy","sad"]
        hslpre=round(hslpre)
        # Perform audio analysis and display text output
        st.subheader("Audio Analysis")
        st.write("the label is :",hash[hslpre] )

    # Add more audio analysis code here if desired

    # Add more visualizations or text output here based on the audio data