import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import threading
import sys
import re
import wave
import numpy as np
import pandas as pd
import datetime
import pyaudio
import wave
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import traceback
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.phonon import Phonon
import librosa.display
import IPython.display
import matplotlib.pyplot as plt
import matplotlib.style as ms
ms.use('classic')

style_Mapping = ["rock", "metal", "pop", "punk", "folk", "hop", "black", "country"]

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

qtUIFile = "musicrecommender.ui" # Enter file here.
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtUIFile)

class MyPlayThread(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.runs = True
    
    def __del__(self):
        self.wait()
    
    def stop(self):
        self.runs = False

    def run(self):
        global music2play
        while self.runs:
            if music2play.isFinished():
                window.pushButton_play.setText("Play")
                self.stop()
                break;


def createModel(nbClasses=8, imageSize1=128, imageSize2=431):
    convnet = input_data(shape=[None, imageSize1, imageSize2, 1], name='input')

    convnet = conv_2d(convnet, 64, 2, activation='relu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation='relu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation='relu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 512, 2, activation='relu', weights_init="Xavier")
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, nbClasses, activation='softmax')
    convnet = regression(convnet, optimizer=optimizer, loss='categorical_crossentropy')
    #convnet = regression(convnet, optimizer='rmsprop', learning_rate=0.001, loss='categorical_crossentropy')
    music_model = tflearn.DNN(convnet)
    return music_model

def recordbuttonChecked():
    window.UploadButton.setEnabled(False)
    
    window.RecordButton.setEnabled(True)


def loadfilebuttonChecked():
    window.UploadButton.setEnabled(True)
    
    window.RecordButton.setEnabled(False)

def Record(CHUNK=1024, FORMAT=pyaudio.paInt16, CHANNELS=2, RATE=44100, RECORD_SECONDS=30,WAVE_OUTPUT_FILENAME="record.wav"):
    #start to record
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,channels=CHANNELS,rate=RATE,input=True,frames_per_buffer=CHUNK)
    frames = []
    totalprogress = int(RATE / CHUNK)
    j = 0; k = 0
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.append(data)
        j += 1
        if j == totalprogress:
            j = 0; k += 1
            window.progressBar.setValue(k)
            QtGui.QApplication.processEvents()

    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    # record is done

def extract_test_feature(fn, offset=5, duration=20):
    sound_clip, sr = librosa.load(fn, sr=11025, offset=5, duration=20)
    melspec = librosa.feature.melspectrogram(y=sound_clip, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    return melspec

def load_and_show_wave():
    global filename
    y, sr = librosa.load(filename, sr=11025, offset=5, duration=20)
    # extract mel spectrogram feature
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
    # convert to log scale
    logmelspec = librosa.power_to_db(melspec)
    plt.figure(figsize=(9, 4))
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    librosa.display.specshow(logmelspec, sr = sr, x_axis = 'time', y_axis = 'mel')
    plt.tight_layout()
    plt.savefig('waveform.png', dpi=100, bbox_inches='tight')
    #display
    png=QtGui.QPixmap('waveform.png')
    window.FigureShow.setPixmap(png)

#########################added by Jay Zong#############start
def getMostSimilarFiveMusic(input_styleID, input_features):

    #the path of the file of music's other infomation
    music_info_dir = './music/MusicInfo/'
    #the path where the extracted music feature files will be saved
    features_files_dir = './music/MusicFeatures/'
    #the path of downloaded music
    music_files_dir = './music/MusicDownload/'
    #set a special ID for the input music
    input_musicID = '00000000000'
    #the separator of the path
    separator = '/'

    #read the file in which the music information is stored
    df_music_info = pd.read_csv(music_info_dir + 'Step3_integrated_music_info.csv', index_col=0)
    #get the music information that the style is as same as the input music's style
    df_music_info = df_music_info[df_music_info.styleID==input_styleID]
    #from music information, get the music features file names
    music_feature_files = df_music_info['musicFeaturesFileName'].tolist()

    #read all music features that have the same style with input music into a list
    features_list = []
    #at the same time, create a youtubeID list which has same length with music features list
    youtubeID_list = []
    for i in range(0,len(music_feature_files)):
        youtubeID = music_feature_files[i][-15:-4]
        
        #if the file doesn't exist, skip this file and load next file
        features = np.load(features_files_dir + music_feature_files[i])

        #in features, there are several arrays of the melspec data (one array per 20 seconds) are stored.
        #in one array (for 20 seconds), there is a 2D array (128 rows x 431 columns)
        #so for one music, there is a 3D array (n x 128 x 431).

        #in order to use cosine_similarity.cosine_similarity, 3D array needs be converted to 2D array
        #the below codes convert 3D array (n x 128 x 431) for each music to 2D array (n x 55168)
        for j in range(0,len(features)):
            youtubeID_list.append(youtubeID)
            #in one array (for 20 seconds), there is a 2D array (128 rows X 431 columns)
            #convert the 2D array to a 1D array (55168)
            feature=[]
            for k in range(0,len(features[j])):
                feature.extend(features[j][k])
            features_list.append(feature)
                
    #convert 3D array (n x 128 x 431) to 2D array (n x 55168) for the input music
    #and append the data of input music to above music features list
    for j in range(0,len(input_features)):
        youtubeID_list.append(input_musicID)
        input_feature=[]
        for k in range(0,len(input_features[j])):
            input_feature.extend(input_features[j][k])
        features_list.append(input_feature)

    #calculate cosine similarity in the music features list (include all same style music features and input music features)
    features_cs = cosine_similarity(features_list)
    #set youtubeIDs as columns and indexes 
    df_features_cs = pd.DataFrame(features_cs, columns=youtubeID_list)
    df_features_cs.index = df_features_cs.columns

    #get 5 youtubeIDs that the mean cosine similarity are highest with input music
    df_features_cs_with_input_music = df_features_cs[input_musicID].reset_index()
    df_features_cs_with_input_music.rename(index=str, columns={'index': 'youtubeID'},inplace=True)
    df_features_cs_with_input_music = df_features_cs_with_input_music.groupby('youtubeID').mean()
    df_features_cs_with_input_music.drop(index=input_musicID,inplace=True)
    recommanded_IDs = df_features_cs_with_input_music.mean(axis=1).sort_values(ascending=False).head(5).index

    recommanded_music_info=pd.DataFrame()
    for i in range(0,len(recommanded_IDs)):
        recommanded_music_info = recommanded_music_info.append(df_music_info[['youtubeID','title','artist','view_count','average_rating','like_count',
                                                                              'dislike_count']][df_music_info['youtubeID'] == recommanded_IDs[i]])
    return recommanded_music_info
#########################added by Jay Zong#############end

class MyApp(QtGui.QMainWindow, Ui_MainWindow):

    def recordwave(self):
        #function to record a wave
        global filename
        global music2play
        # call Record function
        # Show a message box
        result = QMessageBox.question(self, 'Message', "Ready to start record?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if result == QMessageBox.No:
            return

        record_second = int(window.SelectTimeDisplay.value())
        dt = datetime.datetime.now().strftime("%I_%M %d_%m_%Y")
        filename=dt+".wav"
        Record(RECORD_SECONDS=record_second, WAVE_OUTPUT_FILENAME=filename)
        #make it playable
        music2play = QSound(filename)
        # after record , process wav file and display wave
        load_and_show_wave()

    def Play(self):
        global filename
        global music2play
        if bool(filename):
            if self.pushButton_play.text() == "Play":
                self.MyThread = MyPlayThread()
                self.MyThread.start()
                self.pushButton_play.setText("Stop")
                music2play.play();
            else:
                self.pushButton_play.setText("Play")
                music2play.stop();

    def Search(self):
        #search button enabled after load a file or record
        global music_model
        global filename
        if filename=="":
            self.TagShow.setText('----Nothing to search, please record or load a file----')
            return
        self.TagShow.setText("Please wait...")
        #clear RecommendTable
        self.model = QtGui.QStandardItemModel(self.RecommendTable)
        self.model.setRowCount(0)  
        self.model.setColumnCount(0) 
        #set table header
        self.RecommendTable.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)

        self.RecommendTable.setModel(self.model)
        QtGui.QApplication.processEvents()

        mel = extract_test_feature(filename, offset=5, duration=20)
        X_test = np.array(mel).reshape(-1, 128, 431, 1)
        pred_result = music_model.predict(X_test)
        print (pred_result)
        r_max = -1.0
        predicted_result = 0
        for j in range(len(pred_result[0])):
            if pred_result[0][j] > r_max:
                r_max = pred_result[0][j]
                predicted_result = j

        print("predicted result : %d" % predicted_result)
        self.TagShow.setText(style_Mapping[predicted_result])
        QtGui.QApplication.processEvents()

#########################added by Jay Zong#############start
        #########please keep below codes#################################
        #get the five most silimar music information
        df_musics = getMostSimilarFiveMusic(predicted_result, [mel]) 
        
        #set table header
        self.model = QtGui.QStandardItemModel(self.RecommendTable)
        self.model.setRowCount(5)  
        self.model.setColumnCount(6) 
        self.model.setHeaderData(0,QtCore.Qt.Horizontal,'Title')
        self.model.setHeaderData(1,QtCore.Qt.Horizontal,'Artist')
        self.model.setHeaderData(2,QtCore.Qt.Horizontal,'View')
        self.model.setHeaderData(3,QtCore.Qt.Horizontal,'Like')
        self.model.setHeaderData(4,QtCore.Qt.Horizontal,'Dislike')      
        self.model.setHeaderData(5,QtCore.Qt.Horizontal,'Rating')
        self.RecommendTable.setModel(self.model)
         
 
        #set the width of table columns
        self.RecommendTable.setColumnWidth(0,300)
        self.RecommendTable.setColumnWidth(1,150)
        self.RecommendTable.setColumnWidth(2,90)
        self.RecommendTable.setColumnWidth(3,90)
        self.RecommendTable.setColumnWidth(4,90)
        self.RecommendTable.setColumnWidth(5,100)
        self.RecommendTable.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignCenter)
        
        #set the table to uneditable
        self.RecommendTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        
        #set the table's content
        for i in range(0,len(df_musics)):
            self.model.setItem(i,0,QtGui.QStandardItem(df_musics['title'].tolist()[i]))
            self.model.setItem(i,1,QtGui.QStandardItem(str(df_musics['artist'].tolist()[i])))
            self.model.setItem(i,2,QtGui.QStandardItem(str(df_musics['view_count'].tolist()[i])))
            self.model.setItem(i,3,QtGui.QStandardItem(str(df_musics['like_count'].tolist()[i])))
            self.model.setItem(i,4,QtGui.QStandardItem(str(df_musics['dislike_count'].tolist()[i])))
            self.model.setItem(i,5,QtGui.QStandardItem(str(df_musics['average_rating'].tolist()[i])))
            
        self.RecommendTable.setModel(self.model)
#########################added by Jay Zong#############end

    def loadFile(self):
        global filename
        global music2play
        filter = "Music files (*.mp3 *.wav)"
        file_name = QFileDialog.getOpenFileName(self,"Open music file","./",filter)
        if file_name=="":
            self.UploadPath.setText("")
            return
        #check length of music
        y, sr = librosa.load(file_name, sr=None)
        music_length = int(round(librosa.get_duration(y=y, sr=sr)))
        if (music_length<30):
            self.UploadPath.setText("-------Music file is too short, need at least 30 seconds-------")
            filename=""
            return
        else:
            self.UploadPath.setText(file_name)
            filename = file_name

        music2play = QSound(filename)
        #call a function to process file_name
        load_and_show_wave()

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.setFixedSize(self.size())
        self.RecordRadio.clicked.connect(recordbuttonChecked)   # Record enabled
        self.UploadRadio.clicked.connect(loadfilebuttonChecked) # upload enabled
        self.UploadButton.clicked.connect(self.loadFile)        # upload button
        self.RecordButton.clicked.connect(self.recordwave)      # record button
        self.SearchButton.clicked.connect(self.Search)          # search button
        self.pushButton_play.clicked.connect(self.Play)         # record button

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    #init after window shows
    recordbuttonChecked()
    window.progressBar.setRange(0,30)
    filename = ""
    music_model = createModel()
######################### Open the line below once model is trained #############
    music_model.load("content_music_recommender.tfl")
    #end of init
    sys.exit(app.exec_())
