
This project was developed in Python version 3.5 environment, and work well on MacOS, 
But do not fully functional on Windows environment, due to a Python library issue.

Python packages used for the project are listed below:

threading
PyQt4
pyaudio
wave
tensorflow
tflearn
librosa

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Hot to run the demo:
In MacOS or Windows terminal:
type in

python musicrecommender.py

Note: In windows environment, the play function is not working, also we can only
load WAV type files (MP3 types is not supported)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This project contains files listed below:

1: Notebook files
project4-Getting Start.ipynb
Project4-Step1-Get music youtube url from www.last.fm.ipynb
Project4-Step2-Download music from Youtube.ipynb
Project4-Step3-Extract music features.ipynb
project4-Step4-Deep CNN for Contect-based music recommender system.ipynb

2. Datasets, located in ./music/MusicInfo folder
artists.dat
Step1_artists_with_youtube_url.csv
Step2_artists_with_youtube_info.csv
Step3_integrated_music_info.csv
tags.dat
user_taggedartists.dat

3: Music recommender UI and main program
musicrecommender.py
musicrecommender.ui
record.png

4: Trained Deep CNN model
content_music_recommender.tfl.data-00000-of-00001
content_music_recommender.tfl.index
content_music_recommender.tfl.meta

5: Supportive python program, for training and testing
traindnn.py
testdnn.py

6: Music features, located in ./music/MusicFeatures folder
About 7GB feature files
These folder contains the music features for cosine simiarity

