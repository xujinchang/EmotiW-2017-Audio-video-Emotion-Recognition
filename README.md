# EmotiW-2017-Audio-video-Emotion-Recognition
Method strategy for EmotiW 2017 video emotion recognition

# The first method

Using the pre-trained vgg-face and res10 finetuned on fer2013.After that, train the finetuned model on the AFEW train and predict on the AFEW validation.For per video, add all the frames' scores  and predict the label.Vgg-face achieves accuracy 35% and res10 achieves accuracy 33% while the combined model achieves accuracy 38%.However, when simply predict all the validation frames, vgg-face achieves accuracy 35.5% with fixed the conv lr\_mult and res10 achieves accuracy 32.7%.

# The second method

Extract the feature of the fc6 layer and fc7 layer seperately.Apply a one vs rest SVM to make classification.There are also two pre-processing data method.One is splitting all the train data into seven part.The other is splitting the train data based on the video id.Before concate the features,do PCA and l2\_norm. Using the SVM, we can increase the vgg-face accuracy from 35% to 40% and res10 accuracy from 32% to 37%.

# The third method

Using the feature of CNN as the input of LSTM unit. We set the time\_step to 128 and hidden layer to 128. A one LSTM layer can achieve the 40% accuracy.

# The fourth method

We use the C3D network which has the con3d kernel.It can connect the spatial and temporal information. We train the C3D network using the pre-trained sport1m model and the best model can achieve accuracy 41.1%.

# The final

We use all these models to predict the prob and add it with different weights and the accuracy can get 48.2%.

