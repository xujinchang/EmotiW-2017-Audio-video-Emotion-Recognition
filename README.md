# EmotiW-2017-Audio-video-Emotion-Recognition
Method strategy for EmotiW 2017 video emotion recognition
##The first method

Using the pre-trained vgg-face and res10 finetuned on fer2013.After that, train the finetuned model on the AFEW train and predict on the AFEW validation.For per video, add all the frames' scores  and predict the label.Vgg-face achieves accuracy 20% and res10 achieves accuracy 17% while the combined model achieves accuracy 18%.However, when simply predict all the validation frames, vgg-face achieves accuracy 35.5% with fixed the conv lr\_mult and res10 achieves accuracy 32.7%.

##The second method

Extract the feature of the fc6 layer and fc7 layer seperately.Apply a one vs rest SVM to make classification.There are also two pre-processing data method.One is splitting all the train data into seven part.The other is splitting the train data based on the video id.Before concate the features,do PCA and l2\_norm.
