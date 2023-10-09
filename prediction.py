from tensorflow.keras.preprocessing.image import img_to_array
import imutils
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from load_and_process import contrastStretching, findFaces, generateIntensityImages
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/default_model/_mini_XCEPTION.102-0.66.hdf5'
# emotion_model_path = 'models/without_pseudo_mini_XCEPTION.102-0.75.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

with open("./data/custom_predictions.pickle", "rb") as f:
	xtest, ytest, pred, prob, f1, cm = pickle.load(f)
	f.close()

print(cm)
print(f1)

df_cm = pd.DataFrame(cm, index = [i for i in "0123456"],
				columns = [i for i in "0123456"])
plt.figure(figsize = (10,7))
plt.title("Confusion Matrix (Our method)")
sn.heatmap(df_cm, annot=True, fmt='.5g')
plt.show()

pred = []
prob = []
for i, image in enumerate(xtest):
	image = cv2.resize(image, (64, 64))
	grayFace = img_to_array(image)
	grayFace = np.expand_dims(grayFace, axis=0)
	
	preds = emotion_classifier.predict(grayFace)[0]
	emotion_probability = np.max(preds)
	label = preds.argmax()
	pred.append(label)
	prob.append(emotion_probability)

f1 = f1_score(ytest, pred, average='weighted')
cm = confusion_matrix(ytest, pred)

print(cm)
print(f1)

with open("./data/normal_predictions.pickle", "wb") as f:
	pickle.dump((xtest, ytest, pred, prob, f1, cm), f)
	f.close()

df_cm = pd.DataFrame(cm, index = [i for i in "0123456"],
				columns = [i for i in "0123456"])
plt.figure(figsize = (10,7))
plt.title("Confusion Matrix (Native method)")
sn.heatmap(df_cm, annot=True, fmt='.5g')
plt.show()