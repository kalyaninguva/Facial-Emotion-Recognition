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

pseudoImages = []
pseudoLabels = []

base_data_path = "./data/Scraped Cultural Data/"
folders = os.listdir(base_data_path)
count = 0
for folder in folders:
	files = os.listdir(base_data_path + folder)
	for image in files:
		try:
			img = cv2.imread(base_data_path + folder + "/" + image)
		except cv2.error:
			continue

		try:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			for face in findFaces(img):
				grayFace = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
				grayFace = contrastStretching(grayFace)
				
				grayFace = grayFace.astype("float") / 255.0
				grayFace = img_to_array(grayFace)
				grayFace = np.expand_dims(grayFace, axis=0)
				
				preds = emotion_classifier.predict(grayFace)[0]
				emotion_probability = np.max(preds)
				label = preds.argmax()

				pseudoImages.append(grayFace)
				pseudoLabels.append(label)
				count += 1

				for generated in generateIntensityImages(grayFace):
					pseudoImages.append(generated)
					pseudoLabels.append(label)
					count += 1
		except:
			continue
	
		if count % 1000 == 0:
			print("Done %d"%count)

with open("./data/pseudo_data.pickle", "wb") as f:
	pickle.dump((pseudoImages, pseudoLabels), f)
	f.close()