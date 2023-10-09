import pandas as pd
import cv2
import numpy as np
import face_recognition
import pickle
import os

dataset_path = './data/fer2013.csv'
image_size=(48,48)

def load_fer2013(custom=True, pseudo=True):
		data = pd.read_csv(dataset_path)
		pixels = data['pixels'].tolist()
		width, height = 48, 48
		faces = []
		for pixel_sequence in pixels:
			face = [int(pixel) for pixel in pixel_sequence.split(' ')]
			face = np.asarray(face).reshape(width, height)
			face = cv2.resize(face.astype('uint8'),image_size)
			faces.append(face)

		if custom:
			processed = []
			processedLabels = []
			for i, image in enumerate(faces):
				for face in findFaces(image):
					face = contrastStretching(face)
					processed.append(face)
					processedLabels.append(data['emotion'][i])

					for generated in generateIntensityImages(face):
						processed.append(generated)
						processedLabels.append(data['emotion'][i])
			
			if pseudo:
				pseudo_images, pseudo_labels = pseudoLabelledData()
				processed.extend(pseudo_images)
				processedLabels.extend(pseudo_labels)
		else:
			processed = faces
			processedLabels = data['emotion']

		faces = np.asarray(processed)
		faces = np.expand_dims(faces, -1)
		emotions = pd.get_dummies(processedLabels).as_matrix()
		return faces, emotions

# Preprocessing Functions
def contrastStretching(image):
	# Implement contrast streching here
	img_eq = cv2.equalizeHist(image)
	return img_eq

def findFaces(image, model="cnn"):
	# Identifying faces in the image
	faces = face_recognition.face_locations(image)
	croppedFaceImages = []
	for face in faces:
		top, right, bottom, left = face
		croppedFace = image[top:bottom, left:right]
		croppedFace = cv2.resize(croppedFace, (48, 48))
		croppedFaceImages.append(croppedFace)
	return croppedFaceImages

def generateIntensityImages(image):
	# Implement generation of images with different intensites
	generatedImages = []
	for gamma in [0.5, 1.5, 2.0, 2.5]:
		generated_intensity = np.array(255*(image / 255) ** gamma, dtype = 'uint8')
		generatedImages.append(generated_intensity)
	return generatedImages

def pseudoLabelledData():
	# Loading data which has been labelled using pseudo labelling technique
	pseudoImages, pseudoLabels = [], []
	if os.path.isfile("./data/pseudo_data.pickle"):
		with open("./data/pseudo_data.pickle", "rb") as f:
			pseudoImages, pseudoLabels = pickle.load(f)
			f.close()
	
	return (pseudoImages, pseudoLabels)

def preprocess_input(x, v2=True):
	x = x.astype('float32')
	x = x / 255.0
	if v2:
		x = x - 0.5
		x = x * 2.0
	return x