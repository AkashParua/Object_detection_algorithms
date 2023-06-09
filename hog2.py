import numpy as np
from skimage.feature import hog
from skimage import io , transform
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

image_samples_path = "Data"   #insert the path of the 
labels = ['bicycle' ,'bus' ,'car', 'motorcycle' , 'train', 'truck']
samples = []
sample_labels = []

for i, label in enumerate(labels):
    label_samples_path = os.path.join(image_samples_path, label)
    for filename in os.listdir(label_samples_path):
        image_path = os.path.join(label_samples_path, filename)
        image = io.imread(image_path, as_gray=True)
        print(image)
        image = transform.resize(image , (128, 128))
        samples.append(image)
        sample_labels.append(i)


features = []
for sample in samples:
    hog_features = hog(sample, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    features.append(hog_features)



x = np.array(features)
y = np.array(sample_labels)

X_train, X_test, y_train, y_test = train_test_split(np.array(features), np.array(sample_labels), test_size=0.2, random_state=42)

svm_classifier = OneVsRestClassifier(SVC())
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

with open('model.pkl' , 'wb') as f:
    pickle.dump(svm_classifier , f)
    


