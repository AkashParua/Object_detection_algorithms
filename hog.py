import numpy as np
from skimage.feature import hog
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

# Load the image samples and corresponding labels
image_samples_path = "path_to_image_samples_directory"
labels = ["animal", "person", "car", "bus", "truck", "motorcycle", "bicycle", "train","other"]

samples = []
sample_labels = []

for i, label in enumerate(labels):
    label_samples_path = os.path.join(image_samples_path, label)
    for filename in os.listdir(label_samples_path):
        image_path = os.path.join(label_samples_path, filename)
        image = io.imread(image_path, as_gray=True)
        samples.append(image)
        sample_labels.append(i)

# Compute HOG features for the samples
features = []
for sample in samples:
    hog_features = hog(sample, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    features.append(hog_features)

# Convert features and labels to NumPy arrays
X = np.array(features)
y = np.array(sample_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the multi-class SVM classifier using One-vs-Rest approach
svm_classifier = OneVsRestClassifier(SVC())
svm_classifier.fit(X_train, y_train)

# Predict labels for the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

