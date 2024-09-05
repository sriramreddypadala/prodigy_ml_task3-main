import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Use raw string to handle backslashes in Windows paths
DATADIR = r'E:\prodigy\task 3\new one\PetImages'
CATEGORIES = ['cat', 'dog']
IMG_SIZE = 64

# Function to create dataset by loading images
def create_data():
    data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        print(f"Accessing path: {path}")  # Debugging line to check the directory path
        class_num = CATEGORIES.index(category)  # 0 for cat, 1 for dog
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([new_array, class_num])
            except Exception as e:
                print(f"Error processing {img}: {e}")
                pass
    return data

# Create the dataset
print("Loading images and creating dataset...")
data = create_data()


# Split features and labels
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

# Convert to NumPy arrays and normalize
X = np.array(X).reshape(-1, IMG_SIZE * IMG_SIZE) / 255.0
y = np.array(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Classification report
print(classification_report(y_test, y_pred, target_names=CATEGORIES))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Save the model (optional)
joblib.dump(svm_model, 'svm_model_cats_vs_dogs.pkl')

# Optional: Plot some results
for i in range(5):
    plt.imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title(f"Actual: {CATEGORIES[y_test[i]]}, Predicted: {CATEGORIES[y_pred[i]]}")
    plt.show()
