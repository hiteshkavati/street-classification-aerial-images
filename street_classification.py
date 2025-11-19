import os
import cv2
import numpy as np
import pickle
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# **Texture Feature Extraction Function**
def calculate_texture_features(image_path):
    """
    Extract texture features from an image using Grey-Level Co-occurrence Matrix (GLCM).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Resize image for consistency
    resized_image = cv2.resize(image, (128, 128))
    
    # Calculate GLCM and extract features
    glcm = graycomatrix(resized_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    features = {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'ASM': graycoprops(glcm, 'ASM').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean(),
    }
    return features

# **Shadow Feature Extraction Function**
def calculate_shadow_percentage(image_path):
    """
    Calculate the percentage of shadow pixels in the image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Threshold to detect dark (shadow) pixels
    threshold = 50  # Adjust this value based on your dataset
    shadow_pixels = np.sum(image < threshold)
    total_pixels = image.size
    shadow_percentage = (shadow_pixels / total_pixels) * 100
    return shadow_percentage

# **Display Histogram Function**
def display_histogram(image_path):
    """
    Display the pixel intensity histogram of the image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    plt.title("Pixel Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# **Dataset Preparation**
def prepare_data(dataset_dir):
    """
    Prepare dataset by extracting texture and shadow features for all images.
    """
    data = []
    labels = []
    classes = {'non street': 0, 'street': 1}  # Define labels for each class

    for label_name, label_value in classes.items():
        folder_path = os.path.join(dataset_dir, label_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue

        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            try:
                texture_features = calculate_texture_features(image_path)
                shadow_percentage = calculate_shadow_percentage(image_path)

                features = list(texture_features.values()) + [shadow_percentage]
                data.append(features)
                labels.append(label_value)
            except FileNotFoundError as e:
                print(e)

    return np.array(data), np.array(labels)

# **Debugging Folder Structure**
def debug_dataset_structure(dataset_dir):
    """
    Debug the dataset structure to ensure files and folders exist.
    """
    print(f"Dataset Directory Exists: {os.path.exists(dataset_dir)}")
    for folder in ['non street', 'street']:
        path = os.path.join(dataset_dir, folder)
        print(f"{folder} Folder Exists: {os.path.exists(path)}")
        if os.path.exists(path):
            print(f"Number of Files in {folder}: {len(os.listdir(path))}")

# **Training and Testing the Model**
dataset_dir = 'data'  # inside the project folder
  # Update with your dataset path

# Debug dataset structure
debug_dataset_structure(dataset_dir)

# Prepare data
X, y = prepare_data(dataset_dir)

if len(X) == 0 or len(y) == 0:
    print("Dataset is empty or improperly configured. Please check the dataset directory and try again.")
else:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Save the model
    model_path = 'models/aerial_dataset_rf.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(clf, file)
    print(f"Model saved at: {model_path}")

    # **Predict for a Single Image**
    def predict_image_with_visualization(image_path, model, feature_names):
        """
        Predict whether an image belongs to 'street' or 'non street' class and display the image with prediction.
        """
        if not os.path.exists(image_path):
            print(f"Error: File not found at {image_path}")
            return

        try:
            # Extract features
            texture_features = calculate_texture_features(image_path)
            shadow_percentage = calculate_shadow_percentage(image_path)
            features = list(texture_features.values()) + [shadow_percentage]
            features = np.array(features).reshape(1, -1)  # Reshape for model input
            
            # Predict
            prediction = model.predict(features)[0]
            class_name = 'street' if prediction == 1 else 'non street'

            # Load and display the image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct visualization
            
            plt.imshow(image)
            plt.title(f"Prediction: {class_name}")
            plt.axis('off')
            plt.show()

            # Display the histogram
            display_histogram(image_path)

            print(f"Prediction: {class_name}")
            print(f"Feature Details: {dict(zip(feature_names, features.flatten()))}")
        except FileNotFoundError as e:
            print(e)

    feature_names = [
        'contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation', 'shadow_percentage'
    ]

    # Example: Predicting for a single image
    sample_image_path = 'data/non street/Scene0094_View04_target.png'  # Update the path
    predict_image_with_visualization(sample_image_path, clf, feature_names)