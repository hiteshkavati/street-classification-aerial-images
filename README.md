# Street Classification in Aerial Images Using Texture and Shadows

This project classifies aerial images into **street** and **non street** classes using
hand-crafted **texture features** (GLCM) and **shadow information**, combined with a
**Random Forest** classifier.

Instead of using deep learning, this project focuses on classic computer vision and
machine learning: feature extraction → feature vectors → supervised classification.

---

## 🧠 Project Overview

Given an aerial image, the pipeline does the following:

1. Converts the image to grayscale and resizes it to a fixed size.
2. Computes **Grey-Level Co-occurrence Matrix (GLCM)** features:
   - contrast  
   - dissimilarity  
   - homogeneity  
   - ASM (Angular Second Moment)  
   - energy  
   - correlation
3. Computes the **percentage of shadow pixels** (very dark regions).
4. Concatenates these into a feature vector.
5. Trains a **RandomForestClassifier** to predict whether the image contains a street.
6. Evaluates the model and can **predict on a single image** with:
   - image visualization
   - pixel intensity histogram
   - printed feature values.

---

## 🧩 Technologies Used

- **Python**
- **OpenCV** – image loading, resizing, grayscale conversion
- **NumPy** – numerical operations and arrays
- **scikit-image**
  - `graycomatrix`, `graycoprops` for GLCM texture features
- **scikit-learn**
  - `RandomForestClassifier`
  - `train_test_split`
  - `classification_report`, `accuracy_score`
- **matplotlib** – plotting histograms and visualizing images
- **pickle** – saving the trained model to a `.pkl` file

---

## 📁 Folder Structure

```text
street-classification-aerial/
├── src/
│   └── street_classification.py   # main script
├── data/                          # training data (not tracked in git)
│   ├── street/
│   └── non street/
├── models/                        # saved .pkl models
├── requirements.txt
├── .gitignore
└── README.md

📦 Dataset Format

The script expects the dataset in the following structure:

data/
├── street/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── non street/
    ├── imageA.png
    ├── imageB.png
    └── ...


Each image can be .png, .jpg, etc. The class labels are:

street → 1

non street → 0

You can adjust this inside the classes dictionary in the code if needed.

⚙️ Setup Instructions
1. Clone the Repository
git clone https://github.com/hiteshkavati/street-classification-aerial-images.git
cd street-classification-aerial-images

2. Create a Virtual Environment (Optional But Recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

4. Prepare the Dataset

Place your aerial image dataset inside the data/ folder:

data/
├── street/
└── non street/


Update any paths inside src/street_classification.py if needed.

▶️ How to Run

From the project root:

python src/street_classification.py


The script will:

Debug the dataset structure (check folders and file counts).

Extract features (GLCM + shadow percentage) for all images.

Split into train/test sets.

Train a Random Forest classifier.

Print:

Classification report

Accuracy score

Save the trained model into models/aerial_dataset_rf.pkl.

🔍 Predicting a Single Image

Inside street_classification.py, there is:

sample_image_path = 'data/non street/Scene0094_View04_target.png'
predict_image_with_visualization(sample_image_path, clf, feature_names)


To test your own image:

Put the image into data/street/ or data/non street/.

Change sample_image_path to that file’s path.

Run the script again.

The function will:

Display the image with the predicted label (street / non street).

Show the pixel intensity histogram.

Print the exact feature values used for the prediction.

📊 Example Features Used

Feature vector for each image:

contrast

dissimilarity

homogeneity

ASM

energy

correlation

shadow_percentage

These come from GLCM and the fraction of dark pixels in the image.

🚧 Possible Improvements

Add more robust pre-processing (noise reduction, normalization).

Tune Random Forest hyperparameters (number of trees, depth, etc.).

Add cross-validation and better evaluation metrics (ROC, AUC).

Save and load the model and feature configuration from separate files.

Extend to multi-class classification (e.g. different road types).

Compare with a CNN-based baseline.
