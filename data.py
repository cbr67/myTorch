import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

def load_tabular_data(file_path):
    """Load tabular data from multiple potential file types."""
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.npz'):
        data = np.load(file_path)
        return {key: data[key] for key in data.files}  # Extract multiple arrays
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.txt'):
        return pd.read_csv("linear.txt", delim_whitespace=True)
    else:
        raise ValueError("Unsupported file format.")
    
def load_images_from_folder(folder_path):
    """Load all images from a specified folder."""
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            images.append(img)
    return images

def preprocess_image(image, resize_shape=(128, 128)):
    """Preprocess image by resizing and normalizing pixel values."""
    image = cv2.resize(image, resize_shape)

    # Ensure pixel values are within 0-255 before normalization
    assert image.min() >= 0 and image.max() <= 255, "Pixel values out of expected range (0-255) before normalization"

    image = image / 255.0  # Normalize pixel values

    # Ensure pixel values are within 0-1 after normalization
    assert image.min() >= 0.0 and image.max() <= 1.0, "Pixel values out of expected range after normalization"

    return image

def load_text_data(file_path):
    """Load text data from a given file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def preprocess_tabular_text(data_frame, tfid=False, hashing=False):
    '''
    input: pandas data frame
    #tfid == True converts the words within the input vectors into values representing their TFID value rather than one-hot encoding. 
    output: input vectors X representing text inputs, classification labels y.
    '''
    # Ensure input is a DataFrame with text descriptions in column 0 and labels in column 1
    if not isinstance(data_frame, pd.DataFrame) or data_frame.shape[1] != 2:
        raise ValueError("Input data_frame must be a pandas DataFrame with exactly two columns.")
    
    # Extract text descriptions and labels
    text_data = data_frame.iloc[:, 0]  # First column: Text descriptions
    labels = data_frame.iloc[:, 1]  # Second column: Classification labels

    # Preprocess text (Lowercasing, Removing Punctuation)
    def clean_text(text):
        text = text.lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())  # Keep only alphanumeric & space
        return text  # No need to split words explicitly, TfidfVectorizer will handle tokenization
    text_data = text_data.apply(clean_text)  # Apply cleaning function to all rows

    if tfid==True:
        #Option to use tfid vectors instead of onehot
        vectorizer = TfidfVectorizer(max_features=7500, stop_words='english', max_df=0.95, min_df=2) 
        X = vectorizer.fit_transform(text_data)  # sparse representation
        X = X.toarray()

    elif hashing==True:
        #Option to use hashing vectors instead of onehot
        vectorizer = HashingVectorizer(n_features=2**13, alternate_sign=False)  # Adjust feature size as needed
        X = vectorizer.transform(text_data)  # sparse representation
        X = X.toarray()
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    return X, y

import numpy as np

def one_hot_encode(y, num_classes):
    """
    y is an array of integer labels
    Returns a numpy.ndarray for a one-hot encoded matrix of shape (len(y), num_classes)
    """
    # Initialize a zero matrix of shape (samples, num_classes)
    one_hot = np.zeros((len(y), num_classes))
    
    # Set the index corresponding to the label to 1
    one_hot[np.arange(len(y)), y] = 1
    
    return one_hot

def split_data(data, labels, train_size= 0.7, val_size=0.15, test_size=0.15, random_state=42):
    """Split data into training, validation, and test sets."""
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=(val_size + test_size), random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (val_size + test_size)), random_state=random_state)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def save_data(data, file_path):
    """Save processed data to disk in CSV format."""
    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path, index=False)
    elif isinstance(data, list):
        with open(file_path, 'w', encoding='utf-8') as file:
            for line in data:
                file.write(line + '\n')
    else:
        raise ValueError("Unsupported data format for saving.")