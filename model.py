import os
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import cv2
import mediapipe as mp

# Path to processed data
# Ensure this path is correct and contains subdirectories with your images and/or .npy data
DATA_DIR = "C:\\A PROBABILITY ENGINE\\processed" # <-- **VERIFY THIS PATH**

# Initialize MediaPipe Pose for processing images
# Use static_image_mode=True when processing static images
mp_pose = mp.solutions.pose
pose_processor = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def process_image_for_pose(image_path):
    """Loads an image, detects pose, and returns flattened landmarks."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load image {image_path}")
            return None

        # Convert the image to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to find pose landmarks
        results = pose_processor.process(img_rgb)

        # Extract landmarks if detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Extract x, y, z coordinates for each landmark and flatten
            # This ensures we get 33 * 3 = 99 features
            pose_features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            return pose_features
        else:
            # print(f"Warning: No pose detected in image {image_path}") # Uncomment for detailed logs
            return None # Return None if no pose is detected

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def load_all_data():
    X, y = [], []
    supported_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'] # Add more if needed

    # Walk through the data directory
    for root, _, files in os.walk(DATA_DIR):
        # The directory name is used as the label
        label = os.path.basename(root)
        if label == os.path.basename(DATA_DIR): # Skip the root directory itself if it contains files
            continue

        print(f"Processing directory: {label}")

        for file in files:
            file_path = os.path.join(root, file)

            # --- Handle .npy files ---
            if file.endswith('.npy'):
                try:
                    data = np.load(file_path, allow_pickle=True)
                    # Assuming .npy files contain sequences, where each item is a frame's pose data
                    # Each item in the sequence should be a 99-feature array
                    for i, item in enumerate(data):
                         # Ensure the item is a numpy array and has the expected shape
                        if isinstance(item, np.ndarray) and item.shape == (99,):
                            X.append(item)
                            y.append(label)
                        else:
                            print(f"Warning: Skipping item {i} from {file} with unexpected format or feature count: {item.shape if isinstance(item, np.ndarray) else type(item)}. Expected shape (99,).")

                except Exception as e:
                    print(f"Error loading or processing .npy file {file_path}: {e}")

            # --- Handle Image files ---
            elif os.path.splitext(file)[1].lower() in supported_image_extensions:
                 # print(f"Processing image {file_path}") # Uncomment for detailed loading logs
                 pose_features = process_image_for_pose(file_path)
                 if pose_features is not None:
                     # Check if the extracted features have the expected count
                     if pose_features.shape[0] == 99:
                         X.append(pose_features)
                         y.append(label)
                     else:
                         print(f"Warning: Skipping image {file} due to unexpected extracted feature count: {pose_features.shape[0]}. Expected 99.")


    # Close the pose processor after loading all data
    pose_processor.close()

    X = np.array(X)
    y = np.array(y)

    print(f"\n--- Data Loading Summary ---")
    print(f"Total samples loaded: {len(X)}")
    print(f"Feature data shape after stacking: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: {Counter(y)}")
    print(f"--------------------------\n")

    return X, y

# --- Main Training Process ---
print("🧠 Starting model training...")

# Load data
X, y = load_all_data()

# Check if any data was loaded
if len(X) == 0:
    print("❌ No data loaded. Cannot train the model.")
    print(f"Please check if the directory '{DATA_DIR}' exists and contains labeled subdirectories with images or .npy files.")
    exit() # Exit if no data is available

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print("✅ Labels encoded.")

# Split the dataset
# Using stratify=y_encoded helps maintain the proportion of labels in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"✅ Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")

# Train the model
print("🏋️ Training Logistic Regression model...")
# Increased max_iter and changed solver for potentially better convergence on this type of data
clf = LogisticRegression(max_iter=5000, solver='saga', multi_class='ovr', n_jobs=-1) # Use saga and all available cores
clf.fit(X_train, y_train)
print("✅ Model training complete.")

# Evaluate
print("\n📊 Evaluating model performance...")
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
# Generate classification report with target names
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("✅ Evaluation complete.")

# Save model and label encoder
try:
    joblib.dump(clf, "pose_model.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("✅ Model ('pose_model.pkl') and Label Encoder ('label_encoder.pkl') saved successfully.")
except Exception as e:
    print(f"❌ Error saving model or label encoder: {e}")


print("\n🏁 Model training script finished.")
