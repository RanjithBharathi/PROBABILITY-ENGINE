import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Initialize MediaPipe Pose for processing static images
mp_pose = mp.solutions.pose
# Use static_image_mode=True for processing static images from folders
pose_processor = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- Configuration ---
# Path to the root directory containing your raw image folders (can be nested)
# Example: C:\MyRawImages\Cricket\Shots\CoverDrive, C:\MyRawImages\Emotions\Happy
RAW_IMAGE_DATA_DIR = r"C:\A PROBABILITY ENGINE\datasets" # <-- **EDIT THIS PATH**
PROCESSED_DATA_DIR = r"C:\A PROBABILITY ENGINE\processed" # <-- **VERIFY OR EDIT THIS PATH**
# Supported image extensions (add more if needed)
supported_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

print(f"Reading raw images from nested folders under: {RAW_IMAGE_DATA_DIR}")
print(f"Saving processed .npy data to: {PROCESSED_DATA_DIR}")

# --- Setup Output Directory ---
# Create the processed data directory if it doesn't exist
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

# Dictionary to hold pose features, keyed by label
# This will store all features for a label before saving to a single .npy file
all_labels_pose_features = {}

# --- Processing Loop ---
# Walk through the raw image data directory, including nested folders
for root, dirs, files in os.walk(RAW_IMAGE_DATA_DIR):

    # Determine the label based on the current directory's name
    # We'll use the immediate directory name as the label for simplicity
    # If your labeling logic is more complex (e.g., needs a parent folder name),
    # this logic might need adjustment.
    current_label = os.path.basename(root)

    # Skip the root processing directory itself or any empty directories
    if current_label == os.path.basename(RAW_IMAGE_DATA_DIR) or not files:
        continue

    print(f"\nProcessing images in directory: {root} (Label: {current_label})")

    # Process each file in the current directory
    for file in files:
        # Check if the file is a supported image
        if os.path.splitext(file)[1].lower() in supported_image_extensions:
            image_path = os.path.join(root, file)
            # print(f"Processing image: {image_path}") # Uncomment for detailed logs

            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Could not load image {image_path}. Skipping.")
                    continue

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

                    # Check if the extracted features have the expected count
                    if pose_features.shape[0] == 99:
                        # Store the extracted features under the current_label
                        if current_label not in all_labels_pose_features:
                            all_labels_pose_features[current_label] = []
                        all_labels_pose_features[current_label].append(pose_features)
                    else:
                        print(f"Warning: Extracted unexpected feature count ({pose_features.shape[0]}) from image {file} in {root}. Expected 99. Skipping.")

                # else:
                    # print(f"Warning: No pose detected in image {file} in {root}. Skipping.") # Uncomment for detailed logs

            except Exception as e:
                print(f"Error processing image {file_path}: {e}. Skipping.")

# --- Save the collected features for each label ---
print("\nSaving collected pose features to .npy files...")
for label, features_list in all_labels_pose_features.items():
    if features_list:
        # Create a subdirectory in the processed directory for this label
        processed_label_dir = os.path.join(PROCESSED_DATA_DIR, label)
        if not os.path.exists(processed_label_dir):
            os.makedirs(processed_label_dir)

        # Convert the list of pose features to a NumPy array
        label_array = np.array(features_list)
        # Define the file path to save the data (saving all features for this label into one .npy)
        save_file_path = os.path.join(processed_label_dir, f'{label}_pose_data.npy') # Example: coverdrive_pose_data.npy

        try:
            # Save the NumPy array to a .npy file
            np.save(save_file_path, label_array)
            print(f"✅ Saved {len(features_list)} pose feature samples for label '{label}' to {save_file_path}")
        except Exception as e:
            print(f"❌ Error saving .npy file for label '{label}' to {save_file_path}: {e}")

    else:
        print(f"❌ No valid pose features collected for label '{label}'. No .npy file saved.")


# --- Cleanup ---
pose_processor.close() # Close the pose processor
print("\nImage processing and .npy generation finished.")
