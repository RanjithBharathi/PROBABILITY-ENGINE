import cv2
import mediapipe as mp
import numpy as np
import time
import speech_recognition as sr
import joblib
import os # Import os module for path joining

# Load the model and label encoder
# Make sure 'pose_model.pkl' and 'label_encoder.pkl' are in the same directory or provide the full path
MODEL_PATH = "pose_model.pkl" # <-- Verify this path
LABEL_ENCODER_PATH = "label_encoder.pkl" # <-- Verify this path

try:
    clf = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("✅ Model and Label Encoder loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Model or Label Encoder file not found. Ensure '{MODEL_PATH}' and '{LABEL_ENCODER_PATH}' exist.")
    exit() # Exit if model files are not found
except Exception as e:
    print(f"❌ Error loading model or label encoder: {e}")
    exit()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
# Use static_image_mode=False for video stream processing
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Pose Capture ---
def capture_pose_features(duration=3):
    """Captures pose landmarks from webcam for a given duration and returns averaged features."""
    cap = cv2.VideoCapture(0) # Open webcam (usually camera 0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return None

    all_features_list = [] # List to store captured features from multiple frames
    start_time = time.time()

    print(f"📷 Capturing body pose for {duration} seconds...")

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            # print("Warning: Failed to grab frame.") # Uncomment for detailed logs
            time.sleep(0.01) # Wait a bit before trying again
            continue

        # Flip the frame horizontally for a more intuitive view
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for pose detection
        results = pose.process(frame_rgb)

        # Optional: Draw landmarks on the frame for visualization
        # mp_drawing = mp.solutions.drawing_utils
        # if results.pose_landmarks:
        #     mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # cv2.imshow('Pose Capture - PROBIO CORE', frame) # Use a distinct window name
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # Extract landmarks if detected and add to list
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Extract ALL 33 landmarks (x, y, z) and flatten them to get 99 features
            current_features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

            # Ensure the feature vector is indeed 99 features long
            if current_features.shape[0] == 99:
                 all_features_list.append(current_features)
            else:
                 print(f"Warning: Detected pose with {current_features.shape[0]} features, expected 99. Skipping frame.")


        # If you uncommented imshow, add waitKey outside the pose_landmarks check
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


    cap.release() # Release the webcam
    cv2.destroyAllWindows() # Close any OpenCV windows

    # --- Process Collected Features ---
    if all_features_list:
        # Average the features across all successfully captured frames
        averaged_features = np.mean(all_features_list, axis=0)
        print(f"✅ Captured and averaged features from {len(all_features_list)} pose detections.")
        return averaged_features
    else:
        # If no pose was ever detected successfully
        print("❌ No valid pose features were detected during the capture duration.")
        return None


# --- Voice Input ---
def get_audio_input():
    """Records audio from microphone and converts to text."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    try:
        # Adjust for ambient noise before prompt
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1) # Adjust duration as needed

        print("\n🎤 Speak your command:")
        with mic as source:
             # Use a higher timeout if needed, but 7 seconds should be enough for a short command
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=5)

        # Use Google Web Speech API to recognize the audio
        text = recognizer.recognize_google(audio)
        print(f"🗣️ You said: {text}")
        return text

    except sr.WaitTimeoutError:
        print("⏱️ Timeout! No speech detected.")
        return None
    except sr.UnknownValueError:
        print("❌ Couldn't understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"❌ Speech Recognition service is unavailable: {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during voice input: {e}")
        return None

# --- Voice Command Processing (for filtering predictions) ---
def process_voice_command(voice_text, all_labels):
    """Processes voice text to find relevant labels for filtering."""
    if voice_text is None:
        # If no voice input, consider all labels relevant
        return list(all_labels)

    voice_text = voice_text.lower()
    relevant_labels = []

    # Simple keyword matching to filter relevant labels
    for label in all_labels:
        # Check if any word in the label is in the voice text
        # Or if any word in the voice text is in the label
        label_words = label.lower().split('_') # Split labels like 'legglance_flick'
        voice_words = voice_text.split()

        if any(word in voice_words for word in label_words) or \
           any(word in label_words for word in voice_words):
           relevant_labels.append(label)

    # Add some common related terms if needed (e.g., "cricket" for 'drive', 'pullshot', 'sweep')
    # This part can be expanded based on your labels
    if "cricket" in voice_words:
         cricket_labels = [lbl for lbl in all_labels if any(shot in lbl.lower() for shot in ['drive', 'pullshot', 'sweep', 'cricket'])]
         relevant_labels.extend([lbl for lbl in cricket_labels if lbl not in relevant_labels]) # Add unique cricket labels


    # Remove duplicates and return
    relevant_labels = list(dict.fromkeys(relevant_labels)) # Removes duplicates while maintaining order
    if relevant_labels:
        print(f"Filtering predictions based on voice command. Relevant labels: {relevant_labels}")
        return relevant_labels
    else:
        print("No specific relevant category found based on voice command. Considering all categories.")
        # Fallback: if voice command didn't match any label, consider all labels
        return list(all_labels)


# --- Main PROBIO Execution ---
def predict_from_pose_and_text():
    """Main function to capture pose and voice, make prediction, and output results."""
    print("\n🧠 PROBIO: The Real-time Pose Classification Engine is starting...\n")

    # Step 1: Pose Capture
    feature = capture_pose_features(duration=3) # Capture pose for 3 seconds

    if feature is None:
        print("❌ Prediction failed: No valid pose features captured.")
        # Add a small delay before exiting or prompting again
        time.sleep(2)
        return

    # Reshape feature to be a 2D array (1 sample, 99 features) for the model
    feature = feature.reshape(1, -1)

    # Step 2: Voice Input (Optional for prediction)
    voice = get_audio_input()
    # Voice input is optional, prediction proceeds even if voice capture fails.

    # Step 3: Predict using model
    try:
        # Get prediction probabilities for all classes
        probabilities = clf.predict_proba(feature)[0]
        all_labels = label_encoder.classes_ # Get all possible labels known by the model

        # Step 4: Process Voice Command and Determine Relevant Labels
        relevant_labels = process_voice_command(voice, list(all_labels))

        # Step 5: Find the best prediction among the relevant labels
        predicted_label = "Undetermined" # Default if no relevant prediction is possible
        confidence = 0.0

        # Get the indices in the original probabilities array for the relevant labels
        # Handle cases where a relevant label might not be in the model's classes (shouldn't happen if using all_labels)
        relevant_indices = [i for i, label in enumerate(all_labels) if label in relevant_labels]

        if not relevant_indices:
            print("❌ No overlap between relevant voice labels and model labels. Cannot filter.")
            # Fallback: Predict across all labels if filtering was impossible
            predicted_index_in_all = np.argmax(probabilities)
            predicted_label = label_encoder.inverse_transform([predicted_index_in_all])[0]
            confidence = probabilities[predicted_index_in_all]
            print("Falling back to predicting across all categories.")
            # relevant_probs_subset is not defined here, but that's okay as it's only used in the print block below
        else:
            # Find the index corresponding to the maximum probability among the relevant labels
            # Create a view of probabilities only for relevant indices
            relevant_probs_subset = probabilities[relevant_indices]

            if relevant_probs_subset.size > 0:
                 max_relevant_index_in_subset = np.argmax(relevant_probs_subset)
                 # Get the actual label and confidence using the index from the relevant subset
                 predicted_label = relevant_labels[max_relevant_index_in_subset]
                 confidence = relevant_probs_subset[max_relevant_index_in_subset] # Confidence within the relevant subset

            else:
                 # This case should ideally not be reached if relevant_indices is not empty,
                 # but as a safeguard:
                 print("❌ Relevant probabilities subset is empty. Cannot predict.")
                 predicted_label = "Undetermined"
                 confidence = 0.0
            # relevant_probs_subset is defined here


    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        # Add a small delay before exiting or prompting again
        time.sleep(2)
        return

    # Step 6: Output Results
    print(f"\n✅ Predicted Classification: {predicted_label}")
    print(f"📈 Confidence Score: {confidence:.2f}")

    # Optional: Print probabilities for all relevant labels for more insight
    print("\nProbabilities for Relevant Categories:")
    # Make sure relevant_probs_subset is defined before trying to use it
    if 'relevant_probs_subset' in locals() and relevant_probs_subset.size > 0:
        sorted_relevant_indices = np.argsort(relevant_probs_subset)[::-1]
        for i in sorted_relevant_indices:
            label = relevant_labels[i]
            prob = relevant_probs_subset[i]
            print(f"- {label}: {prob:.2f}")
    elif relevant_labels:
         print("No probabilities available for the identified relevant labels.")


    print("\n🔚 PROBIO Classification Completed.\n")
    # Add a small delay before exiting or prompting again (if in a loop)
    time.sleep(2)


if __name__ == "__main__":
    # --- Continuous Classification Loop ---
    # Uncomment the loop below to make the script run continuously
    # while True:
    #     predict_from_pose_and_text()
    #     # Add a prompt to continue or exit
    #     # user_input = input("Press Enter to predict again, or type 'q' and Enter to quit: ")
    #     # if user_input.lower() == 'q':
    #     #     break

    # If not in a loop, just run once:
    predict_from_pose_and_text()
