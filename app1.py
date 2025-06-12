from flask import Flask, request, jsonify, render_template
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
import os
import time
import speech_recognition as sr
import threading # To handle voice recognition in a separate thread
import tempfile # For creating temporary files
import logging # For better logging

app = Flask(__name__)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Load the model and label encoder
MODEL_PATH = "pose_model.pkl" # <-- Verify this path
LABEL_ENCODER_PATH = "label_encoder.pkl" # <-- Verify this path

try:
    clf = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    logging.info("✅ Model and Label Encoder loaded successfully for Web App.")
except FileNotFoundError:
    logging.error(f"❌ Error: Model or Label Encoder file not found. Ensure '{MODEL_PATH}' and '{LABEL_ENCODER_PATH}' exist.")
    clf = None
    label_encoder = None
except Exception as e:
    logging.error(f"❌ Error loading model or label encoder for Web App: {e}")
    clf = None
    label_encoder = None

# Initialize MediaPipe Pose for processing frames
mp_pose = mp.solutions.pose
pose_processor_app = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Global State to store latest data ---
latest_pose_features = None
latest_voice_command = None
last_pose_time = 0
last_voice_time = 0

# --- Pose Processing Function ---
def process_pose_from_frame(frame_bytes):
    global latest_pose_features, last_pose_time
    try:
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose_processor_app.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            pose_features = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

            if pose_features.shape[0] == 99:
                latest_pose_features = pose_features
                last_pose_time = time.time()
                return pose_features
            else:
                 return None
        else:
            return None
    except Exception as e:
        logging.error(f"❌ Error processing frame for pose: {e}")
        return None

# --- Voice Recognition Function (Runs in a separate thread) ---
def recognize_voice_thread(audio_bytes, original_filename="audio_from_browser"):
    """Recognizes speech from audio bytes and updates global state."""
    global latest_voice_command, last_voice_time
    recognizer = sr.Recognizer()
    logging.info(f"🎤 Voice recognition thread started for {original_filename}. Audio size: {len(audio_bytes)} bytes.")

    # Determine a safe extension based on common web audio types, default to .wav if unsure
    # This helps ffmpeg identify the format if it's not a standard WAV
    file_extension = ".wav" # Default
    if "webm" in original_filename.lower():
        file_extension = ".webm"
    elif "ogg" in original_filename.lower():
        file_extension = ".ogg"
    elif "mp3" in original_filename.lower(): # Though browsers rarely send mp3 directly from MediaRecorder
        file_extension = ".mp3"


    # Create a temporary file to store the audio bytes
    # tempfile.NamedTemporaryFile creates a file that is deleted when closed
    temp_audio_file = None
    try:
        # Suffix helps ffmpeg/speech_recognition identify the format
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_audio_file.write(audio_bytes)
        temp_audio_file_path = temp_audio_file.name
        temp_audio_file.close() # Close the file so AudioFile can open it

        logging.info(f"Audio bytes saved to temporary file: {temp_audio_file_path}")

        # Use sr.AudioFile to open the temporary audio file
        # This allows speech_recognition to use ffmpeg for conversion if necessary
        with sr.AudioFile(temp_audio_file_path) as source:
            logging.info(f"Opened temporary audio file with sr.AudioFile. Duration: {source.DURATION}s")
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.5) # Adjust for noise
                logging.info("Adjusted for ambient noise.")
                audio_data = recognizer.record(source) # Record the entire audio file
                logging.info("Audio data recorded from file source.")
            except Exception as e:
                logging.error(f"Error during adjust_for_ambient_noise or record: {e}")
                latest_voice_command = "Error processing audio"
                last_voice_time = time.time()
                return


        # Use Google Web Speech API to recognize the audio
        text = recognizer.recognize_google(audio_data)
        logging.info(f"🗣️ Voice command recognized: {text}")
        latest_voice_command = text
        last_voice_time = time.time()

    except sr.WaitTimeoutError:
        logging.warning("⏳ Voice recognition: No phrase heard (WaitTimeoutError).")
        latest_voice_command = "No speech detected" # More specific than None
        last_voice_time = time.time()
    except sr.UnknownValueError:
        logging.warning("❌ Voice recognition couldn't understand audio.")
        latest_voice_command = "Could not understand audio" # More specific than None
        last_voice_time = time.time()
    except sr.RequestError as e:
        logging.error(f"❌ Voice recognition service error: {e}")
        latest_voice_command = "Recognition service error" # More specific
        last_voice_time = time.time()
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred during voice recognition: {e}", exc_info=True)
        latest_voice_command = "Unexpected voice error" # More specific
        last_voice_time = time.time()
    finally:
        if temp_audio_file and os.path.exists(temp_audio_file_path):
            try:
                os.remove(temp_audio_file_path)
                logging.info(f"Temporary audio file {temp_audio_file_path} deleted.")
            except Exception as e:
                logging.error(f"Error deleting temporary file {temp_audio_file_path}: {e}")


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main index.html page."""
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_route(): # Renamed to avoid conflict with function name
    if clf is None or label_encoder is None:
        return jsonify({"error": "Model or Label Encoder not loaded."}), 500

    if 'frame' not in request.files:
        return jsonify({"error": "No frame file provided."}), 400

    file = request.files['frame']
    if file.filename == '':
        return jsonify({"error": "No selected frame."}), 400

    if file:
        frame_bytes = file.read()
        process_pose_from_frame(frame_bytes)
        return jsonify({"status": "pose processed"}), 200

    return jsonify({"error": "An unexpected error occurred processing the frame."}), 500

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected audio."}), 400

    if file:
        audio_bytes = file.read()
        original_filename = file.filename # Get the original filename sent by client
        logging.info(f"Received audio file: {original_filename}, size: {len(audio_bytes)} bytes")
        voice_thread = threading.Thread(target=recognize_voice_thread, args=(audio_bytes, original_filename))
        voice_thread.start()
        return jsonify({"status": "audio received, processing started"}), 200

    return jsonify({"error": "An unexpected error occurred processing the audio."}), 500


@app.route('/predict_live', methods=['POST'])
def predict_live():
    if clf is None or label_encoder is None:
        return jsonify({"error": "Model or Label Encoder not loaded."}), 500

    current_time = time.time()
    processed_voice_command = latest_voice_command # Default

    if latest_pose_features is None or (current_time - last_pose_time > 5):
        # Still return voice command even if pose is stale/missing
        if latest_voice_command is None or (current_time - last_voice_time > 10):
            processed_voice_command = "None (no recent command)"
        elif latest_voice_command and (current_time - last_voice_time > 10) : # Explicitly check if it was set but is now stale
            processed_voice_command = f"{latest_voice_command} (stale)"

        return jsonify({
            "prediction": "Waiting for pose data...",
            "confidence": 0.0,
            "probabilities": {},
            "voice_command": processed_voice_command
        }), 200

    feature = latest_pose_features.reshape(1, -1)

    try:
        probabilities = clf.predict_proba(feature)[0]
        all_labels = label_encoder.classes_

        voice_command_to_process = None
        if latest_voice_command and (current_time - last_voice_time < 10): # Voice command is recent
            voice_command_to_process = latest_voice_command
            processed_voice_command = latest_voice_command # This will be returned in JSON
        elif latest_voice_command: # Voice command exists but is stale
            processed_voice_command = f"{latest_voice_command} (stale)"
        else: # No voice command at all or it was an error string
            processed_voice_command = latest_voice_command if latest_voice_command else "None (no command)"


        relevant_labels = process_voice_command(voice_command_to_process, list(all_labels))

        predicted_label = "Undetermined"
        confidence = 0.0
        relevant_indices = [i for i, label in enumerate(all_labels) if label in relevant_labels]

        if not relevant_indices:
            predicted_index_in_all = np.argmax(probabilities)
            predicted_label = label_encoder.inverse_transform([predicted_index_in_all])[0]
            confidence = probabilities[predicted_index_in_all]
            relevant_probs_subset = probabilities
        else:
            relevant_probs_subset = probabilities[relevant_indices]
            if relevant_probs_subset.size > 0:
                 max_relevant_index_in_subset = np.argmax(relevant_probs_subset)
                 predicted_label = relevant_labels[max_relevant_index_in_subset]
                 confidence = relevant_probs_subset[max_relevant_index_in_subset]
            else:
                 relevant_probs_subset = np.array([])

        if relevant_indices:
             sorted_relevant_indices = np.argsort(relevant_probs_subset)[::-1]
             probabilities_dict = {relevant_labels[i]: float(relevant_probs_subset[i]) for i in sorted_relevant_indices}
        else:
             sorted_indices_all = np.argsort(probabilities)[::-1]
             probabilities_dict = {all_labels[i]: float(probabilities[i]) for i in sorted_indices_all}

        return jsonify({
            "prediction": predicted_label,
            "confidence": float(confidence),
            "probabilities": probabilities_dict,
            "voice_command": processed_voice_command
        }), 200

    except Exception as e:
        logging.error(f"❌ Error during live prediction: {e}", exc_info=True)
        processed_voice_command = latest_voice_command if latest_voice_command else "None"
        if latest_voice_command and (current_time - last_voice_time > 10):
             processed_voice_command = f"{latest_voice_command} (stale)"
        return jsonify({"error": f"Prediction failed: {e}", "voice_command": processed_voice_command}), 500

def process_voice_command(voice_text, all_labels):
    if voice_text is None:
        return list(all_labels)
    # Avoid processing error messages as voice commands
    if any(err_msg in voice_text for err_msg in ["Could not understand", "No speech detected", "Recognition service error", "Unexpected voice error", "Error processing audio"]):
        return list(all_labels)


    voice_text_lower = voice_text.lower()
    relevant_labels = []

    for label in all_labels:
        label_words = label.lower().split('_')
        voice_words = voice_text_lower.split()

        if any(word in voice_words for word in label_words) or \
           any(word in label_words for word in voice_words):
           relevant_labels.append(label)

    if "cricket" in voice_words:
         cricket_labels = [lbl for lbl in all_labels if any(shot in lbl.lower() for shot in ['drive', 'pullshot', 'sweep', 'cricket'])]
         relevant_labels.extend([lbl for lbl in cricket_labels if lbl not in relevant_labels])

    relevant_labels = list(dict.fromkeys(relevant_labels))
    if relevant_labels:
        return relevant_labels
    else:
        return list(all_labels)


if __name__ == '__main__':
    logging.info("🚀 Starting Flask Web Application (Live)...")
    app.run(debug=True, port=5000, use_reloader=False) # use_reloader=False can be helpful for debugging threads