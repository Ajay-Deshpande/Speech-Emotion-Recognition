import joblib
import moviepy.editor as mp
import librosa
import numpy as np
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, jsonify

# Load the pre-trained Keras model
model = load_model('Emotion_Voice_Detection_Model.h5')
label_encoder = joblib.load('label_encoder.joblib')

app = Flask(__name__, template_folder="./")

# Helper function to extract audio from video
def extract_audio_from_video(video_path):
    clip = mp.VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    clip.audio.write_audiofile(audio_path)
    return audio_path

# Helper function to process audio for model prediction
def process_audio_frame(audio_segment, sr):
    # Extract features like MFCC or others needed for model
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)  # Aggregate over time axis if needed
    return mfcc.reshape(1, -1, 1)  # Reshape to match model input shape

# Overlay text with emotion probabilities on the frame
def overlay_text_on_frame(frame, emotion_probs, encoder):
    # Create the overlay text with probabilities
    text = "Emotion Predictions: "
    cv2.putText(frame, text, (1300, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    emotion_ind = emotion_probs.argmax()
    for i, prob in enumerate(emotion_probs):
        color = (0, 0, 255)
        text = f"{encoder.inverse_transform([i])[0]}: {prob:.2f}"
        if emotion_ind == i:
            color = (0, 255, 0)
        cv2.putText(frame, text, (1400, (50 * i) + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Overlay the text on the frame
    return frame


# Function to play video frame by frame with real-time predictions using last 20 seconds of audio
@app.route('/upload', methods=['POST'])
def play_video_frame_by_frame():
    # Extract audio and load video
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    video_path = f'uploaded_{file.filename}'
    file.save(video_path)
    
    audio_path = extract_audio_from_video(video_path)
    audio_array, sr = librosa.load(audio_path, res_type = 'kaiser_fast')
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    window_dim = (1600, 800)
    frame_idx = 0
    speech_idx = 0
    last_emotion_probs = np.zeros((len(label_encoder.classes_),))  # Initialize empty probabilities
    every_x_sec = 10
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, window_dim, interpolation=cv2.INTER_AREA)

        # Ensure there's enough audio data to process
        if (frame_idx % (every_x_sec * fps) == 0):
            # Process the audio frame and predict the emotion
            processed_audio = process_audio_frame(audio_array[speech_idx : speech_idx + (every_x_sec * sr)], sr)
            print(speech_idx, audio_array.shape)
            speech_idx += (every_x_sec * sr)
            last_emotion_probs = model.predict(processed_audio)[0]  # Get softmax output
            print(last_emotion_probs)
            
        # Overlay the last predicted probabilities on the video frame
        frame = overlay_text_on_frame(frame, last_emotion_probs, label_encoder)
        
        # Show the frame with predictions
        cv2.imshow('Emotion Prediction Frame by Frame', frame)
        frame_idx += 1
        
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return {'status' : 'Success'}

# Web portal
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
