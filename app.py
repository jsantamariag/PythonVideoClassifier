from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from pytube import YouTube
import moviepy.editor as mp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import whisper

# Initialize Flask, CORS and SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


# Load models and tokenizers
tokenizer_text = AutoTokenizer.from_pretrained("jsantamariag/RobertaAnubisLast")
model_text = AutoModelForSequenceClassification.from_pretrained("jsantamariag/RobertaAnubisLast")
model_text.eval()
asr_model = model = whisper.load_model("large")

# Start program
@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    out_file = None
    audio_path = "processed_audio.wav"
    result = {}

    try:
        data = request.json
        youtube_link = data.get('link', '')

        if not youtube_link:
            return jsonify({"error": "No YouTube link provided"}), 400

        # Download the full video
        try:
            print("Downloading full video...")
            socketio.emit('message', {'data': 'Downloading video...'})
            yt = YouTube(youtube_link)
            video = yt.streams.get_highest_resolution()
            out_file = video.download(output_path='.')
        except Exception as e:
            return jsonify({"error": "Error downloading video"}), 500

        # Convert video to audio (.wav)
        try:
            print("Extracting audio...")
            socketio.emit('message', {'data': 'Extracting audio...'})

            if os.path.exists(audio_path):
                os.remove(audio_path)

            video_clip = mp.VideoFileClip(out_file)
            video_clip.audio.write_audiofile(audio_path)
            video_clip.close() # Close the video clip to release resources and prevent concurrency problems when deleting later
        except Exception as e:
            return jsonify({"error": "Error extracting audio"}), 500

        # Perform ASR
        try:
            print("Performing ASR...")
            socketio.emit('message', {'data': 'Performing Speech Recognition...'})
            asr_result = asr_model.transcribe(audio_path)
            transcribed_text = asr_result["text"]
        except Exception as e:
            return jsonify({"error": "Error performing ASR"}), 500

        # Classify the text
        try:
            print("Classifying text...")
            socketio.emit('message', {'data': 'Classifying message...'})

            inputs_text = tokenizer_text(transcribed_text, return_tensors="pt", truncation=True, padding=True,
                                         max_length=512)
            with torch.no_grad():
                outputs_text = model_text(**inputs_text)

            logits = outputs_text.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item() * 100

            print("Preparing report...")
            socketio.emit('message', {'data': 'Preparing report...'})

            result = {
                'transcription': transcribed_text,
                'classification': 'Hate' if predicted_class == 1 else 'Not-Hate',
                'confidence': confidence
            }
        except Exception as e:
            return jsonify({"error": "Error classifying text"}), 500

        return jsonify(result)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        socketio.emit('message', {'data': f"An unexpected error occurred: {e}"})
        return jsonify({"error": "An unexpected error occurred during processing"}), 500

    finally:
        # Cleanup
        if out_file and os.path.exists(out_file):
            os.remove(out_file)
        if os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == '__main__':
    socketio.run(app, debug=True)
