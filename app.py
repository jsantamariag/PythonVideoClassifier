from flask import Flask, jsonify, request
from flask_cors import CORS
from pytube import YouTube
import moviepy.editor as mp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import whisper

# Initialize Flask and CORS
app = Flask(__name__)
CORS(app)

# Load models and tokenizers
tokenizer_text = AutoTokenizer.from_pretrained("jsantamariag/RobertaAnubisLast")
model_text = AutoModelForSequenceClassification.from_pretrained("jsantamariag/RobertaAnubisLast")
model_text.eval()

asr_model = model = whisper.load_model("large")

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    data = request.json
    youtube_link = data.get('link', '')

    if not youtube_link:
        return jsonify({"error": "No YouTube link provided"}), 400

    # Download the full video
    yt = YouTube(youtube_link)
    video = yt.streams.get_highest_resolution()
    out_file = video.download(output_path='.')

    # Convert video to audio (.wav)
    audio_path = "processed_audio.wav" #os.path.splitext(out_file)[0] + ".wav"

    if os.path.exists(audio_path):
        os.remove(audio_path)

    video_clip = mp.VideoFileClip(out_file)
    video_clip.audio.write_audiofile(audio_path)

    # Perform ASR
    print("Performing ASR...")
    asr_result = model.transcribe("processed_audio.wav")
    print(asr_result["text"])

    transcribed_text = asr_result["text"]

    # Classify the text
    inputs_text = tokenizer_text(transcribed_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs_text = model_text(**inputs_text)

    logits = outputs_text.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item() * 100

    result = {
        'transcription': transcribed_text,
        'classification': 'Hate' if predicted_class == 1 else 'Not-Hate',
        'confidence': confidence
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
