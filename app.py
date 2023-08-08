from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# Cargar el modelo y el tokenizador
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
model.eval()


@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    # Usando un texto placeholder
    text = "You are the worst because you are the one who has come here the most to laugh, to act respectful, when you are a fucking ignorant piece of shit, and you talk about my job or what I need or don't need. Imbecile! How disgusting you are! Ugh! How disgusting this pig is, huh! How disgusting this fucking pig is! Empathizing? No, the opposite of empathy. It\'s repulsive, huh? It\'s repulsive that some shithead comes here to tell me what I should or shouldn\'t earn when they have taken away a lot of money from me for fake bullshit, huh? Disgusting! Fucking pig! Fucking pig is what you are! Imbecile! How disgusting you are, huh! I swear it. How fucking disgusting you are! There comes a time when I can\'t stand his fucking shit-eating smile anymore, saying \'I wouldn\'t give you a dime if I knew what you were earning!\' Fucking moron! Fucking moron! How disgusting you are! Ugh! How disgusting you are!"


    # Clasificar el texto
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item() * 100

    result = {
        'classification': 'Hate' if predicted_class == 1 else 'Not-Hate',
        'confidence': confidence
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run()
