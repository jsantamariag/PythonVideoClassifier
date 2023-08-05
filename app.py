from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
#CORS(app, origins=['http://localhost:5000', 'http://localhost:5001', 'http://otrodominioquetucontroles.com'])

@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    # For now, we're just returning some placeholder data
    result = {
        'classification': 'Hate',
        'confidence': 95
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run()
