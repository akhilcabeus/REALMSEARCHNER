from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy

app = Flask(__name__)
CORS(app)

# Load the pre-trained SpaCy model
nlp_ner = spacy.load("model-best")

@app.route('/ner', methods=['POST'])
def extract_entities():
    # Get data from the request
    data = request.json
    text = data.get("text")

    # Make sure that text is provided
    if not text:
        return jsonify({"error": "No text provided for entity recognition."}), 400

    # Process the text with the NER model
    doc = nlp_ner(text)

    # Extract entities from the doc
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    # Return the entities as a JSON response
    return jsonify(entities)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
