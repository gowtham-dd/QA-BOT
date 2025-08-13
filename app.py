from flask import Flask, render_template, request, jsonify
import os
import torch
import joblib
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# app.py
def preprocess(text):
    return text.lower()

tfidf_vectorizer = joblib.load("artifacts/model_trainer/tfidf/tfidf_vectorizer.pkl")

# Load TF-IDF model
with open("artifacts/model_trainer/tfidf/tfidf_sentences.json", "r") as f:
    tfidf_sentences = json.load(f)

# Load DistilBERT model
distilbert_model = DistilBertForQuestionAnswering.from_pretrained(
    "artifacts/model_trainer/distilbert"
).to(device)
distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained(
    "artifacts/model_trainer/distilbert"
)

class QAPipeline:
    def __init__(self):
        self.device = device
    
    def predict_tfidf(self, question: str) -> str:
        """Predict using TF-IDF model"""
        try:
            q_vec = tfidf_vectorizer.transform([question])
            sims = cosine_similarity(q_vec, tfidf_vectorizer.transform(tfidf_sentences))[0]
            return tfidf_sentences[np.argmax(sims)].strip()
        except Exception as e:
            raise Exception(f"TF-IDF prediction failed: {str(e)}")

    def predict_distilbert(self, context: str, question: str) -> str:
        """Predict using DistilBERT model"""
        try:
            inputs = distilbert_tokenizer(
                question,
                context,
                truncation=True,
                max_length=384,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = distilbert_model(**inputs)

            start = torch.argmax(outputs.start_logits)
            end = torch.argmax(outputs.end_logits) + 1
            answer = distilbert_tokenizer.convert_tokens_to_string(
                distilbert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end]))
            
            return answer.strip()
        except Exception as e:
            raise Exception(f"DistilBERT prediction failed: {str(e)}")

qa_pipeline = QAPipeline()

@app.route("/")
def index():
    return render_template("index.html", 
                         answer="", 
                         context="", 
                         question="",
                         selected_model="distilbert")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        context = request.form.get("context", "").strip()
        question = request.form.get("question", "").strip()
        model_type = request.form.get("model_type", "distilbert")

        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        if model_type == "distilbert" and not context:
            return jsonify({"error": "Context is required for DistilBERT"}), 400

        if model_type == "tfidf":
            answer = qa_pipeline.predict_tfidf(question)
        else:
            answer = qa_pipeline.predict_distilbert(context, question)

        return jsonify({
            "answer": answer,
            "model_used": model_type
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080)