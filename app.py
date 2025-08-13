from flask import Flask, render_template, request, jsonify
import os
import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
from src.QABOT.utils.common import load_config

app = Flask(__name__)

# Load model and tokenizer
config = load_config()
model_path = config['model_trainer']['distilbert_model_dir']
tokenizer_path = config['model_trainer']['tokenizer_save_dir']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForQuestionAnswering.from_pretrained(model_path).to(device)
tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)

class QAPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, context: str, question: str) -> str:
        try:
            inputs = self.tokenizer(
                question,
                context,
                truncation=True,
                max_length=384,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
            
            return answer.strip()
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

qa_pipeline = QAPipeline(model, tokenizer)

@app.route("/")
def index():
    return render_template("index.html", answer="", context="", question="")

@app.route("/train", methods=["GET"])
def train():
    try:
        os.system("python main.py")
        return "Training successful!"
    except Exception as e:
        return f"Error Occurred! {e}"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        context = request.form.get("context", "").strip()
        question = request.form.get("question", "").strip()

        if not context or not question:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"error": "Both context and question are required"}), 400
            return render_template("index.html", 
                                answer="⚠ Please provide both context and question",
                                context=context,
                                question=question)

        answer = qa_pipeline.predict(context, question)

        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"answer": answer})

        return render_template("index.html", 
                            answer=answer,
                            context=context,
                            question=question)

    except Exception as e:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": str(e)}), 500
        return render_template("index.html", 
                            answer=f"Error: {str(e)}",
                            context=context,
                            question=question)

if __name__ == "__main__":
    app.run(debug=True, port=8080)