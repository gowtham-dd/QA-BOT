from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast
import torch
from src.QABOT import logger
from src.QABOT.utils.common import read_yaml, create_directories
from src.QABOT.entity.config_entity import ModelEvaluationConfig
import os
class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        create_directories([self.config.root_dir])
        self._should_evaluate = not os.path.exists(self.config.root_dir)  # Only evaluate if folder doesn't exist

    def _metrics_exist(self) -> bool:
        """Check if metrics file exists and is valid"""
        if not os.path.exists(self.config.metrics_file):
            return False
        
        try:
            with open(self.config.metrics_file, "r") as f:
                metrics = json.load(f)
                required_keys = {"tfidf", "distilbert", "test_samples"}
                return all(key in metrics for key in required_keys)
        except:
            return False

    def _load_tfidf_model(self):
        """Load TF-IDF vectorizer and sentences"""
        vectorizer = joblib.load(self.config.tfidf_model_dir / "tfidf_vectorizer.pkl")
        with open(self.config.tfidf_model_dir / "tfidf_sentences.json", "r") as f:
            sentences = json.load(f)
        return vectorizer, sentences

    def _load_distilbert_model(self):
        """Load DistilBERT model and tokenizer"""
        model = DistilBertForQuestionAnswering.from_pretrained(
            self.config.distilbert_model_dir
        ).to(self.device)
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.config.distilbert_model_dir
        )
        return model, tokenizer

    def _load_test_data(self):
        """Load test data in SQuAD format"""
        with open(self.config.test_data_path, "r") as f:
            return json.load(f)

    def _flatten_squad(self, squad_data):
        """Convert SQuAD format to list of QA pairs"""
        rows = []
        for article in squad_data["data"]:
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    if "answers" not in qa or len(qa["answers"]) == 0:
                        continue
                    ans = qa["answers"][0]
                    rows.append({
                        "question": qa["question"],
                        "context": context,
                        "answer_text": ans["text"],
                        "answer_start": ans["answer_start"]
                    })
        return rows

    def _exact_match(self, pred: str, truth: str) -> int:
        """Calculate exact match score"""
        pred = pred.strip().lower()
        truth = truth.strip().lower()
        return int(pred == truth)

    def evaluate_tfidf(self, test_data):
        """Evaluate TF-IDF model using exact match"""
        vectorizer, sentences = self._load_tfidf_model()
        em_scores = []
        
        for item in test_data:
            question = item["question"]
            gold_answer = item["answer_text"]
            
            # Vectorize question and find most similar sentence
            q_vec = vectorizer.transform([question])
            sims = cosine_similarity(q_vec, vectorizer.transform(sentences))[0]
            pred_answer = sentences[np.argmax(sims)].strip()
            
            em_scores.append(self._exact_match(pred_answer, gold_answer))
        
        return {"exact_match": float(np.mean(em_scores))}

    def evaluate_distilbert(self, test_data):
        """Evaluate DistilBERT model using exact match"""
        model, tokenizer = self._load_distilbert_model()
        em_scores = []
        
        model.eval()
        with torch.no_grad():
            for item in test_data:
                question = item["question"]
                context = item["context"]
                gold_answer = item["answer_text"]
                
                # Get model prediction
                inputs = tokenizer(question, context, return_tensors="pt").to(self.device)
                outputs = model(**inputs)
                
                # Convert to answer text
                start = torch.argmax(outputs.start_logits)
                end = torch.argmax(outputs.end_logits) + 1
                pred_answer = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end]))
                
                em_scores.append(self._exact_match(pred_answer, gold_answer))
        
        return {"exact_match": float(np.mean(em_scores))}

    def evaluate_and_save(self):
        """Run evaluation only if model_evaluation folder doesn't exist"""
        try:
            # Skip if evaluation folder exists
            if os.path.exists(self.config.root_dir):
                logger.info(f"Evaluation folder {self.config.root_dir} exists. Skipping evaluation.")
                if self._metrics_exist():
                    with open(self.config.metrics_file, "r") as f:
                        return json.load(f)
                return {"status": "evaluation_skipped", "reason": "folder_exists"}

            # Proceed with evaluation
            logger.info(" Starting new evaluation...")
            test_data = self._flatten_squad(self._load_test_data())
            
            # Evaluate both models
            tfidf_metrics = self.evaluate_tfidf(test_data)
            distilbert_metrics = self.evaluate_distilbert(test_data)
            
            # Combine and save metrics
            metrics = {
                "tfidf": tfidf_metrics,
                "distilbert": distilbert_metrics,
                "test_samples": len(test_data)
            }
            
            with open(self.config.metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f" Evaluation complete. Metrics saved to {self.config.metrics_file}")
            return metrics
            
        except Exception as e:
            logger.error(f" Evaluation failed: {str(e)}")
            raise e