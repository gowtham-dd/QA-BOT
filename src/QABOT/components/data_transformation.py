import nltk
from src.QABOT.constant import *
from src.QABOT.utils.common import read_yaml,create_directories 
import json
from src.QABOT import logger
import joblib

from src.QABOT.entity.config_entity import DataTransformationConfig

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import json
import os
from src.QABOT import logger
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import Dataset
from tqdm import tqdm



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')  # Specifically check for punkt_tab
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)  # Download punkt_tab specifically
          
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased",
            use_auth_token=""  # Replace with your actual token
        )

    def flatten_squad(self, squad_js: Dict) -> List[Dict]:
        """Convert SQuAD format to flattened QA pairs"""
        rows = []
        for article in squad_js["data"]:
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    if "answers" not in qa or len(qa["answers"]) == 0:
                        continue
                    ans = qa["answers"][0]
                    rows.append({
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": context,
                        "answer_text": ans["text"],
                        "answer_start": ans["answer_start"],
                    })
        return rows

    def collect_paragraphs(self, squad_list: List[Dict]) -> List[str]:
        """Extract all unique paragraphs from SQuAD data"""
        paras = []
        for article in squad_list:
            for para in article["paragraphs"]:
                paras.append(para["context"])
        return list(dict.fromkeys(paras))  # Deduplicate

    def build_tfidf_retriever(self, paragraphs: List[str]) -> Tuple:
        """Create TF-IDF vectorizer and vectorize paragraphs"""
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(paragraphs)
        return vectorizer, tfidf_matrix

    def build_sentence_bank(self, paragraphs: List[str]) -> Tuple:
        """Create sentence-level TF-IDF index"""
        all_sentences = []
        sent2para_idx = []
        for p_i, p in enumerate(paragraphs):
            sents = sent_tokenize(p)
            all_sentences.extend(sents)
            sent2para_idx.extend([p_i]*len(sents))
        
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(all_sentences)
        return vectorizer, tfidf_matrix, all_sentences

    def prepare_features(self, rows: List[Dict]) -> List[Dict]:
        """Convert QA pairs to model input features with overflow handling"""
        features = []
        for r in tqdm(rows, desc="Tokenizing examples"):
            question = r["question"]
            context = r["context"]
            answer_text = r["answer_text"]
            answer_start = r["answer_start"]
            answer_end = answer_start + len(answer_text)

            tokenized = self.tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=self.config.MAX_LEN,
                stride=self.config.DOC_STRIDE,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            for i in range(len(tokenized["input_ids"])):
                offsets = tokenized["offset_mapping"][i]
                input_ids = tokenized["input_ids"][i]
                attention_mask = tokenized["attention_mask"][i]
                seq_ids = tokenized.sequence_ids(i)

                # Find context span
                context_start = next((idx for idx, s in enumerate(seq_ids) if s == 1), None)
                context_end = next((idx for idx in reversed(range(len(seq_ids))) if seq_ids[idx] == 1), None)

                if context_start is None or context_end is None:
                    continue

                # Locate answer span
                cls_index = input_ids.index(self.tokenizer.cls_token_id)
                start_token = end_token = cls_index

                for idx in range(context_start, context_end + 1):
                    start_char, end_char = offsets[idx]
                    if start_char <= answer_start and end_char >= answer_start:
                        start_token = idx
                    if start_char <= answer_end-1 and end_char >= answer_end-1:
                        end_token = idx

                if start_token != cls_index or end_token != cls_index:
                    features.append({
                        "input_ids": torch.tensor(input_ids),
                        "attention_mask": torch.tensor(attention_mask),
                        "start_positions": torch.tensor(start_token),
                        "end_positions": torch.tensor(end_token),
                    })
        return features

    def transform(self):
        """Execute full transformation pipeline"""
        try:
            if os.path.exists(self.config.root_dir) and os.listdir(self.config.root_dir):
                logger.info(f" Skipping transformation — found existing data in {self.config.root_dir}")
                # If skipping, you could optionally load artifacts instead of returning datasets
                return None, None
            # Load raw data
            with open(self.config.TRAIN_FILE, "r", encoding="utf-8") as f:
                train_squad = json.load(f)
            with open(self.config.DEV_FILE, "r", encoding="utf-8") as f:
                dev_squad = json.load(f)

            # Flatten QA pairs
            train_rows = self.flatten_squad(train_squad)
            dev_rows = self.flatten_squad(dev_squad)
            logger.info(f"Flattened {len(train_rows)} train and {len(dev_rows)} dev examples")

            # Build retrieval corpus
            all_paragraphs = self.collect_paragraphs(train_squad["data"]) + self.collect_paragraphs(dev_squad["data"])
            para_vectorizer, para_tfidf = self.build_tfidf_retriever(all_paragraphs)
            sent_vectorizer, sent_tfidf, all_sentences = self.build_sentence_bank(all_paragraphs)
            logger.info(f"Built TF-IDF retriever with {len(all_paragraphs)} paragraphs and {len(all_sentences)} sentences")

            # Prepare model features
            train_features = self.prepare_features(train_rows)
            dev_features = self.prepare_features(dev_rows)
            logger.info(f"Prepared {len(train_features)} train and {len(dev_features)} dev features")

            # Save artifacts
            os.makedirs(self.config.root_dir, exist_ok=True)
            joblib.dump(para_vectorizer, self.config.PARA_VECTORIZER_FILE)
            joblib.dump(sent_vectorizer, self.config.SENT_VECTORIZER_FILE)
            with open(self.config.PARAGRAPHS_FILE, "w", encoding="utf-8") as f:
                json.dump(all_paragraphs, f, ensure_ascii=False)
            with open(self.config.SENTENCES_FILE, "w", encoding="utf-8") as f:
                json.dump(all_sentences, f, ensure_ascii=False)
            
            return QADataset(train_features), QADataset(dev_features)

        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            raise e

class QADataset(Dataset):
    """PyTorch Dataset for QA features"""
    def __init__(self, features):
        self.features = features
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx]
