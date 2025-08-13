from src.QABOT import logger
from src.QABOT.entity.config_entity import DataValidationConfig
from typing import List, Dict
from pathlib import Path

import json


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.required_article_keys = {"title", "paragraphs"}
        self.required_para_keys = {"context", "qas"}
        self.required_qa_keys = {"id", "question", "answers"}
        self.required_answer_keys = {"text", "answer_start"}

    def validate_all(self) -> Dict[str, bool]:
        validation_status = {
            "train_file_exists": False,
            "dev_file_exists": False,
            "train_structure_valid": False,
            "dev_structure_valid": False,
            "train_min_examples": False,
            "dev_min_examples": False,
            "train_answer_spans_valid": False,
            "dev_answer_spans_valid": False
        }

        try:
            # File existence checks
            validation_status["train_file_exists"] = self._validate_file_exists(self.config.TRAIN_FILE)
            validation_status["dev_file_exists"] = self._validate_file_exists(self.config.DEV_FILE)

            if validation_status["train_file_exists"]:
                with open(self.config.TRAIN_FILE, "r", encoding="utf-8") as f:
                    train_data = json.load(f)
                    validation_status["train_structure_valid"] = self._validate_squad_structure(train_data)
                    validation_status["train_min_examples"] = self._validate_example_count(train_data)
                    validation_status["train_answer_spans_valid"] = self._validate_answer_spans(train_data)

            if validation_status["dev_file_exists"]:
                with open(self.config.DEV_FILE, "r", encoding="utf-8") as f:
                    dev_data = json.load(f)
                    validation_status["dev_structure_valid"] = self._validate_squad_structure(dev_data)
                    validation_status["dev_min_examples"] = self._validate_example_count(dev_data)
                    validation_status["dev_answer_spans_valid"] = self._validate_answer_spans(dev_data)

            # Write final status
            with open(self.config.STATUS_FILE, "w") as f:
                json.dump(validation_status, f, indent=4)

            logger.info(f"Data validation completed successfully")
            return validation_status

        except Exception as e:
            logger.info(f"Data validation failed: {str(e)}")
            raise e

    def _validate_file_exists(self, filepath: Path) -> bool:
        exists = filepath.exists()
        if not exists:
            logger.info(f"File not found: {filepath}")
        return exists

    def _validate_squad_structure(self, data: Dict) -> bool:
        """Validate the SQuAD format structure"""
        if not isinstance(data, dict):
            logger.error("Root element must be a dictionary")
            return False
        
        if not all(key in data for key in ["data", "version"]):
            logger.info("Missing required root keys: 'data' or 'version'")
            return False
        
        for article in data["data"]:
            if not all(key in article for key in self.required_article_keys):
                logger.info(f"Article missing required keys: {self.required_article_keys - set(article.keys())}")
                return False
            
            for para in article["paragraphs"]:
                if not all(key in para for key in self.required_para_keys):
                    logger.info(f"Paragraph missing required keys: {self.required_para_keys - set(para.keys())}")
                    return False
                
                for qa in para["qas"]:
                    if not all(key in qa for key in self.required_qa_keys):
                        logger.info(f"QA missing required keys: {self.required_qa_keys - set(qa.keys())}")
                        return False
                    
                    if "answers" in qa:
                        for ans in qa["answers"]:
                            if not all(key in ans for key in self.required_answer_keys):
                                logger.info(f"Answer missing required keys: {self.required_answer_keys - set(ans.keys())}")
                                return False
        return True

    def _validate_example_count(self, data: Dict) -> bool:
        """Validate minimum number of examples exists"""
        count = 0
        for article in data["data"]:
            for para in article["paragraphs"]:
                count += len(para["qas"])
        
        valid = count >= self.config.MIN_EXAMPLES
        if not valid:
            logger.info(f"Insufficient examples: {count} (minimum required: {self.config.MIN_EXAMPLES})")
        return valid

    def _validate_answer_spans(self, data: Dict) -> bool:
        """Validate answer spans match context text"""
        for article in data["data"]:
            for para in article["paragraphs"]:
                context = para["context"]
                for qa in para["qas"]:
                    if "answers" not in qa:
                        continue
                    for ans in qa["answers"]:
                        start = ans["answer_start"]
                        text = ans["text"]
                        if context[start:start+len(text)] != text:
                            logger.info(
                                f"Answer span mismatch. Expected '{text}' "
                                f"but found '{context[start:start+len(text)]}' at position {start}"
                            )
                            return False
        return True
