from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_URL:str
    local_data_file:Path
    unzip_dir:Path



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    TRAIN_FILE: str
    DEV_FILE: str
    REQUIRED_KEYS: List[str]  # Required keys in SQuAD format
    MIN_EXAMPLES: int  # Minimum expected examples in dataset


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    TRAIN_FILE: Path
    DEV_FILE: Path
    MAX_LEN: int
    DOC_STRIDE: int
    PARA_VECTORIZER_FILE: Path
    SENT_VECTORIZER_FILE: Path
    PARAGRAPHS_FILE: Path
    SENTENCES_FILE: Path



@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_save_dir: Path
    trained_model_path: Path
    tokenizer_save_dir: Path
    num_train_epochs: int
    batch_size: int
    learning_rate: float



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    tfidf_model_dir: Path
    distilbert_model_dir: Path
    metrics_file: Path
    test_data_path: Path