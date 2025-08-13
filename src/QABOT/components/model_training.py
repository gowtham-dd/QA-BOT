from dataclasses import dataclass
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader, Dataset  # Added Dataset import
from torch.optim import AdamW
from tqdm import tqdm
from src.QABOT import logger
import shutil
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast  # Added tokenizer import
from src.QABOT.entity.config_entity import ModelTrainerConfig
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast  # Added tokenizer import



class ModelTrainer:
    def __init__(
        self,
        config: ModelTrainerConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset = None
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    

    def _check_model_exists(self) -> bool:
        """Check if trained model already exists"""
        return (
            os.path.exists(self.config.trained_model_path) and 
            os.path.exists(self.config.tokenizer_save_dir)
        )

    def train(self):
        try:
            # Skip training if model exists
            if self._check_model_exists():
                logger.info("Model already trained and saved. Skipping training.")
                return

            logger.info("Starting model training...")
            
            # Create fresh output directory
            if os.path.exists(self.config.model_save_dir):
                shutil.rmtree(self.config.model_save_dir)
            os.makedirs(self.config.model_save_dir, exist_ok=True)

            # Initialize model
            model = DistilBertForQuestionAnswering.from_pretrained(
                "distilbert-base-uncased"
            ).to(self.device)
            
            # Setup training
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            optimizer = AdamW(
                model.parameters(),
                lr=self.config.learning_rate
            )

            # Training loop
            model.train()
            for epoch in range(self.config.num_train_epochs):
                loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_train_epochs}")
                for batch in loop:
                    optimizer.zero_grad()
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    loop.set_postfix(loss=loss.item())

            # Save model
            model.save_pretrained(self.config.model_save_dir)
            tokenizer.save_pretrained(self.config.tokenizer_save_dir)
            logger.info(f"✅ Model trained and saved to {self.config.model_save_dir}")

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise e
