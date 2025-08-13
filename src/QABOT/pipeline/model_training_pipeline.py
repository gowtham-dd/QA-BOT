from src.QABOT.components.data_transformation import DataTransformation







from src.QABOT.config.configuration import ConfigurationManager
from src.QABOT.components.model_training import ModelTrainer
from src.QABOT import logger


STAGE_NAME="Model training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):

        try:
        # Initialize configuration
            config_manager = ConfigurationManager()
            trainer_config = config_manager.get_model_trainer_config()
        
        # Load datasets (from previous transformation step)
            transformer = DataTransformation(config_manager.get_data_transformation_config())
            train_dataset, _ = transformer.transform()
        
        # Train model
            trainer = ModelTrainer(
            config=trainer_config,
            train_dataset=train_dataset
            )
            trainer.train()

        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise e

if __name__ == "__main__":
     try:
          logger.info(f">>>> Stage {STAGE_NAME} started")
          obj=ModelTrainingPipeline()
          obj.main()
          logger.info(f">>>>> Stage {STAGE_NAME} completed")

     except Exception as e:
          logger.exception(e)
          raise e
