





from src.QABOT.config.configuration import ConfigurationManager
from src.QABOT.components.data_transformation import DataTransformation
from src.QABOT import logger


STAGE_NAME="Data Validation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):

        try:
            config_manager = ConfigurationManager()
            transformation_config = config_manager.get_data_transformation_config()
            transformer = DataTransformation(config=transformation_config)
            train_dataset, dev_dataset = transformer.transform()
            logger.info(" Data transformation completed successfully")
        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            raise e

if __name__ == "__main__":
     try:
          logger.info(f">>>> Stage {STAGE_NAME} started")
          obj=DataTransformationTrainingPipeline()
          obj.main()
          logger.info(f">>>>> Stage {STAGE_NAME} completed")

     except Exception as e:
          logger.exception(e)
          raise e
