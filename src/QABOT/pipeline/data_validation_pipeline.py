


from src.QABOT.config.configuration import ConfigurationManager
from src.QABOT.components.data_validation import DataValidation
from src.QABOT import logger


STAGE_NAME="Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):

        try:
            config_manager = ConfigurationManager()
            validation_config = config_manager.get_data_validation_config()
            validator = DataValidation(config=validation_config)
            status = validator.validate_all()
            print("Data Validation Status:")
            for k, v in status.items():
                print(f"{k}: {'✅' if v else '❌'}")
        except Exception as e:
            raise e

if __name__ == "__main__":
     try:
          logger.info(f">>>> Stage {STAGE_NAME} started")
          obj=DataValidationTrainingPipeline()
          obj.main()
          logger.info(f">>>>> Stage {STAGE_NAME} completed")

     except Exception as e:
          logger.exception(e)
          raise e
