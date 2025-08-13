
from src.QABOT.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.QABOT import logger
# from src.QABOT.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
# from src.QABOT.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
# from src.QABOT.pipeline.model_training_pipeline import ModelTrainingPipeline
# from src.QABOT.pipeline.model_evaluation_pipeline import ModelEvaluationTrainingPipeline

STAGE_NAME="Data Ingestion stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e
