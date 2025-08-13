
from src.QABOT.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.QABOT import logger
from src.QABOT.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.QABOT.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.QABOT.pipeline.model_training_pipeline import ModelTrainingPipeline
from src.QABOT.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline


# HF API :
STAGE_NAME="Data Ingestion stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME="Data Validation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME="Data Transformation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME="Model Training stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME="Model Evaluation stage"


try:
    logger.info(f">>>> Stage {STAGE_NAME} started")
    obj=ModelEvaluationPipeline()
    obj.main()
    logger.info(f">>>>> Stage {STAGE_NAME} completed")

except Exception as e:
    logger.exception(e)
    raise e
