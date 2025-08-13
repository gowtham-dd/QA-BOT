


from src.QABOT.components.data_transformation import DataTransformation
from src.QABOT.config.configuration import ConfigurationManager
from src.QABOT.components.model_evaluation import ModelEvaluationConfig
from src.QABOT import logger
from pathlib import Path
from src.QABOT.components.model_evaluation import ModelEvaluator


STAGE_NAME="Model Evaluation stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):

    # Example usage
        config = ModelEvaluationConfig(
        root_dir=Path("artifacts/model_evaluation"),
        tfidf_model_dir=Path("artifacts/model_trainer/tfidf"),
        distilbert_model_dir=Path("artifacts/model_trainer/distilbert"),
        metrics_file=Path("artifacts/model_evaluation/metrics.json"),
        test_data_path=Path("artifacts/data_ingestion/dev-v1.1.json")
        )
    
        evaluator = ModelEvaluator(config)
        metrics = evaluator.evaluate_and_save()
        print(metrics)


if __name__ == "__main__":
     try:
          logger.info(f">>>> Stage {STAGE_NAME} started")
          obj=ModelEvaluationPipeline()
          obj.main()
          logger.info(f">>>>> Stage {STAGE_NAME} completed")

     except Exception as e:
          logger.exception(e)
          raise e
