from src.QABOT.constant import *
from src.QABOT.utils.common import read_yaml,create_directories 
from src.QABOT.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvaluationConfig
class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
     ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=Path(config.STATUS_FILE),
            TRAIN_FILE=Path(config.TRAIN_FILE),
            DEV_FILE=Path(config.DEV_FILE),
            REQUIRED_KEYS=config.REQUIRED_KEYS,
            MIN_EXAMPLES=config.MIN_EXAMPLES
        )
        return data_validation_config
    




    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            TRAIN_FILE=Path(config.TRAIN_FILE),
            DEV_FILE=Path(config.DEV_FILE),
            MAX_LEN=self.params.Transformation.MAX_LEN,
            DOC_STRIDE=self.params.Transformation.DOC_STRIDE,
            PARA_VECTORIZER_FILE=Path(config.PARA_VECTORIZER_FILE),
            SENT_VECTORIZER_FILE=Path(config.SENT_VECTORIZER_FILE),
            PARAGRAPHS_FILE=Path(config.PARAGRAPHS_FILE),
            SENTENCES_FILE=Path(config.SENTENCES_FILE)
        )
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.Training

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            model_save_dir=Path(config.model_save_dir),
            trained_model_path=Path(config.trained_model_path),
            tokenizer_save_dir=Path(config.tokenizer_save_dir),
            num_train_epochs=params.num_train_epochs,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate
        )
        return model_trainer_config



    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            tfidf_model_dir=Path(config.tfidf_model_dir),
            distilbert_model_dir=Path(config.distilbert_model_dir),
            metrics_file=Path(config.metrics_file),
            test_data_path=Path(config.test_data_path)
        )