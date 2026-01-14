import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self, data_path: str = "notebook/data/stud.csv"):
        try:
            logging.info("Starting training pipeline")

            # Data Ingestion
            logging.info("Initiating data ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion(data_path)

            # Data Transformation
            logging.info("Initiating data transformation")
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )

            # Model Training
            logging.info("Initiating model training")
            model_trainer = ModelTrainer()
            r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"Training pipeline completed. Best model R2 score: {r2_score:.4f}")
            return r2_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "notebook/data/stud.csv"
    pipeline = TrainPipeline()
    score = pipeline.run_pipeline(data_path)
    print(f"Training completed. Best model R2 score: {score:.4f}")
