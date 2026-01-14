import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation 
from src.components.data_transformation import DataTransformationConfig   


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, data_file_path: str):
        logging.info("Entered the data ingestion metadhod or component")
        try:
            # validate provided path
            if not data_file_path:
                raise CustomException("No data file path provided", sys)

            if not os.path.exists(data_file_path):
                raise CustomException(f"Data file not found: {data_file_path}", sys)

            df = pd.read_csv(data_file_path)
            logging.info("Dataset read as pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-test split done")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Ingestion of data completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.model_trainer import ModelTrainer

    # accept CLI arg or default to notebook/data/stud.csv
    data_path = sys.argv[1] if len(sys.argv) > 1 else "notebook/data/stud.csv"
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion(data_path)

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
    logging.info("Initiated data transformation")

    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    logging.info(f"Model training completed. Best model R2 score: {r2_score:.4f}")
