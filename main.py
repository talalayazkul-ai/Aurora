import argparse
import sys
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import PredictPipeline, CustomData


def train(data_path: str = "notebook/data/stud.csv"):
    """Run the training pipeline."""
    pipeline = TrainPipeline()
    score = pipeline.run_pipeline(data_path)
    print(f"Training completed. Best model R2 score: {score:.4f}")
    return score


def predict(gender, race_ethnicity, parental_level_of_education, lunch,
            test_preparation_course, reading_score, writing_score):
    """Make a prediction using the trained model."""
    data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=int(reading_score),
        writing_score=int(writing_score),
    )
    df = data.get_data_as_dataframe()
    pipeline = PredictPipeline()
    result = pipeline.predict(df)
    print(f"Predicted math score: {result[0]:.2f}")
    return result[0]


def main():
    parser = argparse.ArgumentParser(description="Aurora ML Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--data", default="notebook/data/stud.csv", help="Path to training data"
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make a prediction")
    predict_parser.add_argument("--gender", required=True, choices=["male", "female"])
    predict_parser.add_argument(
        "--race", required=True,
        choices=["group A", "group B", "group C", "group D", "group E"]
    )
    predict_parser.add_argument(
        "--education", required=True,
        choices=["some high school", "high school", "some college",
                 "associate's degree", "bachelor's degree", "master's degree"]
    )
    predict_parser.add_argument("--lunch", required=True, choices=["standard", "free/reduced"])
    predict_parser.add_argument(
        "--test-prep", required=True, choices=["none", "completed"]
    )
    predict_parser.add_argument("--reading", required=True, type=int, help="Reading score (0-100)")
    predict_parser.add_argument("--writing", required=True, type=int, help="Writing score (0-100)")

    args = parser.parse_args()

    if args.command == "train":
        train(args.data)
    elif args.command == "predict":
        predict(
            gender=args.gender,
            race_ethnicity=args.race,
            parental_level_of_education=args.education,
            lunch=args.lunch,
            test_preparation_course=args.test_prep,
            reading_score=args.reading,
            writing_score=args.writing,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
