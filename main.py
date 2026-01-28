import argparse
import uvicorn
from src.train import train_model
from src.predict import predict_price
from src.api import app

def main():
    parser = argparse.ArgumentParser(description="House Price Prediction Project")
    parser.add_argument('--action', type=str, required=True, choices=['train', 'predict', 'api'],
                        help="Action to perform: 'train', 'predict', or 'api'")
    args = parser.parse_args()

    if args.action == 'train':
        print("Starting model training...")
        train_model()
        print("Model training complete.")
    elif args.action == 'predict':
        print("Starting prediction...")
        predict_price()
        print("Prediction complete.")
    elif args.action == 'api':
        print("Starting API server...")
        # Note: In a real production environment, you might run this differently,
        # but this is convenient for development.
        uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == '__main__':
    main()