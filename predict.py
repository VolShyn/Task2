import pandas as pd
import numpy as np
import pickle
import argparse

def load_model(model_path, scaler_path):
    """Load the model and scaler from files"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def make_predictions(df, model, scaler):
    """Transform data and make predictions"""
    X_new = scaler.transform(df)
    return model.predict(X_new)

def main(args):
    test = pd.read_csv(args.input_path)
    
    model, scaler = load_model(args.model_path, args.scaler_path)
    
    predictions = make_predictions(test, model, scaler)
    
    np.savetxt(args.output_path, predictions, delimiter=args.delimiter)
    print(f"pred saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions using trained LightGBM model')
    
    # requires arguments
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input CSV file')
    
    # additional, kind of)
    parser.add_argument('--model_path', type=str, default='best_model.pkl',
                        help='Path to pickled model file')
    parser.add_argument('--scaler_path', type=str, default='scaler.pkl',
                        help='Path to pickled scaler file')
    
    parser.add_argument('--output_path', type=str, default='predictions.csv',
                        help='Path to save predictions')
    parser.add_argument('--delimiter', type=str, default=',',
                        help='Delimiter for output CSV file')

    args = parser.parse_args()
    main(args)