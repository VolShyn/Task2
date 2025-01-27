import warnings # to ignore sklearn warning 'force_all_finite' was renamed to 'ensure_all_finite'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
import numpy as np
import pickle
import argparse
import json


from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from scipy.stats import randint, uniform
from sklearn.metrics import mean_squared_error

def load_param_distributions(param_file=None):
    if param_file:
        with open(param_file, 'r') as f:
            return json.load(f)
    return {
        'num_leaves': randint(20, 50),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(100, 500),
        'min_child_samples': randint(20, 40),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 2),
        'reg_lambda': uniform(0, 2)
    }

def main(args):
    # Read data
    df = pd.read_csv(args.input_path)
    
    X = df.drop([args.target_column], axis=1)
    y = df[args.target_column]

    # Min-Max normalization
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=args.test_size, random_state=args.random_state
    )

    # Load parameter distributions
    param_distributions = load_param_distributions(args.param_file)

    # init base model
    base_model = lgb.LGBMRegressor(
    random_state=42, # reproducibility
    verbose=-1,  # added because of warnings
    min_data_in_leaf=1  # allow smaller leaf sizes
)

    # cross-validation
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    # random search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        verbose=args.verbose
    )

    # Fit random search
    random_search.fit(X_train, y_train)

    # eval on test set
    best_model = random_search.best_estimator_
    test_score = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
    print("Test RMSE:", test_score)

    if args.print_params:
        print("best parameters:", random_search.best_params_)
        print("best cross-val. score:", np.sqrt(-random_search.best_score_))

    if args.print_importance:
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 important features:")
        print(feature_importance.head(10))

    if args.save:
        # creating model directory if it doesn't exist
        import os
        os.makedirs(args.model_dir, exist_ok=True)
        
        # saving model and associated objects
        with open(f'{args.model_dir}/best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)

        with open(f'{args.model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        with open(f'{args.model_dir}/best_params.pkl', 'wb') as f:
            pickle.dump(random_search.best_params_, f)

        print("model saved: ", args.model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LightGBM model with RandomizedSearchCV')
    
    # required argument
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input CSV file')
    
    # optional arguments
    parser.add_argument('--target_column', type=str, default='target',
                        help='target column in CSV')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='test set size ratio')
    parser.add_argument('--random_state', type=int, default=42,
                        help='random state for reproducibility')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='num of cross-validation folds')
    parser.add_argument('--n_iter', type=int, default=50,
                        help='num of random search iterations')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='num of parallel jobs')
    parser.add_argument('--verbose', type=int, default=2,
                        help='Verbosity level')
    
    # flags
    parser.add_argument('--save', action='store_true',
                        help='save model and associated files')
    parser.add_argument('--print_importance', action='store_true',
                        help='print feature importance')
    parser.add_argument('--print_params', action='store_true',
                        help='print best parameters and CV score')
    
    # paths
    parser.add_argument('--model_dir', type=str, default='model',
                        help='dir to save model files')
    parser.add_argument('--param_file', type=str,
                        help='JSON file with parameter distributions for random search')

    args = parser.parse_args()
    main(args)