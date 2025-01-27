In our project, we were given the data with hidden features. 

We've explored data in `eda.ipynb` and concluded that we need to solve the regression problem.

The [LightGBM](https://lightgbm.readthedocs.io/en/stable/) was trained to fit our problem; 

# Train

[Data](https://drive.google.com/drive/folders/1pgimCOWUUPbjwxcZQE78dgV4CAntqyRI?usp=sharing) used to train the model. 

We train/test split our data and use cross-validation to prevent overfitting and maximize generalisation ability. 

Additionally, `gradient boosting` models are sensitive to hyperparameters, which is why we also use randomized search.

Algorithm of how to train the model:

1. Create virtual environment
2. Activate venv
3. pip install requirements.txt
4. ```python main.py --input_path train.csv```

You can also specify a lot of hyperparameters if you want to (arguments can be found in `main.py`);

After training, our model is saved to the `.pkl` file, which can then be used to infer the model. 

We've chosen the `.pkl` data format, as it is very efficient, already installed in Python, and creates a dump of our code in byte-code (instrumental in our case because we can't save just hyperparameters in some `.jsonl`, as we will need to `.fit` our model once again);

The biggest problem with `.pkl` is that it's unsafe; however, we don't need safety in our project! :)

# Prediction

Only for LightGBM;

We are taking previously saved `.pkl` files (i.e., our saved model), UNPICKLE it, and do inference: 

here's an example:
```python predict.py \
    --input_path data/hidden_test.csv \
    --model_path model/best_model.pkl \
    --scaler_path model/scaler.pkl 
```

# Conclusions: 

On the test set, we got `RMSE: 0.02564208507898708`, which is a perfect result! 

Looking at the scores on every step of cross-validation, we can say that our model is not overtraining;

Also, because under the hood, we have an RF model, it is interesting to look at our feature importance:


Here's the data formatted as a markdown table:

| Feature | Importance |
|---------|------------|
| 6       | 3693      |
| 7       | 2998      |
| 0       | 393       |
| 27      | 391       |
| 50      | 385       |
| 28      | 385       |
| 12      | 383       |
| 5       | 383       |
| 44      | 380       |
| 41      | 379       |

we can see , that features `6` and `7` are the most important ones (which is a little bit predictable), other features are useful but not decisive, yet, it is preferable to have as much features as possible to train our model;


