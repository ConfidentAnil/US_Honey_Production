import numpy as np
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from mlxtend.regressor import StackingCVRegressor


def parameter_grids(model):
    """
    Returns the possible values of the hyperparameters used by a specific model 
    for hyperparameter tuning.

    Parameters:
    model (sklearn estimator): The machine learning model to be trained.

    Returns:
    dict: Parameter Grid.
    """
    # RandomForest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'random_state':[42],
        'n_jobs':[-1]
    }

    # XGBoost
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'min_child_weight': [1, 5, 10],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'lambda': [0, 0.1, 1.0],  # L2 regularization term.
        'alpha': [0, 0.1, 1.0],   # L1 regularization term.
        'random_state':[42],
        'n_jobs':[-1]
    }

    # Gradient Boosting
    gb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'criterion': ['friedman_mse', 'squared_error'],
        'random_state':[42],
    }
    
    # Light Gradient Boosting Machine
    lgbm_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [-1, 3, 6, 9],
        'num_leaves': [31, 50, 100],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0],
        'boosting_type': ['gbdt', 'dart', 'goss'],
        'objective': ['regression', 'regression_l1', 'huber', 'fair'],
        'verbose':[-1],
        'force_col_wise':[True],
        'random_state':[42],
        'n_jobs':[-1]
    }

    # Linear Regression
    lr_param_grid = {'n_jobs':[-1]}  # No hyperparameters to tune for LinearRegression
    
    # Ridge Regression
    ridge_param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
    }
    
    # Lasso Regression
    lasso_param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'selection': ['random', 'cyclic']
    }
    
    # Support Vector Regression (SVR)
    svr_param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [3, 4, 5],  # Only relevant for 'poly' kernel
        'gamma': ['scale', 'auto']  # Only relevant for 'rbf', 'poly', 'sigmoid' kernels
    }

    try:
        if model.__class__.__name__ == "RandomForestRegressor":
            return rf_param_grid
        elif model.__class__.__name__ == "XGBRegressor":
            return xgb_param_grid
        elif model.__class__.__name__ == "GradientBoostingRegressor":
            return gb_param_grid
        elif model.__class__.__name__ == "LGBMRegressor":
            return lgbm_param_grid
        elif model.__class__.__name__ == "LinearRegression":
            return lr_param_grid
        elif model.__class__.__name__ == "Ridge":
            return ridge_param_grid
        elif model.__class__.__name__ == "Lasso":
            return lasso_param_grid
        elif model.__class__.__name__ == "SVR":
            return svr_param_grid
    except:
        raise Exception ('''Learning Model unavailable, please chose from: 
                         1. LinearRegression
                         2. Ridge
                         3. Lasso 
                         4. SVR
                         5. RandomForestRegressor
                         6. GradientBoostingRegressor
                         7. XGBRegressor
                         8. LGBMRegressor ''')




def log_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between the logarithm of the predicted value and
    the logarithm of the observed sales price.

    Parameters:
    y_true (array-like): The actual observed sales prices.
    y_pred (array-like): The predicted sales prices.

    Returns:
    float: The RMSE between the logarithm of the predicted and observed sales prices.
    """
    # Ensure the inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute the log of the values
    log_y_true = np.log(y_true)
    log_y_pred = np.log(y_pred)
    
    # Calculate the RMSE
    log_rmse = np.sqrt(mean_squared_error(log_y_true, log_y_pred))
    
    return log_rmse



def hyperparameter_tuning_with_grid_search(model, X, y):
    """
    Perform hyperparameter tuning using GridSearchCV with a custom scoring metric.

    Parameters:
    model (sklearn estimator): The machine learning model to tune.
    X (array-like): Features of the data.
    y (array-like): Target variable of the data.

    Returns:
    list: A list containing the best parameters found by GridSearchCV and the time taken for tuning.
    """
    start = time.time()

    # Get the dict for parameter grid
    param_grid = parameter_grids(model)

    # Create a custom scorer
    log_rmse_scorer = make_scorer(log_rmse, greater_is_better=False)
    
    grid_search = GridSearchCV(model, 
                               param_grid=param_grid, 
                               cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
                               scoring=log_rmse_scorer,
                               return_train_score=True,
                               verbose=2, n_jobs=-1)
    grid_search.fit(X, y)

    time_taken = round(time.time() - start, 6)
    print(f"Hyperparameter tuning completed in {time_taken} seconds.\n")

    return [grid_search.best_params_, time_taken]



def hyperparameter_tuning_with_random_search(model, X, y):
    """
    Perform hyperparameter tuning using RandomSearchCV with a custom scoring metric.

    Parameters:
    model (sklearn estimator): The machine learning model to tune.
    X (array-like): Features of the data.
    y (array-like): Target variable of the data.

    Returns:
    list: A list containing the best parameters found by RandomSearchCV and the time taken for tuning.
    """
    start = time.time()
    
    # Get the dict for parameter grid
    param_grid = parameter_grids(model)

    # Create a custom scorer
    log_rmse_scorer = make_scorer(log_rmse, greater_is_better=False)

    random_search = RandomizedSearchCV(model, 
                                       param_distributions= param_grid, 
                                       cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42), 
                                       scoring=log_rmse_scorer,  # Use the custom log_rmse scorer
                                       return_train_score=True,
                                       verbose=2, n_jobs=-1)  # Provides more detailed output
    random_search.fit(X, y)

    time_taken = round(time.time() - start, 6)
    print(f"Hyperparameter tuning completed in {time_taken} seconds.")
    
    return [random_search.best_params_, time_taken]



def train_and_evaluate_model(model, X, y, tuning=False, CV="RandomSearch"):
    """
    Trains and evaluates a machine learning model with optional hyperparameter tuning.

    Parameters:
    model (sklearn estimator): The machine learning model to be trained.
    X (pd.DataFrame or np.ndarray): The feature matrix.
    y (pd.Series or np.ndarray): The target vector.
    tuning (bool): Whether to perform hyperparameter tuning. Default is False.
    CV (str): The type of cross-validation to use for tuning ("RandomSearch" or "GridSearch"). Default is "RandomSearch".

    Returns:
    list: A list containing the log-RMSE score and the total time taken (including tuning time if applicable).
    """
    to_print = "Training " + model.__class__.__name__
    gap = (75-len(to_print))//2
    print("*"*gap, to_print, "*"*gap)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter Tuning if requested
    tuning_time = 0
    if tuning:
        if CV == "RandomSearch":
            best_params, tuning_time = hyperparameter_tuning_with_random_search(model, X_train, y_train)
        elif CV == "GridSearch":
            best_params, tuning_time = hyperparameter_tuning_with_grid_search(model, X_train, y_train)
        else:
            raise Exception ("Unknown Technique for Cross Validation")
        
        # Update model with best parameters
        model.set_params(**best_params)

    start = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    score = log_rmse(y_pred, y_test)

    # Calculating time taken
    time_taken = round(time.time() - start, 6)
    print(f"Model training completed in {time_taken} seconds.")
    
    if tuning:
        time_taken += tuning_time
        print(f"Total time taken {time_taken} seconds.")
    
    print(f"Model performance (log-RMSE) : {score}.\n\n")  
    return [score, time_taken]



def train_stacked_ensemble(X, y, stack_list=None, meta_model=None, base_tuning=False, meta_tuning=False):
    """
    Trains and evaluates a stacked ensemble model with optional hyperparameter tuning for base models and meta-model.

    Parameters:
    X (pd.DataFrame or np.ndarray): The feature matrix.
    y (pd.Series or np.ndarray): The target vector.
    stack_list (list of sklearn estimators): List of base models for stacking. If None, default models are used.
    meta_model (sklearn estimator): The meta-model for stacking. If None, a default model is used.
    base_tuning (bool): Whether to perform hyperparameter tuning on base models.
    meta_tuning (bool): Whether to perform hyperparameter tuning on the meta-model.

    Returns:
    list: Contains model performance (log-RMSE) and time taken for training and evaluation.

    Available models:
        1. LinearRegression
        2. Ridge
        3. Lasso
        4. SVR
        5. RandomForestRegressor
        6. GradientBoostingRegressor
        7. XGBRegressor
        8. LGBMRegressor
    """
    start = time.time()

    to_print = "Training Stack Model"
    gap = (75 - len(to_print)) // 2
    print("*" * gap, to_print, "*" * gap)

    if stack_list is None:
        stack_list = [
            LinearRegression(),
            Ridge(),
            Lasso(),
            RandomForestRegressor(),
            XGBRegressor(),
        ]

    if meta_model is None:
        meta_model = GradientBoostingRegressor()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter Tuning for base models
    if base_tuning:
        print(f"Tuning Base Models")
        for i, model in enumerate(stack_list):
            best_params, _ = hyperparameter_tuning_with_random_search(model, X_train, y_train)
            stack_list[i].set_params(**best_params)

    # Hyperparameter Tuning for meta-model
    if meta_tuning:
        print(f"Tuning Meta Model")
        best_params, _ = hyperparameter_tuning_with_random_search(meta_model, X_train, y_train)
        meta_model.set_params(**best_params)

    # Train the stacked model
    stack = StackingCVRegressor(regressors=stack_list,
                                meta_regressor=meta_model,
                                cv=5,
                                use_features_in_secondary=True,
                                store_train_meta_features=True,
                                shuffle=False,
                                random_state=42,
                                n_jobs=-1)
    
    stack.fit(X_train, y_train)
    
    # Evaluation
    y_pred = stack.predict(X_test)
    error = log_rmse(y_pred, y_test)
    
    # Calculating time taken
    time_taken = round(time.time() - start, 6)
    print(f"Stack-model training completed in {time_taken} seconds.")
    print(f"Stack-model performance (log-RMSE) : {error}.\n\n")

    return [error, time_taken]