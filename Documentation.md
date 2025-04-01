# Documentation for Zindi International Women's Day Challenge Submission

## Overview and Objectives
### Purpose
This solution addresses the Zindi International Women's Day Challenge, aiming to predict a continuous target variable using a dataset provided by the competition. The primary purpose is to deliver an accurate and reproducible machine learning model that can be evaluated on Zindi's leaderboard and potentially implemented by the hosts.

### Objectives
- Develop a robust regression model leveraging ensemble techniques.
- Optimize model performance through hyperparameter tuning.
- Ensure reproducibility and clarity for Zindi's code review team.
- Produce a submission file (`submission.csv`) with predictions in the required format (ward, target).

### Expected Outcomes
- High accuracy on the competition leaderboard (measured by Mean Squared Error).
- A well-documented, maintainable codebase suitable for review and potential production use.

## Architecture Diagram
+---------------------+       +---------------------------------+
| Data Sources        |       | ETL: load_and_prepare_data()    |
| - Train.csv         | ----> | - Align train/test datasets     |
| - Test.csv          |       | - Drop identifiers (ward, ADM4) |
| - SampleSubmission  |       | - Separate target               |
+---------------------+       +---------------------------------+
         |                            |
         v                            v
+---------------------+       +---------------------------------+
| Hyperparameter      | <---- | Base Models: train_base_models()|
| Tuning:             |       | - Standalone CatBoost (rs=29)   |
| optimize_catboost() |       | - StackingRegressor (rs=65)     |
| - Optuna (50 trials)|       | - StackingRegressor (rs=27)     |
| - Tune CatBoost     |       +---------------------------------+
+---------------------+                |
         |                            v
         |                    +---------------------------------+
         +------------------> | Blending: blend_predictions() |
                              | - Average stacking predictions |
                              | - Blend with CatBoost          |
                              | - Final Ridge + CatBoost       |
                              +---------------------------------+
                                       |
                                       v
                              +---------------------+
                              | Output:             |
                              | submission.csv      |
                              | - ward, target      |
                              +---------------------+

- **Data Flow**: Starts with raw CSV files, proceeds through preprocessing, modeling, and ends with prediction output.
- **Components**: ETL (data prep), modeling (optimization and training), inference (blending).

## ETL Process
### Extract
- **Data Sources**: 
  - `Train.csv`: Training data with features and target.
  - `Test.csv`: Test data for predictions.
  - `SampleSubmission.csv`: Template for submission format.
- **Location**: `/kaggle/input/international-womens-day-challengedataset/`
- **Format**: CSV files.
- **Method**: Loaded using `pandas.read_csv()`.
- **Considerations**: Assumes static data; no real-time extraction needed.

### Transform
- **Logic**: 
  - Aligns train and test datasets to ensure consistent columns using `train.align(test, join='inner', axis=1)`.
  - Adds a `separator` column to distinguish train (0) and test (1) data during concatenation.
  - Drops identifier columns (`ward`, `ADM4_PCODE`) and separates the target variable.
- **Cleansing**: No explicit missing value handling; assumes competition data is pre-cleaned.
- **Preprocessing**: All transformations occur within `load_and_prepare_data()` to prepare data for modeling.

### Load
- **Storage**: Data is loaded into memory as pandas DataFrames (`X`, `y`, `X_test`).
- **Mechanism**: No persistent storage; processed data is passed directly to modeling functions.
- **Optimization**: In-memory processing suitable for the dataset size (assumed to fit within 16GB RAM).

## Data Modeling
### Data Model
- **Description**: Tabular data with numerical features and a continuous target variable.
- **Assumptions**: Features are predictive of the target; no significant multicollinearity issues.

### Feature Engineering
- **Selection**: All features except `ward` and `ADM4_PCODE` are used.
- **Engineering**: No additional feature creation; relies on raw features.
- **Normalization**: Not explicitly applied; models like CatBoost and tree-based methods handle scale internally.

### Model Training
- **Algorithms**:
  1. **CatBoostRegressor**: Standalone and in stacking/blending.
  2. **XGBRegressor**: Base estimator in stacking.
  3. **LinearRegression**: Base estimator in stacking.
  4. **RandomForestRegressor**: Base estimator in stacking.
  5. **LGBMRegressor**: Base estimator in stacking.
  6. **SVR**: Base estimator in stacking.
  7. **Lasso**: Base estimator in stacking.
  8. **KNeighborsRegressor**: Base estimator in stacking.
  9. **Ridge**: Final blending.
  10. **StackingRegressor**: Ensemble method.
- **Hyperparameters**:
  - CatBoost: Tuned via Optuna (e.g., `iterations`, `depth`, `learning_rate`).
  - Others: Default settings with `random_state=SEED` for reproducibility.
- **Process**: 
  - `optimize_catboost()`: 50 trials to minimize MSE.
  - `train_base_models()`: Trains standalone CatBoost and two StackingRegressors.
  - `blend_predictions()`: Combines predictions using Ridge and CatBoost.
- **Evaluation Metrics**: MSE used during optimization; final performance assessed via Zindi leaderboard.

### Model Validation
- **Method**: 30% validation split in `optimize_catboost()` with `random_state=SEED`.
- **Performance**: Measured by MSE; best parameters selected from Optuna trials.

## Inference
### Deployment
- **Infrastructure**: Runs in Kaggle Notebooks; no external deployment.
- **Services**: Pure Python script with library dependencies.

### Input/Output
- **Input**: Test features (`X_test`) from `load_and_prepare_data()`.
- **Output**: Predictions saved as `submission.csv` with `ward` and `target` columns.
- **Interpretation**: Continuous values representing the predicted target.

### Updates
- **Versioning**: Static script; no versioning implemented.
- **Retraining**: Not applicable; designed for one-time submission.

## Run Time
- **Total Script**: Approximately 5-10 minutes in Kaggle Notebooks (varies with dataset size).
- **Components**:
  - `load_and_prepare_data()`: ~10 seconds.
  - `optimize_catboost()`: ~4-8 minutes (50 trials).
  - `train_base_models()`: ~1-2 minutes (three model fits).
  - `blend_predictions()`: ~30 seconds.
- **Models**:
  - Standalone CatBoost: ~20 seconds.
  - Each StackingRegressor: ~40 seconds.
  - Ridge/CatBoost blending: ~15 seconds each.

## Performance Metrics
- **ETL**: No specific metrics; success measured by correct data loading.
- **Model Accuracy**: 
  - **MSE**: Reported from Optuna optimization (e.g., best value printed).
  Hints: Best MSE from optimization indicates validation performance; actual leaderboard scores not available here.
- **Public/Private Scores**: To be determined post-submission on Zindi.
- **Other Metrics**: None used; focus was on MSE for optimization.

## Error Handling and Logging
- **Error Handling**: Minimal; assumes data and environment are stable. Exceptions may halt execution (e.g., file not found).
- **Logging**: Basic print statements for optimization results and completion; no formal logging.

## Maintenance and Monitoring
- **Monitoring**: Manual review of output file and console messages.
- **Maintenance**: Script is single-use; updates would require manual code changes.
- **Scaling**: Designed for current dataset size; scaling to larger data may require memory optimization or distributed computing.

## Design Decisions
- **Ensemble Approach**: Combines diverse models to improve robustness.
- **CatBoost Focus**: Chosen for its handling of categorical data and strong performance.
- **Blending**: Weighted averaging simplifies combination while leveraging multiple predictions.
- **Kaggle**: Matches competition environment for reproducibility.

## Implementation Details
- **Reproducibility**: Fixed `SEED=42` throughout.
- **Dependencies**: Specified in `requirements.txt` with exact versions.
- **Code Style**: Modular functions with docstrings, following Zindi guidelines.

## Potential Issues
- **Memory**: May exceed 16GB RAM with larger datasets.
- **Run Time**: Optimization could be slow; adjust `n_trials` if needed.
- **Overfitting**: Multiple models might overfit; validation MSE monitors this.

## Best Practices
- All preprocessing in-script.
- Clear function names and documentation.
- Only free, accessible libraries used.

## Notes
- EDA could enhance feature understanding but omitted for brevity.
- Assumes competition data quality; additional cleaning may improve results.

## Contact
For questions, contact: Khutso Mphelo at khutsomphelo@gmail.com