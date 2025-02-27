# USER-ACTIVITY
# Machine Learning Model for Task Completion Prediction

## Project Overview
This project develops a machine learning model to predict the likelihood of users completing tasks based on activity patterns. The dataset includes user activity metrics such as app usage time, steps taken, sleep duration, and mood scores.

## Features and Target Variable
- **Features Used:**
  - `app_usage_time (minutes)`
  - `steps_taken`
  - `calories_burned`
  - `sleep_duration (hours)`
  - `mood_score`
  - `screen_time (minutes)`
  - `heart_rate (bpm)`
  - `hydration_level (liters)`
  - `stress_level (1-10)`
  - `day_of_week`
- **Target Variable:**
  - `task_completed` (1 = Likely to complete, 0 = Unlikely to complete)

## Model Development
- Data preprocessing includes feature extraction and standardization.
-StandardScaler is applied to normalize numerical features.
- A Random Forest Classifiers used for classification.
- The dataset is split into 70% training and 30% testing.

## Model Performance
- Accuracy:100% (due to small dataset size, possible overfitting)
- Precision, Recall, F1-score: Perfect scores due to dataset characteristics.
-Insights: The model performed exceptionally well on this dataset, likely due to clear patterns in the features. However, in real-world applications, the accuracy may drop when applied to more complex datasets with unseen variations.

## Running the Model
### Requirements
Ensure Python is installed with the following libraries:
```sh
pip install pandas numpy scikit-learn
```

### Running the Script
1. Load the dataset (`user_activity_large_dataset.csv`).
2. Preprocess and extract features.
3. Train the model.
4. Generate predictions for sample inputs.

Run the script:
```sh
python ml_model_task_completion.py
```

## Sample Prediction Output
For a sample input:
```
app_usage_time: 120 minutes
steps_taken: 8000
calories_burned: 250
sleep_duration: 7.5 hours
mood_score: 8
screen_time: 300 minutes
heart_rate: 70 bpm
hydration_level: 2.5 liters
stress_level: 5
day_of_week: 2
```
**Predicted Output:** Task Likely to be Completed

## Limitations and Future Work

Small Dataset: The model performed perfectly but may not generalize well to larger datasets.
Feature Expansion: More user activity metrics could improve prediction accuracy.
Model Optimization: Testing other classifiers like XGBoost or Neural Networks for better generalization.

## Conclusion

This project demonstrates the effectiveness of machine learning in predicting user task completion likelihood based on activity metrics. Future work involves expanding the dataset and refining model performance.
