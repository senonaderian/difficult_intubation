import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib

np.random.seed(42)  # Set a specific random seed
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Read the CSV file into a DataFrame
df = pd.read_csv('3.csv')

# Extract the 'antubation.result' column and store it in a separate variable
target = df['antubation.result']

# Select all columns except for the 'antubation.result' column
data = df.drop('antubation.result', axis=1)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Apply a transformation to make the data non-negative
data_non_negative = data_imputed - np.min(data_imputed) + 1e-6

# Create a SelectKBest object using mutual info function and fit to data
selector_mutual_info = SelectKBest(mutual_info_classif, k=5)
X_mutual_info = selector_mutual_info.fit_transform(data_non_negative, target)

# Print selected features using mutual info function
print("Selected Features (using Mutual Information):")
selected_features_mutual_info = data.columns[selector_mutual_info.get_support()].tolist()
print(selected_features_mutual_info)

# Convert selected features to DataFrame
selected_features_df = pd.DataFrame({'selected_features': selected_features_mutual_info})

# Save the selected features to a file
selected_features_df.to_csv('selected_features1.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features1.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Calculate class weights
class_weights = {1: len(y) / (2 * (y == 1).sum()), 2: len(y) / (2 * (y == 2).sum())}

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Train a decision tree classifier with hyperparameter tuning
clf = DecisionTreeClassifier(class_weight=class_weights, random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X, y)
clf_optimized = grid_search.best_estimator_

# Save the trained Decision Tree model
joblib.dump(clf_optimized, 'decision_tree_model.pkl')

# Evaluate the classifier using cross-validation
scores = cross_val_score(clf_optimized, X, y, cv=5)

# Print the accuracy and F1 score
print("Accuracy (Mutual Information):", scores.mean())
print("F1 score (Mutual Information):", f1_score(y, clf_optimized.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = clf_optimized.predict(X)
cm = confusion_matrix(y, y_pred, labels=[1, 2])  # Specify all possible labels
if cm.size >= 4:
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Sensitivity (Mutual Information):", sensitivity)
    print("Specificity (Mutual Information):", specificity)
else:
    print("Insufficient values in the confusion matrix.")

# Plot the decision tree
plt.figure(figsize=(30, 20))
plot_tree(clf_optimized, filled=True, feature_names=X.columns.tolist(), class_names=["1", "2"])
plt.savefig("decision_tree.png", dpi=200)
plt.show()

# Plot feature importances out of 100
importances = clf_optimized.feature_importances_
importances_scaled = importances * 100  # Scale importances to be out of 100
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances_scaled})
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
plt.figure(figsize=(20, 20))
plt.bar(feature_importances['feature'], feature_importances['importance'])
plt.xlabel('Features')
plt.ylabel('Importance (out of 100)')
plt.title('Feature Importances (out of 100)')

# Specify the y-axis ticks and labels in intervals of 5
y_ticks = range(0, 101, 2)
plt.savefig('feature_importancesdt.jpg', format='jpeg')
plt.yticks(y_ticks)

plt.xticks(rotation='vertical')
plt.show()

# Encode the target variable to 0 and 1
target_encoded = (target == 1).astype(int)

# Calculate p-values and association of each selected feature with each class
from scipy.stats import chi2_contingency, ttest_ind

# Loop through each selected feature
for feature in selected_features_df['selected_features']:
    # If the feature is categorical
    if data[feature].dtype == 'object':
        contingency_table = pd.crosstab(data[feature], target)
        chi2, p, _, _ = chi2_contingency(contingency_table)
        print(f"Feature: {feature}, p-value: {p}")
    # If the feature is continuous
    else:
        class_1_data = data[target == 1][feature]
        class_2_data = data[target == 2][feature]
        t_stat, p = ttest_ind(class_1_data, class_2_data)
        print(f"Feature: {feature}, p-value: {p}")
