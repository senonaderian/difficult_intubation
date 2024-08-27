import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from scipy.stats import chi2_contingency, ttest_ind

# Set a specific random seed
np.random.seed(42)

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

# Create a SelectKBest object using chi2 score function and fit to data
selector_chi2 = SelectKBest(chi2, k=5)
X_chi2 = selector_chi2.fit_transform(data_non_negative, target)

# Print selected features using chi2 score function
print("Selected Features (using chi-square score function):")
selected_features_chi2 = data.columns[selector_chi2.get_support()].tolist()
print(selected_features_chi2)

# Convert selected features to DataFrame
selected_features_df = pd.DataFrame({'selected_features': selected_features_chi2})

# Save the selected features to a file
selected_features_df.to_csv('selected_features_nb.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features_nb.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Calculate class weights
class_weights = {1: len(y) / (2 * (y == 1).sum()), 2: len(y) / (2 * (y == 2).sum())}

# Define the parameter grid for hyperparameter tuning (not applicable for Naive Bayes)

# Train a Naive Bayes classifier
nb = GaussianNB()
nb.fit(X, y)

# Evaluate the classifier using cross-validation
scores = cross_val_score(nb, X, y, cv=5)

# Print the accuracy and F1 score
print("Accuracy:", scores.mean())
print("F1 score:", f1_score(y, nb.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = nb.predict(X)
cm = confusion_matrix(y, y_pred, labels=[1, 2])  # Specify all possible labels
if cm.size >= 4:
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
else:
    print("Insufficient values in the confusion matrix.")

# No plotting function for Naive Bayes

# Encode the target variable to 0 and 1
target_encoded = (target == 1).astype(int)

# Calculate p-values and association of each selected feature with each class
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
