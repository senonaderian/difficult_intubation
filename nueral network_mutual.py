import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.stats import ttest_ind

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

# Create a SelectKBest object using mutual_info_classif score function and fit to data
selector_mutual_info = SelectKBest(mutual_info_classif, k=5)
X_mutual_info = selector_mutual_info.fit_transform(data_non_negative, target)

# Print selected features using mutual_info_classif score function
print("Selected Features (using mutual_info_classif score function):")
selected_features_mutual_info = data.columns[selector_mutual_info.get_support()].tolist()
print(selected_features_mutual_info)

# Convert selected features to DataFrame
selected_features_df = pd.DataFrame({'selected_features': selected_features_mutual_info})

# Save the selected features to a file
selected_features_df.to_csv('selected_features_mlp_mutual_info.csv', index=False)

# Load the dataset
selected_features = pd.read_csv("selected_features_mlp_mutual_info.csv")

# Split the data into features (X) and target (y)
X = data[selected_features['selected_features']]
y = target

# Define the parameter grid for hyperparameter tuning (not applicable for MLP)

# Train an MLP classifier
mlp = MLPClassifier(max_iter=2000)
mlp.fit(X, y)

# Evaluate the classifier using cross-validation
scores = cross_val_score(mlp, X, y, cv=5)

# Print the accuracy and F1 score
print("Accuracy:", scores.mean())
print("F1 score:", f1_score(y, mlp.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = mlp.predict(X)
cm = confusion_matrix(y, y_pred, labels=[1, 2])  # Specify all possible labels
if cm.size >= 4:
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
else:
    print("Insufficient values in the confusion matrix.")

# No plotting function for MLP

# Encode the target variable to 0 and 1
target_encoded = (target == 1).astype(int)

# Calculate p-values and association of each selected feature with each class
# Loop through each selected feature
for feature in selected_features_df['selected_features']:
    # If the feature is continuous
    if data[feature].dtype != 'object':
        class_1_data = data[target == 1][feature]
        class_2_data = data[target == 2][feature]
        t_stat, p = ttest_ind(class_1_data, class_2_data)
        print(f"Feature: {feature}, p-value: {p}")
