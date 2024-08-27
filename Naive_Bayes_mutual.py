import pandas as pd
from sklearn.naive_bayes import GaussianNB
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

# Split the data into features (X) and target (y)
X = data[selected_features_mutual_info]
y = target

# Train a Naive Bayes classifier
nb_mutual_info = GaussianNB()
nb_mutual_info.fit(X, y)

# Evaluate the classifier using cross-validation
scores_mutual_info = cross_val_score(nb_mutual_info, X, y, cv=5)

# Print the accuracy and F1 score
print("Accuracy (mutual_info_classif):", scores_mutual_info.mean())
print("F1 score (mutual_info_classif):", f1_score(y, nb_mutual_info.predict(X), average="weighted"))

# Calculate the confusion matrix
y_pred = nb_mutual_info.predict(X)
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
for feature in selected_features_mutual_info:
    # If the feature is continuous
    if data[feature].dtype != 'object':
        class_1_data = data[target == 1][feature]
        class_2_data = data[target == 2][feature]
        t_stat, p = ttest_ind(class_1_data, class_2_data)
        print(f"Feature (mutual_info_classif): {feature}, p-value: {p}")
