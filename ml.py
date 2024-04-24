import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
stress = pd.read_csv('StressLevelDataset.csv')
X = stress.drop('stress_level', axis=1)
y = stress['stress_level']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model with specified parameters
random_forest_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees in the forest
    max_depth=2,           # Maximum depth of the trees
    min_samples_split=2,   # Minimum number of samples required to split an internal node
    min_samples_leaf=1,    # Minimum number of samples required to be at a leaf node
    random_state=42
)
# Measure training time
start_time = time.time()
random_forest_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Make predictions on the test set
start_time = time.time()
y_pred = random_forest_model.predict(X_test)
prediction_time = time.time() - start_time

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Print the results
print("Training Time:", training_time, "seconds")
print("Prediction Time:", prediction_time, "seconds")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_result)

from sklearn.metrics import confusion_matrix
# Obtain confusion matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix (Random Forest):\n", conf_matrix_rf)
new_predictions = random_forest_model.predict(X_test)

# Display the predictions
print("Predictions for the new data:")
print(new_predictions)
# DECISION TREE

from sklearn.tree import DecisionTreeClassifier, plot_tree
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model with specified parameters
decision_tree_model = DecisionTreeClassifier(
    max_depth=3,          # Maximum depth of the tree
    min_samples_split=2,  # Minimum number of samples required to split an internal node
    min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
    random_state=42
)

# Measure training time
start_time = time.time()
decision_tree_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Make predictions on the test set
start_time = time.time()
y_pred = decision_tree_model.predict(X_test)
prediction_time = time.time() - start_time

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree_model, filled=True, feature_names=X.columns, class_names=[str(i) for i in decision_tree_model.classes_])
plt.title("Decision Tree Visualization")
plt.show()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Print the results
print("Training Time:", training_time, "seconds")
print("Prediction Time:", prediction_time, "seconds")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_result)
#ADABOOST
from sklearn.ensemble import AdaBoostClassifier
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the AdaBoost model with specified parameters
adaboost_model = AdaBoostClassifier(
    n_estimators=50,        # Number of weak learners (usually decision trees)
    learning_rate=1.0,      # Contribution of each weak learner to the final prediction
    random_state=42
)

# Measure training time
start_time = time.time()
adaboost_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Make predictions on the test set
start_time = time.time()
y_pred = adaboost_model.predict(X_test)
prediction_time = time.time() - start_time

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Print the results
print("Training Time:", training_time, "seconds")
print("Prediction Time:", prediction_time, "seconds")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_result)
#XGBOOST
from xgboost import XGBClassifier
y = stress['stress_level']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model with specified parameters
xgboost_model = XGBClassifier(
    n_estimators=100,       # Number of boosting rounds
    learning_rate=0.1,      # Step size shrinkage to prevent overfitting
    max_depth=3,            # Maximum depth of a tree
    random_state=42
)

# Measure training time
start_time = time.time()
xgboost_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Make predictions on the test set
start_time = time.time()
y_pred = xgboost_model.predict(X_test)
prediction_time = time.time() - start_time

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Print the results
print("Training Time:", training_time, "seconds")
print("Prediction Time:", prediction_time, "seconds")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_result)

