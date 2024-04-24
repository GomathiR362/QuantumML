import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

stress = pd.read_csv('/content/StressLevelDataset.csv')
X = stress.drop('stress_level', axis=1)
y = stress['stress_level']
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
plt.figure(figsize=(5,5))
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

# Obtain confusion matrix for Decision Tree
conf_matrix_dt = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix (Decision Tree):\n", conf_matrix_dt)