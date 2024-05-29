# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Importing the dataset
data = pd.read_csv('/kaggle/input/drugs-a-b-c-x-y-for-decision-trees/drug200.csv')

# Splitting the dataset into features and target variables
X = data.drop(columns=['Drug'])
y = data['Drug']
X = pd.get_dummies(X)  # Encode categorical variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the decision tree model using Gini index
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

# Training the decision tree model using entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

# Plotting the decision tree using Gini index
plt.figure(figsize=(15, 10))
plot_tree(clf_gini, filled=True, feature_names=X_train.columns, class_names=data['Drug'].unique(), rounded=True)
plt.show()

# Plotting the decision tree using entropy
plt.figure(figsize=(15, 10))
plot_tree(clf_entropy, filled=True, feature_names=X_train.columns, class_names=data['Drug'].unique(), rounded=True)
plt.show()
