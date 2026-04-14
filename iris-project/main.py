import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("iris.csv")

# Prepare features and target
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

# Convert species to numbers (0,1,2)
y_encoded = y.astype("category").cat.codes

# Rename columns to your format
X.columns = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

# Combine into one table
display_df = X.copy()
display_df["target"] = y_encoded

# Show first 5 rows
print("first 5 rows:\n")
print(display_df.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# Model
model = KNeighborsClassifier(n_neighbors=3)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Predicted class (convert back to name)
predicted_label = pd.Series(y).astype("category").cat.categories[y_pred[0]]

print("Predicted class:", predicted_label)