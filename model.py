##from google.colab import files
##ss = files.upload()
##df = pd.read_csv('RTA Dataset.csv')
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("RTA Dataset.csv")  # Change file path accordingly

# Display basic info
print(df.info())
print(df.head())

# Handle missing values
df.ffill(inplace=True)  # Forward fill as an example

# Check column names
print("Dataset Columns:", df.columns)

# Strip any extra spaces from column names
df.rename(columns=lambda x: x.strip(), inplace=True)

# Ensure correct column reference (update if necessary)
target_column = 'Accident_severity'  # Adjusted to match dataset
if target_column not in df.columns:
    raise KeyError(f"Column '{target_column}' not found. Available columns: {df.columns}")

# Check unique values and datatype of the target column
print(f"Unique values in {target_column}:", df[target_column].unique())
print(f"Data type of {target_column}:", df[target_column].dtype)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode Accident Severity
y = df[target_column]
severity_encoder = LabelEncoder()
df[target_column] = severity_encoder.fit_transform(df[target_column])
print("Accident Severity Encoding:", dict(zip(severity_encoder.classes_, severity_encoder.transform(severity_encoder.classes_))))

# Define features and target
X = df.drop(columns=[target_column])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC()
}

# Train and evaluate models
accuracy_scores = {}
plt.figure(figsize=(15, 5))

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores[name] = accuracy

    # Confusion Matrix
    plt.subplot(2, 2, i + 1)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# Plot accuracy comparison
plt.figure(figsize=(8, 5))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

# Print classification reports
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred))

# Boxplot to check for outliers with log scaling for better visualization
plt.figure(figsize=(14, 7))
ax = sns.boxplot(data=df)
ax.set_yscale("log")  # Apply logarithmic scale
plt.xticks(rotation=90)
plt.title("Boxplot of Features (Log Scale)")
plt.show()

# Correlation Matrix with adjusted font size for readability
plt.figure(figsize=(14, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f')
plt.xticks(fontsize=10, rotation=90)
plt.yticks(fontsize=10)
plt.title("Correlation Matrix")
plt.show()

# Plot Severity Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=df[target_column], palette='coolwarm')
plt.title("Accident Severity Distribution")
plt.xlabel("Severity Level")
plt.ylabel("Count")
plt.show()

# Correlation with Accident Severity
plt.figure(figsize=(12, 6))
df.corr()[target_column].drop(target_column).sort_values().plot(kind='barh', cmap='coolwarm')
plt.title("Feature Correlation with Accident Severity")
plt.show()
