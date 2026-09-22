import pandas as pd

# Load the dataset
df = pd.read_csv("Student_Performance.csv")

# Basic exploration
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Preprocessing

# Convert categorical column to numeric
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Check for missing values again (should be 0)
print(df.isnull().sum())

# Separate features (X) and target (y)
X = df.drop('Performance Index', axis=1)
y = df['Performance Index']

print(X.head())
print(y.head())

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Train the model
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Check the learned coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Feature names:", X.columns.tolist())

#  Predict and evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

#  EDA Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# 2. Actual vs Predicted scatter plot
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")
plt.title("Actual vs Predicted Performance")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

# 3. Hours Studied vs Performance Index
plt.figure(figsize=(7, 6))
sns.scatterplot(x='Hours Studied', y='Performance Index', data=df, alpha=0.4)
plt.title("Hours Studied vs Performance Index")
plt.tight_layout()
plt.savefig("hours_vs_performance.png")
plt.show()

# Predict performance for a new student
import pandas as pd

new_student = pd.DataFrame({
    'Hours Studied': [6],
    'Previous Scores': [85],
    'Extracurricular Activities': [1],   # 1 = Yes, 0 = No
    'Sleep Hours': [7],
    'Sample Question Papers Practiced': [5]
})

predicted_performance = model.predict(new_student)
print("Predicted Performance Index:", predicted_performance[0])