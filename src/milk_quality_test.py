import pandas as pd

# Load dataset
df = pd.read_csv('data/data.csv')

# Show first 15 rows
print("=== First 15 Rows ===")
print(df.head(15))

# Dataset information
print("\n=== Dataset Info ===")
print(df.info())

# Summary statistics
print("\n=== Summary Statistics ===")
print(df.describe())


# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing numeric values with mean
num_cols = ['pH', 'Temperature']

for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nCleaning Completed!")
print("Total Rows:", len(df))

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())




#Chart 1 — pH vs Grade (Scatter Plot)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.scatter(df['pH'], df['Grade'],
            color='blue',
            s=80,
            alpha=0.7)

plt.xlabel('pH Value')
plt.ylabel('Milk Grade')
plt.title('pH vs Milk Grade')

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig('output/chart1_ph_grade.png')

plt.show()



#Chart 2 — Average Temperature by Grade (Bar Chart)
grade_temp = df.groupby('Grade')['Temperature'].mean()

plt.figure(figsize=(6,4))

grade_temp.plot(kind='bar',
                color=['red', 'orange', 'green'],
                edgecolor='black')

plt.xlabel('Milk Grade')
plt.ylabel('Average Temperature')

plt.title('Average Temperature by Grade')

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig('output/chart2_temperature.png')

plt.show()




#Chart 3 — Correlation Heatmap

import seaborn as sns

plt.figure(figsize=(8,5))

sns.heatmap(df.corr(),
            annot=True,
            cmap='Greens',
            fmt='.2f')

plt.title('Correlation Heatmap')

plt.tight_layout()

plt.savefig('output/chart3_heatmap.png')

plt.show()



#Build a Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Features (inputs)
X = df[['pH', 'Temperature', 'Taste',
        'Odor', 'Fat', 'Turbidity']]

# Target (output)
y = df['Grade']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")



#Predicting on a new sample
new_sample = pd.DataFrame({
    'pH': [float(input("Enter pH value (0-14): "))],
    'Temperature': [float(input("Enter temperature value (°C): "))],
    'Taste': [float(input("Enter taste value (1 = Good, 0 = Bad) :"))],
    'Odor': [float(input("Enter odor value (1 = Bad smell, 0 = Good): "))],
    'Fat': [float(input("Enter fat value(1 = High fat, 0 = Low fat): "))],
    'Turbidity': [float(input("Enter turbidity value(1 = Cloudy, 0 = Clear): "))]
})

prediction = model.predict(new_sample)

print("\nPredicted Milk Grade:", prediction[0])

# Convert number to text
if prediction[0] == 0:
    print("Your milk is Low Quality")
elif prediction[0] == 1:
    print("Your milk is Medium Quality")
elif prediction[0] == 2:
    print("Your milk is High Quality")