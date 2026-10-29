#Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("C:/Users/niran/Documents/Predective Analysics/Ca2/city_day.csv")

#Data cleaning and Preprocessing
# 2. Display original dataset information
print("Original Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

#3.Statistical Summary and Dataset Information
print(df.describe())
print(df.info())

# 4. Remove duplicate records
print("\nDuplicate rows before cleaning:", df.duplicated().sum())
df = df.drop_duplicates()


# 5. Convert Date column into date format
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Count invalid dates
print("Invalid dates:", df["Date"].isnull().sum())



# 6. Remove invalid AQI values
# Retain rows with a valid, non-negative AQI target
df = df.dropna(subset=["AQI"])
df = df[df["AQI"] >= 0]

# 7. Check missing values before filling them
print("\nMissing Values Before Imputation:")
print(df.isnull().sum())

# 8 Fill missing numerical values using mean
# Do not fill the target AQI column with invented values
numeric_cols = ["AQI", "PM2.5", "PM10", "NO", "NO2", "NOx","NH3", "CO", "SO2", "O3", "Benzene","Toluene", "Xylene"]
for col in numeric_cols:
    if col != "AQI":
        df[col] = df[col].fillna(df[col].mean())

# 9. Fill missing categorical values with mode
categorical_cols = ["City", "AQI_Bucket"]
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# 10. Display cleaned dataset information
print("\nCleaned Dataset Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows After Cleaning:", df.duplicated().sum())

print("\nCleaned Dataset Preview:")
print(df.head())

# 11. Save cleaned dataset
df.to_csv("C:/Users/niran/Documents/Predective Analysics/Ca2/city_day_cleaned.csv", index=False)
print("Cleaned dataset saved successfully!")


# ==========================================================
# 3. OBJECTIVES AND EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

print("\nPROJECT OBJECTIVES")
print("1. Analyze the distribution of AQI values.")
print("2. Identify the most common air-quality categories.")
print("3. Compare average AQI across cities.")
print("4. Examine yearly changes in average AQI.")
print("5. Investigate relationships between pollutants and AQI.")



# ----------------------------------------------------------
# OBJECTIVE 1: AQI DISTRIBUTION
# ----------------------------------------------------------

print("\nObjective 1: AQI Distribution")
print("Minimum AQI:", df["AQI"].min())
print("Maximum AQI:", df["AQI"].max())
print("Average AQI:", round(df["AQI"].mean(), 2))

plt.figure(figsize=(9, 5))
plt.hist(df["AQI"].dropna(), bins=30, edgecolor="black", alpha=0.8)
plt.axvline(df["AQI"].mean(), linestyle="--",label="Average AQI")
plt.title("Distribution of AQI Values")
plt.xlabel("AQI")
plt.ylabel("Number of Records")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ----------------------------------------------------------
# OBJECTIVE 2: AQI CATEGORY FREQUENCY
# ----------------------------------------------------------

category_counts = df["AQI_Bucket"].value_counts()

print("\nObjective 2: AQI Category Frequency")
print(category_counts)

plt.figure(figsize=(9, 5))
ax = category_counts.plot(kind="bar", edgecolor="black")

plt.title("Distribution of AQI Categories")
plt.xlabel("AQI Category")
plt.ylabel("Number of Records")
plt.xticks(rotation=30, ha="right")
plt.bar_label(ax.containers[0], padding=3)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ----------------------------------------------------------
# OBJECTIVE 3: CITY-WISE AQI COMPARISON
# ----------------------------------------------------------

city_aqi = df.groupby("City")["AQI"].mean()
city_aqi = city_aqi.sort_values(ascending=False).head(10)

print("\nObjective 3: Top 10 Cities by Average AQI")
print(city_aqi.round(2))

plt.figure(figsize=(10, 6))
ax = city_aqi.sort_values().plot(kind="barh", edgecolor="black")

plt.title("Top 10 Cities with Highest Average AQI")
plt.xlabel("Average AQI")
plt.ylabel("City")
plt.bar_label(ax.containers[0], fmt="%.1f", padding=3)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()


# ----------------------------------------------------------
# OBJECTIVE 4: YEARLY AQI TREND
# ----------------------------------------------------------

df["Year"] = df["Date"].dt.year
yearly_aqi = df.groupby("Year")["AQI"].mean()

print("\nObjective 4: Yearly Average AQI")
print(yearly_aqi.round(2))

plt.figure(figsize=(10, 5))
plt.plot(yearly_aqi.index, yearly_aqi.values, marker="o")

plt.title("Yearly Average AQI Trend")
plt.xlabel("Year")
plt.ylabel("Average AQI")
plt.xticks(yearly_aqi.index, rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ----------------------------------------------------------
# OBJECTIVE 5: POLLUTANT CORRELATION WITH AQI
# ----------------------------------------------------------

numeric_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3","CO", "SO2", "O3", "Benzene", "Toluene","Xylene", "AQI"]
correlation = df[numeric_cols].corr()
print("\nObjective 5: Correlation with AQI")
print(correlation["AQI"].sort_values(ascending=False).round(3))
plt.figure(figsize=(12, 9))
sns.heatmap(correlation,annot=True,cmap="coolwarm",center=0,fmt=".2f",linewidths=0.5)
plt.title("Correlation Matrix of AQI and Pollutants")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
print("\nData Cleaning and EDA Completed Successfully!")



# FEATURE ENGINEERING
# Extract year, month, and day of the week
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek

# Display the new features
print("\nDataset After Feature Engineering:")
print(df[["Date", "Year", "Month", "DayOfWeek"]].head())
print("\nFeature Engineering Completed Successfully!")

#------------------------------------------------------------------
# Machine Learning Model Implementation and Evaluation
#------------------------------------------------------------------

# Select input features
features = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3","CO", "SO2", "O3", "Benzene", "Toluene", "Xylene","Year", "Month", "DayOfWeek", "City"]


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#----------------------------------
# MODEL 1: KNN CLASSIFIER
#----------------------------------


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Prepare input and target
X = pd.get_dummies(df[features], columns=["City"], dtype=int)
y = df["AQI_Bucket"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Predict AQI category
y_pred = model.predict(X_test)

# Evaluate the model
print("KNN CLASSIFIER RESULTS")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
knn_accuracy = accuracy_score(y_test, y_pred)

#------------------------------------
# MODEL 2: DECISION TREE CLASSIFIER
#------------------------------------

from sklearn.tree import DecisionTreeClassifier

# Prepare input and target
X = pd.get_dummies(df[features], columns=["City"], dtype=int)
y = df["AQI_Bucket"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict AQI category
y_pred = model.predict(X_test)

# Evaluate the model
print("DECISION TREE CLASSIFIER RESULTS")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
dt_accuracy = accuracy_score(y_test, y_pred)


#------------------------------------
# MODEL 3: RANDOM FOREST CLASSIFIER
#-----------------------------------
from sklearn.ensemble import RandomForestClassifier


# Prepare input and target
X = pd.get_dummies(df[features], columns=["City"], dtype=int)
y = df["AQI_Bucket"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict AQI category
y_pred = model.predict(X_test)

# Evaluate the model
print("RANDOM FOREST CLASSIFIER RESULTS")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
rf_accuracy = accuracy_score(y_test, y_pred)

#--------------------------------------
# MODEL 4: Naive Bayes Classification
#--------------------------------------

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Remove Date and AQI columns
df_nb = df.drop(["Date", "AQI"], axis=1).copy()

# Encode categorical columns
label = LabelEncoder()
df_nb["City"] = label.fit_transform(df_nb["City"])

# Remove rows with missing values
df_nb = df_nb.dropna()

# Select input and target
x = df_nb.drop("AQI_Bucket", axis=1)
y = df_nb["AQI_Bucket"]

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and train Naive Bayes model
nb = GaussianNB()
nb.fit(x_train, y_train)

# Predict air quality category
y_pred = nb.predict(x_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
nb_accuracy = accuracy_score(y_test, y_pred)



# ----------------------------------------------------------
# CLASSIFIER MODEL ACCURACY COMPARISON
# Visualization: Bar Chart
# ----------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd

accuracy_df = pd.DataFrame({
    "Model": ["KNN Classifier","Decision Tree","Random Forest","Naive Bayes"],
    "Accuracy": [knn_accuracy * 100,dt_accuracy * 100,rf_accuracy * 100,nb_accuracy * 100]})

colors = ["#3498DB", "#E67E22", "#2ECC71", "#9B59B6"]
plt.figure(figsize=(10, 6))
bars = plt.bar(accuracy_df["Model"],accuracy_df["Accuracy"],color=colors,edgecolor="black",width=0.6)
plt.title("Classifier Model Accuracy Comparison",fontsize=16, fontweight="bold")
plt.xlabel("Machine Learning Models", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.ylim(0, 100)
plt.bar_label(bars, fmt="%.2f%%", padding=5, fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

#--------------------------------------
# BEST-FIT MODEL SELECTION USING ACCURACY
#--------------------------------------

# Compare the accuracy of all classifiers
results_df = pd.DataFrame({
    "Model": ["KNN Classifier", "Decision Tree Classifier",
              "Random Forest Classifier", "Naive Bayes Classifier"],
    "Accuracy": [knn_accuracy,dt_accuracy,rf_accuracy,nb_accuracy]})

# Display model comparison
print("\nCLASSIFIER ACCURACY COMPARISON")
print(results_df)

# Select the model with the highest accuracy
best_model = results_df.loc[results_df["Accuracy"].idxmax()]

# Display best-fit model
print("\nBEST-FIT MODEL")
print("Model:", best_model["Model"])
print("Accuracy:", round(best_model["Accuracy"] * 100, 2), "%")


