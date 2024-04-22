#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import seaborn as sns
import tensorflow as tf

#Data Preprocessing
data = pd.read_csv("bank-full.csv", sep=';')   #45211 columns data by the file name bank-additional-full.csv

#Data Visualisation
sns.countplot(x=data['y'], data=data)
counts = data['y'].value_counts()
counts

data['age_range'] = pd.cut(x=data['age'], bins=[17,25,35,50,65,100], labels=['18-25', '26-35', '36-50','51-65', '65+'])

# Exploring the relationship between 'age_range' and 'y' (subscription)
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=data, x='age_range', hue='y', palette='magma')
plt.title('Relationship between Age Range and Subscription')
plt.xlabel('Age Range')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Subscription', labels=['No', 'Yes'])

# Calculating and annotating the count values on the plot
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='baseline')

plt.show()


# Exploring the relationship between 'Job' and 'y' (subscription)
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=data, x='job', hue='y', palette='mako')
plt.title('Relationship between Housing and Subscription')
plt.xlabel('Job')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Subscription', labels=['No', 'Yes'])

# Calculating and annotating the count values on the plot
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='baseline')

plt.show()



# Exploring the relationship between 'Marital status' and 'y' (subscription)
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=data, x='marital', hue='y', palette='flare')
plt.title('Relationship between Marital status and Subscription')
plt.xlabel('Marital')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Subscription', labels=['No', 'Yes'])

# Calculating and annotating the count values on the plot
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='baseline')

plt.show()



# Exploring the relationship between 'Education' and 'y' (subscription)
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=data, x='education', hue='y', palette='crest')
plt.title('Relationship between Housing and Subscription')
plt.xlabel('Education')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Subscription', labels=['No', 'Yes'])

# Calculate and annotate the count values on the plot
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='baseline')

plt.show()



# Exploring the relationship between 'Housing' and 'y' (subscription)
plt.figure(figsize=(10, 6))
ax = sns.countplot(data=data, x='housing', hue='y', palette='rocket')
plt.title('Relationship between Housing and Subscription')
plt.xlabel('Housing')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Subscription', labels=['No', 'Yes'])

# Calculating and annotating the count values on the plot
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='baseline')

plt.show()



#Check univariate analysis
for col in data.columns:
    fig, ax= plt.subplots(figsize=(10,5))
    ax.tick_params(axis='x', rotation=90)
    plt.title(f'{col} histogram')
    sns.histplot(data=data, x=col, ax=ax)
    plt.show()


# Checking the unique values of each column
for col in data:
    print(col + ':')
    print(data[col].unique())

for column in data.columns:
    if data[column].dtype == 'float64' or data[column].dtype == 'int64':
        data[column].fillna(data[column].median(), inplace=True)


# Checking null values
missing_data=data.isnull().sum()
missing_data

#Feature Engineering
# Encode 'month' column with values 1-12
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
data['month'] = data['month'].map(month_mapping)

# Encode 'y' column with 'yes' as 1 and 'no' as 0
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Encode the remaining categorical columns
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])
# Encode the 'age_range' column
data['age_range_encoded'] = label_encoder.fit_transform(data['age_range'])

# Drop the original 'age_range' column
data = data.drop(columns=['age_range'])

# Now, 'age_range_encoded' contains the numerical encoding of the 'age_range' values
# Split the data into train and test
X = data.drop(columns=['y'])  # Feature variables
y = data['y']  # Target variable

# X_train, X_test, y_train, and y_test contain the training and testing data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Development
# Spliting the data into features (X) and target (y)
X_train = data.drop(columns=['y'])
y_train = data['y']
X_test = data.drop(columns=['y'])
y_test = data['y']


# Convert target variable to binary labels

# Initialize and train different models
models = [
    ("Logistic Regression", LogisticRegression(max_iter=10000)),
    ("Random Forest", RandomForestClassifier()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("Neural Network", MLPClassifier(max_iter=1000)),
    ("SVM", SVC(probability=True))
]

best_model = None
best_accuracy = 0

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{name} Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

    if name != "SVM":
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc:.2f}")

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    if accuracy > best_accuracy and accuracy<1:
        best_accuracy = accuracy
        best_model = model

print(f"The best model is {best_model}")


#Feature important Analysis
# Fit the best model (RandomForestClassifier) to the training data
best_model.fit(X_train, y_train)

# Get feature importances from the model
feature_importances = best_model.feature_importances_

# Create a DataFrame to store feature names and their importances
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by feature importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the sorted feature importances
print("Feature Importance Analysis:")
print(feature_importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Analysis')
plt.show()


#Propensity Score Model Development
# Step 1: Propensity Score Model Development
propensity_model = LogisticRegression(max_iter=10000)
propensity_model.fit(X_train, y_train)

# Step 2: Predict Propensity Scores
train_propensity_scores = propensity_model.predict_proba(X_train)[:, 1]
test_propensity_scores = propensity_model.predict_proba(X_test)[:, 1]

# Step 3: Identify High-Value Leads
threshold = 0.5
train_high_value_leads = (train_propensity_scores >= threshold)
test_high_value_leads = (test_propensity_scores >= threshold)

# Print the counts of high-value leads in the training and testing sets
print("High-Value Leads in Training Data:", train_high_value_leads.sum())
print("High-Value Leads in Testing Data:", test_high_value_leads.sum())

# Strategy based on month and day of the week
month_optimized = data.groupby('month').size().idxmax()
day_optimized = data.groupby('day').size().idxmax()

# Strategy based on previous contact
recent_contact = data[data['pdays'] < 15]  # contacts made in the last 15 days

# Strategy based on economic conditions
good_economic_condition = data[data['housing'] > data['housing'].median()]
bad_economic_condition = data[data['housing'] <= data['housing'].median()]

print(f"Best month to contact: {month_optimized}")
print(f"Best day of the week to contact: {day_optimized}")

# Adjusting messaging based on economic conditions
if len(good_economic_condition) > len(bad_economic_condition):
    print("Focus on messaging highlighting the positive economic condition.")
else:
    print("Focus on messaging catering to a cautious economic condition.")

# Adjusting frequency based on previous contacts
if len(recent_contact) > len(data) * 0.5:
    print("You're contacting many leads recently. Ensure not to over-contact.")
else:
    print("Keep the communication frequent, but don't overwhelm the leads.")


# Selecting top features
important_features = [
    'duration',  'pdays',  'month',
    'poutcome', 'age'
]

X = data[important_features]
y = data['y']

# Segmenting data based on age
segments = {
    "young": data[data['age'] <= 30].reset_index(drop=True),
    "middle_aged": data[(data['age'] > 30) & (data['age'] <= 50)].reset_index(drop=True),
    "senior": data[data['age'] > 50].reset_index(drop=True)
}
segment_models = {}

for segment, segment_data in segments.items():
    X_segment = segment_data[important_features]
    y_segment = segment_data['y']

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X_segment, y_segment, test_size=0.3, random_state=42)

    # Train a model for this segment
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Store the model
    segment_models[segment] = model

    # Print the accuracy for this segment
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy for {segment} segment: {accuracy:.2f}")


important_features = [
    'duration',  'pdays',  'month',
    'poutcome',  'age', 'housing','day','campaign'
]

X = data[important_features]
y = data['y']

segments = {
    "short": data[data['duration'] <= 100],
    "medium": data[(data['duration'] > 100) & (data['duration'] <= 300)],
    "long": data[data['duration'] > 300]
}

segment_models = {}

for segment, segment_data in segments.items():
    X_segment = segment_data[important_features]
    y_segment = segment_data['y']

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X_segment, y_segment, test_size=0.3, random_state=42)

    # Train a model for this segment
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Store the model
    segment_models[segment] = model

    # Print the accuracy for this segment
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy for {segment} segment: {accuracy:.2f}")




