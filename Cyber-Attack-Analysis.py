import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('cybersecurity_attacks.csv')

# Filters out irrelevant columns
columns_to_drop = ['User Information', 'Network Segment']
data_filtered = data.drop(columns=columns_to_drop)

# Preprocessing
# Use SimpleImputer for handling missing values
imputer = SimpleImputer(strategy='constant', fill_value=0)
data_imputed = pd.DataFrame(imputer.fit_transform(data_filtered), columns=data_filtered.columns)
data_imputed['Payload Length'] = data_imputed['Payload Data'].apply(lambda x: len(str(x)))

# Converts categorical variables to numerical 
label_encoder = LabelEncoder()
categorical_columns = ['Protocol', 'Packet Type', 'Traffic Type', 'Action Taken']
for column in categorical_columns:
    data_imputed[column] = label_encoder.fit_transform(data_imputed[column])

# Exploratory Data Analysis (EDA)
print(data_imputed.describe())
plt.scatter(data_imputed['Packet Length'], data_imputed['Anomaly Scores'])
plt.title('Scatter Plot of Packet Length vs Anomaly Scores')
plt.xlabel('Packet Length')
plt.ylabel('Anomaly Scores')
plt.show()

# Clustering using k-means
features = ['Packet Length', 'Anomaly Scores']
X = data_imputed[features]
scaler = StandardScaler() #standardizes data
X_scaled = scaler.fit_transform(X)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data_imputed['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.scatter(data_imputed['Packet Length'], data_imputed['Anomaly Scores'], c=data_imputed['cluster'], cmap='viridis')
plt.title('Clustering Results')
plt.xlabel('Packet Length')
plt.ylabel('Anomaly Scores')
plt.show()

# Decision Tree model
target = 'Severity Level'
X_train, X_test, y_train, y_test = train_test_split(X, data_imputed[target], test_size=0.2, random_state=42)
param_grid_dt = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
dt_classifier = DecisionTreeClassifier()
grid_dt = GridSearchCV(dt_classifier, param_grid_dt)
grid_dt.fit(X_train, y_train)
best_params_dt = grid_dt.best_params_
dt_model = DecisionTreeClassifier(**best_params_dt, random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
