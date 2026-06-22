import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

# Load dataset
df = pd.read_csv('Student Performance.csv')
features = df.drop(columns=['Exam_Score'])  # remove target

# Identify types of features
numeric_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = features.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])

# Fit and transform to full feature matrix
X_full = preprocessor.fit_transform(features)

# Get feature names after encoding
encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(encoded_cat_names)

# Forward Feature Selection using silhouette score
selected_features = []
remaining_indices = list(range(X_full.shape[1]))

for _ in range(5):  # select 5 best features
    best_score = -1
    best_index = None
    for idx in remaining_indices:
        temp_features = selected_features + [idx]
        X_subset = X_full[:, temp_features]
        labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_subset)
        score = silhouette_score(X_subset, labels)
        if score >= best_score:
            best_score = score
            best_index = idx
    selected_features.append(best_index)
    remaining_indices.remove(best_index)

# Extract selected features
X_selected = X_full[:, selected_features]
selected_names = [all_feature_names[i] for i in selected_features]
print("Selected Features:", selected_names)

# Reduce dimensions for visualization
X_pca = PCA(n_components=2).fit_transform(X_selected)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_selected)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set1')
plt.title('KMeans Clustering (Top 5 Features via Silhouette Score)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
