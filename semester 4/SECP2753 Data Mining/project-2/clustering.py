import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold


# Load dataset
dataset = pd.read_csv('Student Performance.csv')

dataset_copy = dataset.copy()

# Categorizing the features
categorical_features = ['Extracurricular_Activities', 'Internet_Access', 'School_Type', 'Learning_Disabilities', 'Gender', 'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Peer_Influence', 'Parental_Education_Level', 'Distance_from_Home']

continuous_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity',] # exclude the 'Exam_Score' feature since this is not supervised learning

continuous_features_include_exam_score = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score'] # exclude the 'Exam_Score' feature since this is not supervised learning

# Impute missing values
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

dataset_copy[continuous_features] = num_imputer.fit_transform(dataset_copy[continuous_features])
dataset_copy[categorical_features] = cat_imputer.fit_transform(dataset_copy[categorical_features])

# # Eliminate outliers

for col in continuous_features:

    q1 = dataset_copy[col].quantile(0.25)
    q3 = dataset_copy[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    dataset_copy = dataset_copy[(dataset_copy[col] >= lower_bound) & (dataset_copy[col] <= upper_bound)]
    
dataset_copy.reset_index(drop= True, inplace = True)
# Encoding Categorical Features

ohe = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')
encoded_categorical = ohe.fit_transform(dataset_copy[categorical_features])
encoded_categorical = pd.DataFrame(encoded_categorical, columns = ohe.get_feature_names_out(categorical_features))

# Scaling continuous features

scaler = StandardScaler()
scaled_continous = scaler.fit_transform(dataset_copy[continuous_features])
scaled_continous = pd.DataFrame(scaled_continous, columns = continuous_features)

# Combining the encoded categorical features with Continuous Features
combined_features = pd.concat([scaled_continous, encoded_categorical], axis = 1)
combine_unscaled = pd.concat([dataset_copy[continuous_features_include_exam_score], encoded_categorical], axis = 1)

print(combined_features)
print(combined_features.shape)

# Applying feature selection to select most important features via Variance Threshold

selector = VarianceThreshold(threshold = 0.9)
selected_features_indices = selector.fit_transform(combined_features)
selected_features = combined_features.columns[selector.get_support()].to_list()

print(selected_features) # 6 important features selected


def KMeansClustering(combined_features, selected_features, original_data):

    # Using elbow method to find the optimal number of clusters
    
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(combined_features[selected_features])
        inertia.append(kmeans.inertia_)

    # Plot the Elbow Method graph
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow Method for Optimal K (K-Means)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

    # Applying PCA for dimensionality reduction
    pca = PCA(n_components = 2)
    features_pca = pca.fit_transform(combined_features[selected_features])

    # Applying KMeans Clustering

    kmeans = KMeans(n_clusters = 3, random_state = 42, n_init = 10)
    kmeans_labels = kmeans.fit_predict(features_pca)

    combined_features['KMeans_Cluster'] = kmeans_labels

    # Visualize the KMeans_Cluster 2d scatter plot
    features_pca_df = pd.DataFrame(features_pca, columns = ['pc_1', 'pc_2'])
    features_pca_df['KMeans_Cluster'] = combined_features['KMeans_Cluster']

    plt.figure(figsize = (10, 6))
    sns.scatterplot(data = features_pca_df, x = 'pc_1', y = 'pc_2', hue = 'KMeans_Cluster', palette = 'Set2')
    plt.show()

    # KMeans_Cluster Profiling 
    original_data['KMeans_Cluster'] = combined_features['KMeans_Cluster']
    print(original_data.groupby('KMeans_Cluster')[combined_features.columns].agg(['mean', 'min', 'max']))

    profile = original_data.groupby('KMeans_Cluster')['Exam_Score'].agg(['mean', 'min', 'max'])
    print(profile)

    return features_pca

def agglomerativeClustering(combined_features, selected_features, original_data):

    # Using elbow method to find the optimal number of clusters

    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(combined_features[selected_features])
        inertia.append(kmeans.inertia_)

    # Plot the Elbow Method graph
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Elbow Method for Optimal K (K-Means)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

    # Visualize the Cluster 2d scatter plot
    pca = PCA(n_components = 2)
    features_pca = pca.fit_transform(combined_features[selected_features])

    # Applying Agglomerative Clustering

    agglomerative = AgglomerativeClustering(n_clusters = 3, linkage = 'ward')
    agglomerative_labels = agglomerative.fit_predict(features_pca)

    combined_features['Agglomerative_Cluster'] = agglomerative_labels

    # Applying PCA for dimensionality reduction
    features_pca_df = pd.DataFrame(features_pca, columns = ['pc_1', 'pc_2'])
    features_pca_df['Agglomerative_Cluster'] = combined_features['Agglomerative_Cluster']

    plt.figure(figsize = (10, 6))
    sns.scatterplot(data = features_pca_df, x = 'pc_1', y = 'pc_2', hue = 'Agglomerative_Cluster', palette = 'Set2')
    plt.show()

    # Agglomerative_Cluster Profiling 
    original_data['Agglomerative_Cluster'] = combined_features['Agglomerative_Cluster']
    print(original_data.groupby('KMeans_Cluster')[combined_features.columns].agg(['mean', 'min', 'max']))

    profile = original_data.groupby('Agglomerative_Cluster')['Exam_Score'].agg(['mean', 'min', 'max'])
    print(profile)


    return features_pca

def evaluate_cluster(combined_features, features_pca, cluster_name):
    # Evaluating the cluster using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index

    silhouette_scores = silhouette_score(features_pca, combined_features[cluster_name])
    davies_bouldin_scores = davies_bouldin_score(features_pca, combined_features[cluster_name])
    calinski_harabasz_scores = calinski_harabasz_score(features_pca, combined_features[cluster_name])

    print("Silhouette Score:", silhouette_scores)
    print("Davies-Bouldin Index:", davies_bouldin_scores)
    print("Calinski-Harabasz Index:", calinski_harabasz_scores)


kmeans_features_pca = KMeansClustering(combined_features, selected_features, combine_unscaled)
evaluate_cluster(combined_features, kmeans_features_pca, 'KMeans_Cluster')

agnes_features_pca = agglomerativeClustering(combined_features, selected_features, combine_unscaled)
evaluate_cluster(combined_features, agnes_features_pca, 'Agglomerative_Cluster')