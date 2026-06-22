import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score
from kneed import KneeLocator

def preprocessing_data(dataset):

    ## Data Cleaning

    # Eliminate missing values
    dataset.isnull().sum()
    dataset = dataset.dropna()

    # Eliminate duplicate values
    dataset.duplicated().sum() # no duplicated values

    # Eliminate inconsistent values
    dataset.drop(dataset[dataset['Exam_Score'] > 100].index , inplace = True)
    dataset.drop(dataset[dataset['Exam_Score'] < 0].index , inplace = True)
    dataset.drop(dataset[dataset['Attendance'] > 100].index , inplace = True)
    dataset.drop(dataset[dataset['Attendance'] < 0].index , inplace = True)
    dataset.drop(dataset[dataset['Previous_Scores'] > 100].index , inplace = True)
    dataset.drop(dataset[dataset['Previous_Scores'] < 0].index , inplace = True)

    # reset index
    dataset.reset_index(drop=True, inplace=True)

    ## Data Transformation

    categorical_features = ['Extracurricular_Activities', 'Internet_Access', 'School_Type', 'Learning_Disabilities', 'Gender', 'Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Peer_Influence', 'Parental_Education_Level', 'Distance_from_Home']

    continuous_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score',]

    # encoding categorical features

    encoded_categorical_features = pd.get_dummies(dataset[categorical_features], drop_first=False)

    # Create processed_dataset by dropping original categorical columns and concatenating one-hot encoded ones

    processed_dataset = dataset.drop(columns=categorical_features).copy()
    processed_dataset = pd.concat([processed_dataset, encoded_categorical_features], axis=1)

    processed_dataset.to_csv('processed_data.csv', index = False)

    ## IMPROVED Outlier Detection and Removal
    # Use a more conservative approach - only remove extreme outliers
    initial_size = len(processed_dataset)
    
    # Only remove outliers from key performance features to avoid over-removal
    key_features_for_outlier_detection = ['Exam_Score', 'Previous_Scores', 'Hours_Studied', 'Attendance']
    
    outlier_indices = set()
    
    for col in key_features_for_outlier_detection:
        if col in processed_dataset.columns:
            Q1 = processed_dataset[col].quantile(0.25)
            Q3 = processed_dataset[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Use more conservative bounds (2.5 instead of 1.5)
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR
            
            # Find outlier indices for this column
            col_outliers = processed_dataset[
                (processed_dataset[col] < lower_bound) | 
                (processed_dataset[col] > upper_bound)
            ].index
            
            outlier_indices.update(col_outliers)
    
    # Remove outliers while maintaining index alignment
    clean_indices = [i for i in processed_dataset.index if i not in outlier_indices]
    processed_dataset = processed_dataset.loc[clean_indices].reset_index(drop=True)
    
    # print(f"Removed {len(outlier_indices)} outlier records ({(len(outlier_indices)/initial_size)*100:.1f}%)")

    ## Sampling
    features_column = [f for f in encoded_categorical_features.columns] + [f for f in continuous_features]

    # Ensure that all columns in features_column exist in processed_dataset
    missing_cols = [col for col in features_column if col not in processed_dataset.columns]
    if missing_cols:
        raise KeyError(f"The following columns are missing from processed_dataset: {missing_cols}")
     
    ## Normalize the numerical features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(processed_dataset[features_column].values)
    print(features_scaled.shape)

    return features_scaled, processed_dataset, continuous_features


def K_Means_Clustering(features_scaled, preprocessed_data):
    # --- K-Means Clustering ---

    # Determine the optimal number of clusters using the Elbow Method

    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_scaled)
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

    # Based on the elbow method, Perform clustering with optimal K = 3
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter = 500, init = 'k-means++')
    kmeans_labels = kmeans.fit_predict(features_scaled)
    preprocessed_data['KMeans_Cluster'] = kmeans_labels

    return preprocessed_data

def K_Means_visualization(features_scaled, preprocessed_data):

    # Apply PCA for dimensionality reduction to 3 components
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features_scaled)

    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2', 'principal_component_3'])
    pca_df['KMeans_Cluster'] = preprocessed_data['KMeans_Cluster']

    # Create the 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Get cluster labels and a color map
    unique_clusters = pca_df['KMeans_Cluster'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))

    # Plot each cluster separately to get distinct colors and a legend
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = pca_df[pca_df['KMeans_Cluster'] == cluster_id]
        ax.scatter(
            cluster_data['principal_component_1'],
            cluster_data['principal_component_2'],
            cluster_data['principal_component_3'],
            color=colors[i],
            s=100,
            alpha=0.7,
            label=f'Cluster {cluster_id}'
        )

    ax.set_title('K-Means Clusters Visualized with PCA (3 Components)')
    ax.set_xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
    ax.set_ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
    ax.set_zlabel(f'Principal Component 3 (Explained Variance: {pca.explained_variance_ratio_[2]:.2f})')
    ax.legend()
    plt.tight_layout()
    plt.show()

def Agnes_Clustering(features_scaled, preprocessed_data):

    # --- AGNES (Agglomerative Hierarchical) Clustering ---
    optimal_k = 3
    agnes = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    agnes_labels = agnes.fit_predict(features_scaled)
    preprocessed_data['AGNES_Cluster'] = agnes_labels

    return preprocessed_data

def Agnes_visualization(features_scaled, preprocessed_data):

    # Apply PCA for dimensionality reduction to 3 components
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features_scaled)

    # Create a DataFrame for plotting
    pca_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2', 'principal_component_3'])
    pca_df['AGNES_Cluster'] = preprocessed_data['AGNES_Cluster']

    # Create the 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Get cluster labels and a color map
    unique_clusters = pca_df['AGNES_Cluster'].unique()
    unique_clusters.sort() # Ensure consistent order for coloring

    colors = plt.cm.get_cmap('viridis', len(unique_clusters))

    # Plot each cluster separately
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = pca_df[pca_df['AGNES_Cluster'] == cluster_id]
        ax.scatter(
            cluster_data['principal_component_1'],
            cluster_data['principal_component_2'],
            cluster_data['principal_component_3'],
            color=colors(i),
            s=100,
            alpha=0.7,
            label=f'Cluster {cluster_id}'
        )

    ax.set_title('AGNES Clusters Visualized with PCA (3 Components)')
    ax.set_xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
    ax.set_ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
    ax.set_zlabel(f'Principal Component 3 (Explained Variance: {pca.explained_variance_ratio_[2]:.2f})')
    ax.legend()
    plt.tight_layout()
    plt.show()

def evaluate_clustering(features_scaled, preprocessed_data, column_name):
    # Evaluate clustering performance using silhouette score and Davies-Bouldin index

    silhouette_scores = silhouette_score(features_scaled, preprocessed_data[column_name])
    davies_bouldin_scores = davies_bouldin_score(features_scaled, preprocessed_data[column_name])

    print("Silhouette Score:", silhouette_scores)
    print("Davies-Bouldin Index:", davies_bouldin_scores)

def cluster_profiling(preprocessed_data, column_name):

    continuous_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']

    cluster_profiles = preprocessed_data.groupby(column_name)[continuous_features].mean().round(2)

    print("\n--- Cluster Profiles (Showing Average Values) ---")
    print(cluster_profiles)
    
    

if __name__ == "__main__":
    
    dataset = pd.read_csv('Student Performance.csv')
    features_scaled, preprocessed_data, continuous_features = preprocessing_data(dataset)

    # K-Means-Clustering
    k_means_clustered_data = K_Means_Clustering(features_scaled, preprocessed_data)
    K_Means_visualization(features_scaled, k_means_clustered_data)
    evaluate_clustering(features_scaled, k_means_clustered_data, 'KMeans_Cluster')
    cluster_profiling(k_means_clustered_data, 'KMeans_Cluster')

    # AGNES-Clustering
    agnes_clustered_data = Agnes_Clustering(features_scaled, preprocessed_data)
    Agnes_visualization(features_scaled, agnes_clustered_data)
    evaluate_clustering(features_scaled, agnes_clustered_data, 'AGNES_Cluster')
    cluster_profiling(agnes_clustered_data, 'AGNES_Cluster')