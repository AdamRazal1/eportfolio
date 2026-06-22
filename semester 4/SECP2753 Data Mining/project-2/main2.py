import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_selection import SelectKBest, f_classif
from kneed import KneeLocator

def preprocessing_data(dataset):
    ## Data Cleaning
    print("Original dataset shape:", dataset.shape)
    
    # Eliminate missing values
    print("Missing values:", dataset.isnull().sum().sum())
    dataset = dataset.dropna()

    # Eliminate duplicate values
    duplicates = dataset.duplicated().sum()
    print("Duplicate values:", duplicates)
    if duplicates > 0:
        dataset = dataset.drop_duplicates()

    # Eliminate inconsistent values with more flexible bounds
    initial_size = len(dataset)
    dataset = dataset[
        (dataset['Exam_Score'] >= 0) & (dataset['Exam_Score'] <= 100) &
        (dataset['Attendance'] >= 0) & (dataset['Attendance'] <= 100) &
        (dataset['Previous_Scores'] >= 0) & (dataset['Previous_Scores'] <= 100) &
        (dataset['Hours_Studied'] >= 0) & (dataset['Hours_Studied'] <= 24) &
        (dataset['Sleep_Hours'] >= 0) & (dataset['Sleep_Hours'] <= 24) &
        (dataset['Tutoring_Sessions'] >= 0) &
        (dataset['Physical_Activity'] >= 0)
    ]
    print(f"Removed {initial_size - len(dataset)} inconsistent records")

    # Reset index
    dataset.reset_index(drop=True, inplace=True)

    ## Feature Engineering
    # Create new meaningful features
    dataset['Study_Sleep_Ratio'] = dataset['Hours_Studied'] / (dataset['Sleep_Hours'] + 1)  # +1 to avoid division by zero
    dataset['Performance_Consistency'] = abs(dataset['Exam_Score'] - dataset['Previous_Scores'])
    dataset['Total_Support'] = dataset['Tutoring_Sessions'] + dataset['Physical_Activity']
    
    ## Data Transformation
    categorical_features = ['Extracurricular_Activities', 'Internet_Access', 'School_Type', 
                          'Learning_Disabilities', 'Gender', 'Parental_Involvement', 
                          'Access_to_Resources', 'Motivation_Level', 'Family_Income', 
                          'Teacher_Quality', 'Peer_Influence', 'Parental_Education_Level', 
                          'Distance_from_Home']

    continuous_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 
                          'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']
    
    all_features = continuous_features + categorical_features

    # Encoding categorical features
    encoded_categorical_features = pd.get_dummies(dataset[categorical_features], drop_first=True)


    # Create processed_dataset
    processed_dataset = dataset[continuous_features].copy()
    processed_dataset = pd.concat([processed_dataset, encoded_categorical_features], axis=1)

    print("Processed dataset shape:", processed_dataset.shape)
    
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(processed_dataset)

    # Feature Selection
    n_clusters = 3
    random_state = 42
    max_features = None
    selected_features = []
    remaining_features = list(all_features)
    best_overall_score = -1.0 # Silhouette score ranges from -1 to 1
    best_feature_set_so_far = []

    history_selected_features = []
    history_scores = []

    print("Starting Forward Feature Selection for Clustering...")
    print(f"Number of initial features: {len(all_features)}")
    print(f"Number of clusters (KMeans): {n_clusters}")

    iteration = 0
    while remaining_features:
        iteration += 1
        current_best_feature_for_iter = None
        current_best_score_for_iter = -1.0

        candidates = []
        for feature in remaining_features:
            temp_features = selected_features + [feature]
            # Ensure at least 2 features for silhouette_score if only one feature exists initially
            if len(temp_features) == 1:
                # Silhouette score is not meaningful for a single feature,
                # you might want to use another metric or skip evaluation
                # for single-feature sets. For simplicity, we'll assign a very low score
                # or handle it as a special case.
                # For a true single feature, it's hard to define clusters meaningfully with KMeans based on centroids.
                # Here, we'll just skip adding single features unless it's the first one.
                # In practice, you often want at least 2 features for meaningful clustering.
                if not selected_features: # If it's the very first feature being considered
                    # For a single feature, KMeans will likely just create n_clusters based on value ranges.
                    # Silhouette score for a single feature can be misleading.
                    # You might consider using a different metric or thresholding this.
                    # For this example, let's allow it to be evaluated.
                    pass
                else:
                    continue # Skip if already selected features and only adding one more leading to just 2 total

            X_subset = X_processed[temp_features]
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            clusters = kmeans.fit_predict(X_subset)

            # Check if there's only one unique cluster label, which can happen with
            # very few or trivial features, making silhouette score undefined.
            if len(np.unique(clusters)) < 2:
                score = -1.0 # Assign a very low score if only one cluster is formed
            else:
                score = silhouette_score(X_subset, clusters)

            candidates.append((feature, score))

        # Find the best candidate feature to add in this iteration
        if not candidates:
            print("No valid candidates to add in this iteration. Stopping.")
            break

        candidates.sort(key=lambda x: x[1], reverse=True) # Sort by score (descending)
        current_best_feature_for_iter, current_best_score_for_iter = candidates[0]

        # Stopping criterion: If adding the best candidate doesn't improve the score
        if current_best_score_for_iter > best_overall_score:
            best_overall_score = current_best_score_for_iter
            selected_features.append(current_best_feature_for_iter)
            remaining_features.remove(current_best_feature_for_iter)
            best_feature_set_so_far = list(selected_features) # Update best set
            history_selected_features.append(list(selected_features))
            history_scores.append(best_overall_score)
            print(f"Iteration {iteration}: Added '{current_best_feature_for_iter}', new best score: {best_overall_score:.4f}, current features: {selected_features}")
        else:
            print(f"Iteration {iteration}: No significant improvement. Best score did not improve from {best_overall_score:.4f} by adding '{current_best_feature_for_iter}' (score: {current_best_score_for_iter:.4f}). Stopping.")
            break

        if max_features is not None and len(selected_features) >= max_features:
            print(f"Maximum number of features ({max_features}) reached. Stopping.")
            break

    print("\nForward Feature Selection Complete.")
    print(f"Best selected features: {best_feature_set_so_far}")
    print(f"Best silhouette score: {best_overall_score:.4f}")
    

    # ## Outlier Detection and Treatment
    # # Use IQR method to cap outliers instead of removing them
    # for col in continuous_features:
    #     if col in processed_dataset.columns:
    #         Q1 = processed_dataset[col].quantile(0.25)
    #         Q3 = processed_dataset[col].quantile(0.75)
    #         IQR = Q3 - Q1
    #         lower_bound = Q1 - 1.5 * IQR
    #         upper_bound = Q3 + 1.5 * IQR
            
    #         # Cap outliers instead of removing them
    #         processed_dataset[col] = processed_dataset[col].clip(lower_bound, upper_bound)

    # processed_dataset.to_csv('processed_data.csv', index=False)

    ## Sampling and Scaling
    features_column = processed_dataset.columns.tolist()
    
    # Use RobustScaler instead of StandardScaler for better outlier handling
    # scaler = StandardScaler()  # Less sensitive to outliers than StandardScaler
    # features_scaled = scaler.fit_transform(processed_dataset[features_column].values)
    
    print("Scaled features shape:", features_scaled.shape)
    
    return features_scaled, processed_dataset, continuous_features, scaler

def find_optimal_clusters(features_scaled, max_k=15, methods=['elbow', 'silhouette']):
    """Find optimal number of clusters using multiple methods"""
    
    results = {}
    k_range = range(2, max_k + 1)
    
    # Elbow Method
    if 'elbow' in methods:
        inertias = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
            kmeans.fit(features_scaled)
            inertias.append(kmeans.inertia_)
        
        # Use KneeLocator to find the elbow automatically
        try:
            kl = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
            results['elbow'] = kl.elbow if kl.elbow else 3
        except:
            results['elbow'] = 3
            
        # Plot Elbow Method
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.xticks(k_range)
        plt.grid(True)
        if 'elbow' in results:
            plt.axvline(x=results['elbow'], color='red', linestyle='--', 
                       label=f'Optimal K = {results["elbow"]}')
            plt.legend()
        plt.show()
    
    # Silhouette Analysis
    if 'silhouette' in methods:
        silhouette_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
            cluster_labels = kmeans.fit_predict(features_scaled)
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        results['silhouette'] = k_range[np.argmax(silhouette_scores)]
        
        # Plot Silhouette Analysis
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, marker='o')
        plt.title('Silhouette Analysis for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Average Silhouette Score')
        plt.xticks(k_range)
        plt.grid(True)
        plt.axvline(x=results['silhouette'], color='red', linestyle='--', 
                   label=f'Optimal K = {results["silhouette"]}')
        plt.legend()
        plt.show()
    
    return results

def improved_kmeans_clustering(features_scaled, processed_data, optimal_k=None):
    """Improved K-Means with better initialization and parameters"""
    
    if optimal_k is None:
        cluster_results = find_optimal_clusters(features_scaled)
        optimal_k = cluster_results.get('silhouette', 3)
    
    print(f"Using K = {optimal_k} for K-Means clustering")
    
    # Try multiple initializations and pick the best one
    best_score = -1
    best_kmeans = None
    
    for init_method in ['k-means++', 'random']:
        for random_state in [42, 123, 456]:
            kmeans = KMeans(
                n_clusters=optimal_k, 
                random_state=random_state, 
                n_init=20,  # Increased from 10
                max_iter=500,  # Increased iterations
                init=init_method,
                algorithm='elkan'  # Faster for dense data
            )
            labels = kmeans.fit_predict(features_scaled)
            score = silhouette_score(features_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_kmeans = kmeans
    
    kmeans_labels = best_kmeans.predict(features_scaled)
    processed_data = processed_data.copy()
    processed_data['KMeans_Cluster'] = kmeans_labels
    
    print(f"Best K-Means Silhouette Score: {best_score:.4f}")
    
    return processed_data, best_kmeans

def improved_agnes_clustering(features_scaled, processed_data, optimal_k=None):
    """Improved AGNES clustering with different linkage methods"""
    
    if optimal_k is None:
        optimal_k = 3
    
    print(f"Using K = {optimal_k} for AGNES clustering")
    
    # Try different linkage methods
    best_score = -1
    best_agnes = None
    best_linkage = None
    
    linkage_methods = ['ward', 'complete', 'average']
    
    for linkage_method in linkage_methods:
        try:
            agnes = AgglomerativeClustering(
                n_clusters=optimal_k, 
                linkage=linkage_method,
                metric='euclidean' if linkage_method == 'ward' else 'cosine'
            )
            labels = agnes.fit_predict(features_scaled)
            score = silhouette_score(features_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_agnes = agnes
                best_linkage = linkage_method
        except:
            continue
    
    agnes_labels = best_agnes.fit_predict(features_scaled)
    processed_data = processed_data.copy()
    processed_data['AGNES_Cluster'] = agnes_labels
    
    print(f"Best AGNES Silhouette Score: {best_score:.4f} (linkage: {best_linkage})")
    
    return processed_data, best_agnes

def dbscan_clustering(features_scaled, processed_data):
    """DBSCAN clustering as an alternative method"""
    
    # Find optimal eps using k-distance graph
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(features_scaled)
    distances, indices = neighbors_fit.kneighbors(features_scaled)
    distances = np.sort(distances[:, 4], axis=0)
    
    # Use knee point detection or set eps manually
    try:
        kl = KneeLocator(range(len(distances)), distances, curve="convex", direction="increasing")
        optimal_eps = distances[kl.knee] if kl.knee else np.percentile(distances, 90)
    except:
        optimal_eps = np.percentile(distances, 90)
    
    print(f"Using eps = {optimal_eps:.4f} for DBSCAN")
    
    # DBSCAN clustering
    dbscan = DBSCAN(eps=optimal_eps, min_samples=5, metric='euclidean')
    dbscan_labels = dbscan.fit_predict(features_scaled)
    
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"DBSCAN found {n_clusters} clusters with {n_noise} noise points")
    
    if n_clusters > 1:
        # Only calculate silhouette score if we have more than 1 cluster
        mask = dbscan_labels != -1  # Exclude noise points
        if np.sum(mask) > 0 and len(set(dbscan_labels[mask])) > 1:
            score = silhouette_score(features_scaled[mask], dbscan_labels[mask])
            print(f"DBSCAN Silhouette Score: {score:.4f}")
        
        processed_data = processed_data.copy()
        processed_data['DBSCAN_Cluster'] = dbscan_labels
        return processed_data, dbscan
    else:
        print("DBSCAN could not find meaningful clusters")
        return None, None

def comprehensive_evaluation(features_scaled, processed_data, cluster_column):
    """Comprehensive clustering evaluation with multiple metrics"""
    
    labels = processed_data[cluster_column].values
    
    # Remove noise points for DBSCAN
    if -1 in labels:
        mask = labels != -1
        features_eval = features_scaled[mask]
        labels_eval = labels[mask]
    else:
        features_eval = features_scaled
        labels_eval = labels
    
    # Skip evaluation if not enough clusters
    if len(set(labels_eval)) < 2:
        print(f"Cannot evaluate {cluster_column}: insufficient clusters")
        return
    
    print(f"\n=== Evaluation for {cluster_column} ===")
    
    # Silhouette Score (higher is better, range: -1 to 1)
    sil_score = silhouette_score(features_eval, labels_eval)
    print(f"Silhouette Score: {sil_score:.4f}")
    
    # Davies-Bouldin Index (lower is better, range: 0 to infinity)
    db_score = davies_bouldin_score(features_eval, labels_eval)
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    
    # Calinski-Harabasz Index (higher is better)
    ch_score = calinski_harabasz_score(features_eval, labels_eval)
    print(f"Calinski-Harabasz Index: {ch_score:.4f}")
    
    # Cluster distribution
    unique, counts = np.unique(labels_eval, return_counts=True)
    print("Cluster sizes:", dict(zip(unique, counts)))
    
    return {
        'silhouette': sil_score,
        'davies_bouldin': db_score,
        'calinski_harabasz': ch_score,
        'cluster_sizes': dict(zip(unique, counts))
    }

def enhanced_visualization(features_scaled, processed_data, cluster_column, method_name):
    """Enhanced visualization with better PCA analysis"""
    
    # Apply PCA
    pca = PCA()
    principal_components = pca.fit_transform(features_scaled)
    
    # Print explained variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    print(f"\nPCA Analysis for {method_name}:")
    print(f"First 3 components explain {cumsum_var[2]:.2%} of variance")
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame(
        data=principal_components[:, :3], 
        columns=['PC1', 'PC2', 'PC3']
    )
    pca_df[cluster_column] = processed_data[cluster_column]
    
    # 3D Plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_clusters = sorted(pca_df[cluster_column].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = pca_df[pca_df[cluster_column] == cluster_id]
        label = f'Noise ({len(cluster_data)})' if cluster_id == -1 else f'Cluster {cluster_id} ({len(cluster_data)})'
        
        ax.scatter(
            cluster_data['PC1'],
            cluster_data['PC2'], 
            cluster_data['PC3'],
            color=colors[i],
            s=60,
            alpha=0.7,
            label=label
        )
    
    ax.set_title(f'{method_name} Clusters (PCA Visualization)')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} var)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def enhanced_cluster_profiling(processed_data, cluster_column, continuous_features):
    """Enhanced cluster profiling with statistical analysis"""
    
    print(f"\n=== Cluster Profiling for {cluster_column} ===")
    
    # Basic statistics
    cluster_profiles = processed_data.groupby(cluster_column)[continuous_features].agg(['mean', 'std']).round(3)
    print("\nCluster Profiles (Mean ± Std):")
    print(cluster_profiles)
    
    # Feature importance analysis
    cluster_means = processed_data.groupby(cluster_column)[continuous_features].mean()
    
    # Calculate coefficient of variation for each feature across clusters
    feature_variation = {}
    for feature in continuous_features:
        if feature in cluster_means.columns:
            cv = cluster_means[feature].std() / cluster_means[feature].mean()
            feature_variation[feature] = cv
    
    # Sort features by variation (most discriminative first)
    sorted_features = sorted(feature_variation.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nMost discriminative features for {cluster_column}:")
    for feature, cv in sorted_features[:5]:
        print(f"- {feature}: CV = {cv:.3f}")

# Main execution function
def run_improved_clustering_analysis(dataset_path):
    """Run the complete improved clustering analysis"""
    
    # Load data
    dataset = pd.read_csv(dataset_path)
    print("Dataset loaded successfully!")
    
    # Preprocessing
    features_scaled, processed_data, continuous_features, scaler = preprocessing_data(dataset)
    
    # Find optimal number of clusters
    optimal_clusters = find_optimal_clusters(features_scaled, max_k=10)
    optimal_k = optimal_clusters.get('silhouette', 3)
    
    print(f"\nOptimal number of clusters: {optimal_k}")
    
    # K-Means Clustering
    print("\n" + "="*50)
    print("IMPROVED K-MEANS CLUSTERING")
    print("="*50)
    
    kmeans_data, kmeans_model = improved_kmeans_clustering(features_scaled, processed_data, optimal_k)
    enhanced_visualization(features_scaled, kmeans_data, 'KMeans_Cluster', 'K-Means')
    kmeans_scores = comprehensive_evaluation(features_scaled, kmeans_data, 'KMeans_Cluster')
    enhanced_cluster_profiling(kmeans_data, 'KMeans_Cluster', continuous_features)
    
    # AGNES Clustering
    print("\n" + "="*50)
    print("IMPROVED AGNES CLUSTERING")
    print("="*50)
    
    agnes_data, agnes_model = improved_agnes_clustering(features_scaled, processed_data, optimal_k)
    enhanced_visualization(features_scaled, agnes_data, 'AGNES_Cluster', 'AGNES')
    agnes_scores = comprehensive_evaluation(features_scaled, agnes_data, 'AGNES_Cluster')
    enhanced_cluster_profiling(agnes_data, 'AGNES_Cluster', continuous_features)
    
    # DBSCAN Clustering
    print("\n" + "="*50)
    print("DBSCAN CLUSTERING")
    print("="*50)
    
    dbscan_data, dbscan_model = dbscan_clustering(features_scaled, processed_data)
    if dbscan_data is not None:
        enhanced_visualization(features_scaled, dbscan_data, 'DBSCAN_Cluster', 'DBSCAN')
        dbscan_scores = comprehensive_evaluation(features_scaled, dbscan_data, 'DBSCAN_Cluster')
        enhanced_cluster_profiling(dbscan_data, 'DBSCAN_Cluster', continuous_features)
    
    # Summary comparison
    print("\n" + "="*50)
    print("CLUSTERING COMPARISON SUMMARY")
    print("="*50)
    
    if kmeans_scores:
        print(f"K-Means    - Silhouette: {kmeans_scores['silhouette']:.4f}, Davies-Bouldin: {kmeans_scores['davies_bouldin']:.4f}")
    if agnes_scores:
        print(f"AGNES      - Silhouette: {agnes_scores['silhouette']:.4f}, Davies-Bouldin: {agnes_scores['davies_bouldin']:.4f}")
    if 'dbscan_scores' in locals() and dbscan_scores:
        print(f"DBSCAN     - Silhouette: {dbscan_scores['silhouette']:.4f}, Davies-Bouldin: {dbscan_scores['davies_bouldin']:.4f}")
    
    return {
        'kmeans_data': kmeans_data,
        'agnes_data': agnes_data, 
        'dbscan_data': dbscan_data,
        'features_scaled': features_scaled,
        'scaler': scaler
    }

# Run the analysis
if __name__ == "__main__":
    results = run_improved_clustering_analysis('Student Performance.csv')