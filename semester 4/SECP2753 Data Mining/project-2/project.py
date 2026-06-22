import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

def preprocessing_data(dataset):

    dataset = pd.read_csv('Student Performance.csv')

    ## Data Cleaning

    # Eliminate missing values
    print(dataset.isnull().sum())
    dataset = dataset.dropna()

    # Eliminate duplicate values
    print(dataset.duplicated().sum()) # no duplicated values

    # Eliminate inconsistent values
    dataset.drop(dataset[dataset['Exam_Score'] > 100].index , inplace = True)
    dataset.drop(dataset[dataset['Exam_Score'] < 0].index , inplace = True)
    dataset.drop(dataset[dataset['Attendance'] > 100].index , inplace = True)
    dataset.drop(dataset[dataset['Attendance'] < 0].index , inplace = True)
    dataset.drop(dataset[dataset['Previous_Scores'] > 100].index , inplace = True)
    dataset.drop(dataset[dataset['Previous_Scores'] < 0].index , inplace = True)

    dataset = dataset.reset_index(drop=True)

    ## Data Transformation

    nominal_features = ['Extracurricular_Activities', 'Internet_Access', 'School_Type', 'Learning_Disabilities', 'Gender',]
    ordinal_features = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Peer_Influence', 'Parental_Education_Level', 'Distance_from_Home']
    continuous_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity', 'Exam_Score']

    # encoding nominal and ordinal data
    ne = OneHotEncoder(sparse_output = False)
    nominal_encoded = ne.fit_transform(dataset[nominal_features])
    nominal_encoded = pd.DataFrame(nominal_encoded, columns = ne.get_feature_names_out(nominal_features))
    print('this is nominal encoded')
    print(nominal_encoded)


    oe = OrdinalEncoder()
    ordinal_encoded = oe.fit_transform(dataset[ordinal_features])
    ordinal_encoded = pd.DataFrame(ordinal_encoded, columns = oe.get_feature_names_out(ordinal_features))
    print('this is ordinal encoded')
    print(ordinal_encoded)

    combine_encoded = pd.concat([nominal_encoded, ordinal_encoded], axis = 1)


    ## Storing copy of processed data
    processed_dataset = pd.concat([combine_encoded, dataset[continuous_features]], axis = 1)

    print('this is the processed dataset')
    print(processed_dataset)

    ## Data Normalization

    scaler = StandardScaler()
    processed_dataset_scaled = scaler.fit_transform(processed_dataset[continuous_features].values)
    processed_dataset_scaled = pd.DataFrame(processed_dataset_scaled, columns = processed_dataset[continuous_features].columns)

    processed_dataset_scaled = pd.concat([combine_encoded, processed_dataset_scaled], axis = 1)

    print(processed_dataset_scaled)

    all_features = processed_dataset_scaled.columns
    print(all_features)

    # selected_features = []
    # best_features_combinations = []
    # best_score = -1
    # remaining_features = list(all_features)
    # print(remaining_features)

    # for idx in range(len(remaining_features)):
    #     best_score = -1
    #     best_candidate = None
    #     for feature in remaining_features:
    #         candidate_features = selected_features + [feature]
    #         X_feature = processed_dataset_scaled[candidate_features]
    #         kmeans = KMeans(n_clusters = 2, random_state = 42, n_init = 10)
    #         labels = kmeans.fit_predict(X_feature)
    #         score = silhouette_score(X_feature, labels)
    #         print(candidate_features)
    #         print(labels)
    #         print(score)

    #         if score >= best_score:
    #             best_score = score
    #             best_candidate = feature

    #     if best_candidate is not None:
    #         selected_features.append(best_candidate)
    #         remaining_features.remove(best_candidate)
    #         best_features_combinations.append((candidate_features, score))

    #     print(selected_features)

    # print(best_features_combinations)

    # Selecting best features based on our insight and best feature combinations

    selected_features = [
        'Gender_Male', 
        'Gender_Female', 
        'Internet_Access_Yes', 
        'Internet_Access_No', 
    ]

    selected_processed_dataset_scaled = processed_dataset_scaled[selected_features]

    selected_processed_dataset = processed_dataset[selected_features]

    return selected_processed_dataset_scaled, selected_processed_dataset

def KMeansClustering(selected_processed_dataset_scaled, selected_processed_data):

    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(selected_processed_dataset_scaled)
        inertia.append(kmeans.inertia_)

    # Perform clustering with optimal K = 3
    optimal_k = 5
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(selected_processed_dataset_scaled)
    selected_processed_data['KMeans_Cluster'] = kmeans_labels

    # Apply PCA for dimensionality reduction (2d)

    pca = PCA(n_components = 2)
    features_pca = pca.fit_transform(selected_processed_dataset_scaled)

    # Visualize the Cluster 2d scatter plot
    features_pca_df = pd.DataFrame(features_pca, columns = ['pc_1', 'pc_2'])
    features_pca_df['KMeans_Cluster'] = selected_processed_data['KMeans_Cluster']

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
    data=features_pca_df,
    x="pc_1", y="pc_2",
    hue="KMeans_Cluster",
    palette="Set2",
    alpha=0.6,
    edgecolor='k'
    )

    plt.title("KMeans Clustering Results (k=5) on Student Performance Factors")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":

    dataset = pd.read_csv('Student Performance.csv')
    selected_processed_dataset_scaled, selected_processed_data = preprocessing_data(dataset)
    KMeansClustering(selected_processed_dataset_scaled, selected_processed_data)