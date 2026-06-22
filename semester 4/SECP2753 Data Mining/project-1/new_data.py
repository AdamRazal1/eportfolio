import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torch import nn, optim
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


LEARNING_RATE = 0.001
EPOCHS = 300

def preprocessing_data(dataset):

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

    ## Data Transformation

    nominal_features = ['Extracurricular_Activities', 'Internet_Access', 'School_Type', 'Learning_Disabilities', 'Gender',]
    ordinal_features = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Peer_Influence', 'Parental_Education_Level', 'Distance_from_Home']
    continuous_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity']

    # encoding nominal and ordinal data
    for feature in nominal_features:
        ne = LabelEncoder()
        dataset[feature + '_encoded_n'] = ne.fit_transform(dataset[feature])

    for feature in ordinal_features:
        oe = OrdinalEncoder()
        dataset[feature + '_encoded_o'] = oe.fit_transform(dataset[[feature]])

    # binning continuous data

    dataset['Exam_Score_binned'] = dataset['Exam_Score'].apply(categorize_score)
    dataset['Hours_Studied_binned'] = dataset['Hours_Studied'].apply(categorize_study_hours)
    dataset['Attendance_binned'] = dataset['Attendance'].apply(categorize_attendance)
    dataset['Previous_Scores_binned'] = dataset['Previous_Scores'].apply(categorize_previous_score)
    dataset['Tutoring_Sessions_binned'] = dataset['Tutoring_Sessions'].apply(categorize_tutoring_sessions)
    dataset['Physical_Activity_binned'] = dataset['Physical_Activity'].apply(categorize_physical_activity)
    dataset['Sleep_Hours_binned'] = dataset['Sleep_Hours'].apply(categorize_sleep_hours)
    binned_continuous_features = ['Hours_Studied_binned', 'Attendance_binned', 'Previous_Scores_binned', 'Sleep_Hours_binned', 'Tutoring_Sessions_binned', 'Physical_Activity_binned', 'Exam_Score_binned']

    print(binned_continuous_features)

    ## Storing copy of processed data
    processed_dataset = dataset.copy()
    # processed_dataset.to_csv('processed_data.csv', index = False)

    ## Sampling
    features_column = [f + '_encoded_n' for f in nominal_features] + [f + '_encoded_o' for f in ordinal_features if f != '' ] + [f for f in binned_continuous_features if f != 'Exam_Score_binned']

    label = ['Exam_Score_binned', 'Motivation_Level_encoded_o', 'Access_to_Resources_encoded_o']

    # Creating X and y
    X = processed_dataset[features_column].values
    y = processed_dataset[label[0]].values


    # Creating training and testing set

    X_train, X_test, y_train, y_test = train_test_split(X ,y, random_state = 42, train_size = 0.7, test_size = 0.3, stratify = y)

    ## Normalize the numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def training_and_testing(model, X_train, X_test, y_train, y_test):
    ## Prepare for training and testing
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    
    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses = []
    test_losses = []

    for epoch in range(EPOCHS):

        ## Training phase
        model.train()
        # 1. Forward pass
        y_pred = model(X_train)
        # 2. Calculate loss
        loss = loss_fn(y_pred, y_train)
        # 3. Zero grad optimizer
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Step the optimizer
        optimizer.step()
        ## Testing phase
        model.eval()

        # 1. Forward pass
        with torch.inference_mode():
            test_pred = model(X_test)
            # 2. Calculate the loss
            test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 0 or epoch == (EPOCHS - 1):
            print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

    return train_losses, test_losses


def model_evaluation(model, X_test, y_test, train_loss = None, test_loss = None):
    """
    Evaluate the model and calculate accuracy, precision, recall, and F1-score
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Get device
    device = next(model.parameters()).device
    
    # Make predictions
    with torch.inference_mode():
        # Get raw predictions (logits)
        test_logits = model(X_test)
        
        # Convert logits to predicted classes
        test_pred_probs = torch.softmax(test_logits, dim=1)
        test_pred_labels = torch.argmax(test_pred_probs, dim=1)
    
    # Move predictions and true labels to CPU for sklearn metrics
    y_pred_cpu = test_pred_labels.cpu().numpy()
    y_true_cpu = y_test.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_cpu, y_pred_cpu)
    precision = precision_score(y_true_cpu, y_pred_cpu, average='weighted')
    recall = recall_score(y_true_cpu, y_pred_cpu, average='weighted')
    f1 = f1_score(y_true_cpu, y_pred_cpu, average='weighted')
    
    # Plotting testing against training loss
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
        
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, test_loss, 'r-', label='Testing Loss', linewidth=2)
        
    plt.title(f'Training vs Testing Loss Over {EPOCHS} Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true_cpu, y_pred_cpu)   

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Poor', 'Below Average', 'Average', 'Good', 'Excellent'],
                yticklabels=['Poor', 'Below Average', 'Average', 'Good', 'Excellent'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

     
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'y_true': y_true_cpu,
        'y_pred': y_pred_cpu,
        'confusion_matrix': cm
    }  

def categorize_score(score):
    if score < 60 and score >= 50:
        return 0  # Poor
    elif score < 70 and score >= 60:
        return 1  # Average
    elif score < 80 and score >= 70:
        return 2  # Average
    elif score < 90 and score >= 80:
        return 3  # Average
    else:
        return 4  # Good
    
def categorize_study_hours(study_hours):
    if study_hours < 10 and study_hours >= 0:
        return 0  # Poor
    elif study_hours < 20 and study_hours >= 10:
        return 1  # Average
    elif study_hours < 30 and study_hours >= 20:
        return 2  # Average
    elif study_hours < 40 and study_hours >= 30:
        return 3  # Average
    else:
        return 4  # Good
    
def categorize_attendance(attendance):
    if attendance < 20:
        return 0
    elif attendance < 40:
        return 1
    elif attendance < 60:
        return 2
    elif attendance < 80:
        return 3
    else:
        return 4

def categorize_previous_score(score):
    if score < 20:
        return 0
    elif score < 40:
        return 1
    elif score < 60:
        return 2
    elif score < 80:
        return 3
    else:
        return 4

def categorize_tutoring_sessions(sessions):
    if sessions < 2:
        return 0
    elif sessions < 4:
        return 1
    elif sessions < 6:
        return 2
    else:
        return 3

def categorize_physical_activity(activity):
    if activity < 2:
        return 0
    elif activity < 4:
        return 1
    elif activity < 6:
        return 2
    else:
        return 3

def categorize_sleep_hours(hours):
    if hours <= 6:
        return 0
    elif hours <= 9:
        return 1
    else:
        return 2

def create_correlation_heatmap(dataset, figsize=(15, 12), save_path=None):
    """
    Create a correlation heatmap for all features in the dataset.
    
    Parameters:
    dataset (pd.DataFrame): The input dataset
    figsize (tuple): Figure size for the heatmap (width, height)
    save_path (str): Optional path to save the heatmap image
    
    Returns:
    pd.DataFrame: Correlation matrix of all features
    """
    
    # Create a copy of the dataset to avoid modifying the original
    df_encoded = dataset.copy()
    
    # Identify categorical columns
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical variables using Label Encoder
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    # Calculate correlation matrix
    correlation_matrix = df_encoded.corr()
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    
    # Create mask for upper triangle (optional - to show only lower triangle)
    # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(correlation_matrix, 
                annot=True,           # Show correlation values
                cmap='coolwarm',      # Color scheme
                center=0,             # Center colormap at 0
                square=True,          # Make cells square-shaped
                fmt='.2f',            # Format numbers to 2 decimal places
                cbar_kws={"shrink": .8},  # Shrink colorbar
                # mask=mask           # Uncomment to show only lower triangle
                )
    
    plt.title('Correlation Heatmap of All Features\n(Student Performance Dataset)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    plt.show()
    
    # Print some interesting correlations
    print("\n" + "="*50)
    print("TOP POSITIVE CORRELATIONS:")
    print("="*50)
    
    # Get upper triangle of correlation matrix (excluding diagonal)
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # Find top positive correlations
    correlations_series = upper_tri.unstack().dropna()
    top_positive = correlations_series.nlargest(10)
    
    for i, (pair, corr) in enumerate(top_positive.items(), 1):
        print(f"{i:2d}. {pair[0]} ↔ {pair[1]}: {corr:.3f}")
    
    print("\n" + "="*50)
    print("TOP NEGATIVE CORRELATIONS:")
    print("="*50)
    
    # Find top negative correlations
    top_negative = correlations_series.nsmallest(10)
    
    for i, (pair, corr) in enumerate(top_negative.items(), 1):
        print(f"{i:2d}. {pair[0]} ↔ {pair[1]}: {corr:.3f}")
    
    print("\n" + "="*50)
    print("CORRELATIONS WITH EXAM_SCORE:")
    print("="*50)
    
    # Show correlations with Exam_Score (target variable)
    exam_score_corr = correlation_matrix['Exam_Score'].sort_values(ascending=False)
    exam_score_corr = exam_score_corr.drop('Exam_Score')  # Remove self-correlation
    
    for feature, corr in exam_score_corr.items():
        print(f"{feature}: {corr:.3f}")
    
    return correlation_matrix

def create_focused_heatmap(dataset, target_feature='Exam_Score', top_n=10, figsize=(10, 8)):
    """
    Create a focused heatmap showing correlations with a specific target feature.
    
    Parameters:
    dataset (pd.DataFrame): The input dataset
    target_feature (str): The target feature to focus on
    top_n (int): Number of top correlated features to display
    figsize (tuple): Figure size for the heatmap
    
    Returns:
    pd.Series: Top correlations with the target feature
    """
    
    # Create a copy and encode categorical variables
    df_encoded = dataset.copy()
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
    
    # Calculate correlations with target feature
    correlations = df_encoded.corr()[target_feature].sort_values(ascending=False)
    correlations = correlations.drop(target_feature)  # Remove self-correlation
    
    # Get top N positive and negative correlations
    top_positive = correlations.head(top_n // 2)
    top_negative = correlations.tail(top_n // 2)
    top_features = pd.concat([top_positive, top_negative])
    
    # Create subset of correlation matrix
    selected_features = list(top_features.index) + [target_feature]
    correlation_subset = df_encoded[selected_features].corr()
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_subset, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8})
    
    plt.title(f'Top {top_n} Features Correlated with {target_feature}', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return top_features 

class DeepLearning(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, dropout=0.3):
        super(DeepLearning, self).__init__()

        self.layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, output_features)
        )

    def forward(self, x):
        return self.layer_stack(x)
    


if __name__ == "__main__":


    dataset = pd.read_csv('Student Performance.csv')

    # Create full correlation heatmap
    print("Creating full correlation heatmap...")
    corr_matrix = create_correlation_heatmap(dataset, 
                                           figsize=(16, 14), 
                                           save_path='correlation_heatmap.png')
    
    # Create focused heatmap for Exam_Score
    print("\nCreating focused heatmap for Exam_Score...")
    top_corr = create_focused_heatmap(dataset, 
                                    target_feature='Exam_Score', 
                                    top_n=12,
                                    figsize=(12, 10))

    X_train, X_test, y_train, y_test = preprocessing_data(dataset)

    model = DeepLearning(X_train.shape[1], 128, 5)
    train_loss, test_loss = training_and_testing(model = model, X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test)

    model_results = model_evaluation(model, X_test = X_test, y_test = y_test, train_loss = train_loss, test_loss = test_loss)

    
    print(f'Accuracy : {model_results["accuracy"]:.2f}')
    print(f'precision: {model_results["precision"]:.2f}')
    print(f'recall: {model_results["recall"]:.2f}')
    print(f'f1-score: {model_results["f1_score"]:.2f}')