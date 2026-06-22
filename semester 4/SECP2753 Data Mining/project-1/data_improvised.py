import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 1000
BATCH_SIZE = 32

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load and explore data
stud_performance = pd.read_csv('Student Performance.csv')
print("Dataset shape:", stud_performance.shape)
print("\nFirst 5 rows:")
print(stud_performance.head())
print("\nDataset info:")
print(stud_performance.info())

# Data cleaning
print(f"\nMissing values:\n{stud_performance.isnull().sum()}")
stud_performance = stud_performance.dropna()

print(f"Duplicates: {stud_performance.duplicated().sum()}")
stud_performance = stud_performance.drop_duplicates()

# Remove outliers (scores > 100)
print(f"Exam scores > 100: {len(stud_performance[stud_performance['Exam_Score'] > 100])}")
stud_performance = stud_performance[stud_performance['Exam_Score'] <= 100]

# Create target variable - categorize exam scores into performance levels
def categorize_score(score):
    if score < 60:
        return 0  # Poor
    elif score < 75 and score >= 60:
        return 1  # Average
    else:
        return 2  # Good

stud_performance['Performance_Category'] = stud_performance['Exam_Score'].apply(categorize_score)
print(f"\nPerformance distribution:")
print(stud_performance['Performance_Category'].value_counts().sort_index())

# Prepare features
# Numerical features - normalize but don't bin (preserve information)
numerical_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 
                     'Tutoring_Sessions', 'Physical_Activity']

# Categorical features
categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
                       'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
                       'School_Type', 'Peer_Influence', 'Learning_Disabilities',
                       'Parental_Education_Level', 'Distance_from_Home', 'Gender']

# Create a copy for preprocessing
df_processed = stud_performance.copy()

# Encode categorical variables
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    df_processed[feature + '_encoded'] = le.fit_transform(df_processed[feature])
    label_encoders[feature] = le

# Prepare feature matrix
feature_columns = numerical_features + [f + '_encoded' for f in categorical_features]
X = df_processed[feature_columns].values
y = df_processed['Performance_Category'].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class StudentPerformanceNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], num_classes=3, dropout=0.3):
        super(StudentPerformanceNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Initialize model
model = StudentPerformanceNN(input_size=X_train_tensor.shape[1])
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

# Training function
def train_epoch(model, X_train, y_train, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Move data to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += y_train.size(0)
    correct += (predicted == y_train).sum().item()
    
    return total_loss, 100 * correct / total

# Testing function
def test_epoch(model, X_test, y_test, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
        outputs = model(X_test)
        loss = criterion(outputs, y_test)
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()
    
    return total_loss, 100 * correct / total, predicted.cpu().numpy()

# Training loop
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

print("\nStarting training...")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, X_train_tensor, y_train_tensor, criterion, optimizer, device)
    test_loss, test_acc, _ = test_epoch(model, X_test_tensor, y_test_tensor, criterion, device)
    
    scheduler.step(test_loss)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    if epoch % 50 == 0:
        print(f'Epoch [{epoch}/{EPOCHS}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# Final evaluation
model.eval()
with torch.no_grad():
    X_test_device = X_test_tensor.to(device)
    outputs = model(X_test_device)
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()

# Calculate metrics
accuracy = accuracy_score(y_test, predicted)
print(f"\nFinal Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
class_names = ['Poor (0-59)', 'Average (60-74)', 'Good (75-100)']
print(classification_report(y_test, predicted, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predicted))

# Feature importance (simple method using gradients)
def get_feature_importance(model, X_sample, feature_names):
    model.eval()
    X_sample = torch.tensor(X_sample, dtype=torch.float32, requires_grad=True).to(device)
    
    output = model(X_sample)
    # Take the max class for each sample
    max_scores, _ = torch.max(output, dim=1)
    grad = torch.autograd.grad(max_scores.sum(), X_sample)[0]
    
    # Average absolute gradients across samples
    importance = torch.abs(grad).mean(dim=0).cpu().numpy()
    
    return importance

# Get feature importance
importance = get_feature_importance(model, X_test_scaled[:100], feature_columns)
feature_importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10))

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\nModel has {sum(p.numel() for p in model.parameters())} parameters")
print("Training completed!")