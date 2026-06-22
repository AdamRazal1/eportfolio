import torch
import sklearn
import pandas as pd
import seaborn as sns

from torch import nn, optim
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Hyperparam

LEARNING_RATE = 0.001
EPOCHS = 400

# Device Agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# importing data
stud_performance = pd.read_csv('Student Performance.csv')
print(stud_performance.head()) # show first 5 sample of dataset
print(stud_performance.describe()) # Shows mean, std, min, percentile for numerical values of the dataset(per column)
print(stud_performance.info()) # show dtype and null values(per column)

### Data preprocessing step
len(stud_performance.columns)

## Data cleaning

# check for missing values
print(stud_performance.isnull().sum()) # have missing values
stud_performance = stud_performance.dropna() # drop missing values

# check for duplicate
print(stud_performance.duplicated().sum()) # no duplicates

# check for inconsistency
print(stud_performance[stud_performance['Exam_Score'] > 100])
stud_performance.drop(stud_performance[stud_performance['Exam_Score'] > 100].index, inplace = True)
print(stud_performance[stud_performance['Attendance'] > 100])
print(stud_performance[stud_performance['Previous_Scores'] > 100])

print(stud_performance[stud_performance['Exam_Score'] < 0])
print(stud_performance[stud_performance['Attendance'] < 0])
print(stud_performance[stud_performance['Previous_Scores'] < 0])

# check for uniqueness
print(stud_performance['Exam_Score'].min())
print(stud_performance['Exam_Score'].max())
print(stud_performance.nunique())

## Data Integration
## Data Transformation

# turning continuous data to numerical categorical data(equal-width binning)
continuous_features = ['Hours_Studied', 'Attendance', 'Previous_Scores']

bin_edges = {}

for column in continuous_features:
    stud_performance[column + '_binned'], bin_edges[column + '_binned'] = pd.cut(stud_performance[column], bins = 5, labels = [0, 1, 2, 3, 4], right = False, retbins=True)

stud_performance['Exam_Score_binned'] = pd.cut(stud_performance['Exam_Score'], bins = [0, 60, 65, 70, 75, 101], labels = [0, 1, 2, 3, 4], right = False)

stud_performance['Sleep_Hours_binned'] = pd.cut(stud_performance['Sleep_Hours'], bins = [0, 6, 9, float('inf')], labels = [0, 1, 2], right = False)

stud_performance['Tutoring_Sessions_binned'] = pd.cut(stud_performance['Tutoring_Sessions'], bins = [0, 2, 4, 6, 10], labels = [0, 1, 2, 3], right = False)

stud_performance['Physical_Activity_binned'] = pd.cut(stud_performance['Physical_Activity'], bins = [0, 2, 4, 6, 8], labels = [0, 1, 2, 3], right = False)


for column in continuous_features:
    print(stud_performance[column + '_binned'].unique())
    
exam_score_binned_percentage = stud_performance['Exam_Score_binned'].value_counts(normalize=True) * 100
print(f"Percentage distribution of Exam_Score_binned:\n{exam_score_binned_percentage}")

# turning categorical(nominal + ordinal) to numerical data
nominal_categorical_features = ['Extracurricular_Activities', 'Internet_Access', 'School_Type', 'Learning_Disabilities', 'Gender',]

ordinal_categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Peer_Influence', 'Parental_Education_Level', 'Distance_from_Home']

for column in stud_performance.columns:
    le = LabelEncoder()
    oe = OrdinalEncoder()
    if stud_performance[column].dtype != int:
        if column in nominal_categorical_features:
            stud_performance[column + '_encoded_n'] = le.fit_transform(stud_performance[column])
        elif column in ordinal_categorical_features:
            stud_performance[column + '_encoded_o'] = oe.fit_transform(stud_performance[[column]])


# turning all dtype of columns to either int/float

stud_performance['Hours_Studied_binned'] = stud_performance['Hours_Studied_binned'].cat.codes
stud_performance['Attendance_binned'] = stud_performance['Attendance_binned'].cat.codes
stud_performance['Sleep_Hours_binned'] = stud_performance['Sleep_Hours_binned'].cat.codes
stud_performance['Tutoring_Sessions_binned'] = stud_performance['Tutoring_Sessions_binned'].cat.codes
stud_performance['Physical_Activity_binned'] = stud_performance['Physical_Activity_binned'].cat.codes
stud_performance['Previous_Scores_binned'] = stud_performance['Previous_Scores_binned'].cat.codes
stud_performance['Exam_Score_binned'] = stud_performance['Exam_Score_binned'].cat.codes


print(stud_performance.info()) # show dtype and null values(per column)


# uploading modified dataset into new csv file

# stud_performance.to_csv('modif_student_performance.csv', index = False)
print('modified student performance succesfully loaded into new file')

## Data Reduction
## Data discretization

X = stud_performance[['Hours_Studied_binned', 'Attendance_binned', 'Parental_Involvement_encoded_o',
       'Access_to_Resources_encoded_o', 'Extracurricular_Activities_encoded_n', 'Sleep_Hours_binned',
       'Previous_Scores_binned',
       'Motivation_Level_encoded_o', 'Internet_Access_encoded_n',
       'Tutoring_Sessions_binned',
       'Family_Income_encoded_o', 'Teacher_Quality_encoded_o', 'School_Type_encoded_n',
       'Peer_Influence_encoded_o',
       'Physical_Activity_binned',
       'Learning_Disabilities_encoded_n',
       'Distance_from_Home_encoded_o', 'Gender_encoded_n', ]]


y = stud_performance['Parental_Education_Level_encoded_o']

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.to_numpy().astype('int64')

### Post-process data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

### Classification Model - Neural Network

class neural_network_model(nn.Module):
    def  __init__(self, input_layer, output_layer, hidden_layer, dropout = 0.3):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features = input_layer, out_features = hidden_layer),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features = hidden_layer, out_features = hidden_layer),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features = hidden_layer, out_features = output_layer),
        )

    def forward(self, x):
        return self.layer_stack(x)
        
model = neural_network_model(input_layer = X.shape[1], output_layer = 3, hidden_layer = 128) 
print(model.state_dict())

# Initializing loss function and optimizer
loss_fn = nn.CrossEntropyLoss() # expect raw logits not predictions
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=1e-4)

# metrics of evaluation


# Training the model

model = model.to(device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(EPOCHS):
    ### Training
    model.train() # train mode is on by default after construction

    # 1. Forward pass
    y_pred = model(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # # calculating accuracy
    # acc = accuracy_fn(y_true = y_train, y_pred = y_pred)

    # 3. Zero grad optimizer
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    ### Testing
    model.eval() # put the model in evaluation mode for testing (inference)
    # 1. Forward pass
    with torch.inference_mode():
        test_pred = model(X_test)
    
        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")
