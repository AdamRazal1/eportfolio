import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# importing data
stud_performance = pd.read_csv('Student Performance.csv')
print(stud_performance.head()) # show first 5 sample of dataset
print(stud_performance.describe()) # Shows mean, std, min, percentile for numerical values of the dataset(per column)
print(stud_performance.info()) # show dtype and null values(per column)

len(stud_performance.columns)

### Data preprocessing step

## Data cleaning

# check for missing values
print(stud_performance.isnull().sum()) # have missing values
stud_performance = stud_performance.dropna() # drop missing values

# check for duplicate
print(stud_performance.duplicated().sum()) # no duplicates

# check for inconsistency
print(stud_performance[stud_performance['Exam_Score'] > 100])
print(stud_performance[stud_performance['Attendance'] > 100])
print(stud_performance[stud_performance['Previous_Scores'] > 100])

print(stud_performance[stud_performance['Exam_Score'] < 0])
print(stud_performance[stud_performance['Attendance'] < 0])
print(stud_performance[stud_performance['Previous_Scores'] < 0])

# check for uniqueness
print(stud_performance.nunique())

## Data Integration
## Data Transformation

# turning continuous data to numerical categorical data(equal-width binning)
continuous_features = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Exam_Score']

bin_edges = {}

for column in continuous_features:
    stud_performance[column + '_binned'], bin_edges[column + '_binned'] = pd.cut(stud_performance[column], bins = 5, labels = [0, 1, 2, 3, 4], right = False, retbins=True)

print(bin_edges['Exam_Score_binned'])

stud_performance['Sleep_Hours_binned'] = pd.cut(stud_performance['Sleep_Hours'], bins = [0, 6, 9, float('inf')], labels = [0, 1, 2], right = False)

stud_performance['Tutoring_Sessions_binned'] = pd.cut(stud_performance['Tutoring_Sessions'], bins = [0, 2, 4, 6, 10], labels = [0, 1, 2, 3], right = False)

stud_performance['Physical_Activity_binned'] = pd.cut(stud_performance['Physical_Activity'], bins = [0, 2, 4, 6, 8,], labels = [0, 1, 2, 3], right = False)

for column in continuous_features:
    print(stud_performance[column + '_binned'].unique())
    

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


(stud_performance['Exam_Score_binned'].value_counts()) / len(stud_performance) * 100


# stud_performance.to_csv('modif_student_performance.csv', index = False)
print('modified student performance succesfully loaded into new file')