import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

train = pd.read_csv('data/train_radiomics_hipocamp.csv')
test = pd.read_csv('data/test_radiomics_hipocamp.csv')
control = pd.read_csv('data/train_radiomics_occipital_CONTROL.csv')

pattern = re.compile(r'^diagnostics_')
train = train.loc[:, ~train.columns.str.contains(pattern)]
test = test.loc[:, ~test.columns.str.contains(pattern)]
control = control.loc[:, ~control.columns.str.contains(pattern)]

train.drop(['ID','Mask','Image'], inplace=True, axis=1)
test.drop(['ID','Mask','Image'], inplace=True, axis=1)
control.drop(['ID','Mask','Image'], inplace=True, axis=1)

train['Transition'] = train['Transition'].map({'CN-CN': 0, 'CN-MCI': 1, 'MCI-MCI': 2, 'MCI-AD': 3, 'AD-AD': 4})

test['Transition'] = None
control['Transition'] = None

correlation_threshold = 0.1
correlations = train.corr()
selected_columns = correlations['Transition'][abs(correlations['Transition']) > correlation_threshold].index

train = train[selected_columns]
test = test[selected_columns]
control = control[selected_columns]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train.drop('Transition', axis=1))
test_scaled = scaler.transform(test.drop('Transition', axis=1))
control_scaled = scaler.transform(control.drop('Transition', axis=1))

train_scaled_df = pd.DataFrame(train_scaled, columns=train.columns[:-1])
train_scaled_df['Transition'] = train['Transition'].values

test_scaled_df = pd.DataFrame(test_scaled, columns=test.columns[:-1])
test_scaled_df['Transition'] = test['Transition'].values

control_scaled_df = pd.DataFrame(control_scaled, columns=control.columns[:-1])
control_scaled_df['Transition'] = control['Transition'].values

X_train = train_scaled_df.drop('Transition', axis=1)
y_train = train_scaled_df['Transition']
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

pca = PCA(n_components=10, random_state=42)
X_train_pca = pca.fit_transform(X_train_balanced)
X_test_pca = pca.transform(test_scaled_df.drop('Transition', axis=1))
control_pca = pca.transform(control_scaled_df.drop('Transition', axis=1))

train_balanced_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
train_balanced_pca_df['Transition'] = y_train_balanced

test_pca_df = pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(X_test_pca.shape[1])])
test_pca_df['Transition'] = test_scaled_df['Transition'].values

control_pca_df = pd.DataFrame(control_pca, columns=[f'PC{i+1}' for i in range(control_pca.shape[1])])
control_pca_df['Transition'] = control_scaled_df['Transition'].values

control_pca_df.drop('Transition', axis=1, inplace=True)
test_pca_df.drop('Transition', axis=1, inplace=True)

train_balanced_pca_df.to_csv('data/train_pca.csv', index=False)
test_pca_df.to_csv('data/test_pca.csv', index=False)
control_pca_df.to_csv('data/control_pca.csv', index=False)