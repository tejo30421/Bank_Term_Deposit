import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification

url='bank-additional-full.csv'
urln='bank-additional-names.txt'

btd = pd.read_csv(url, delimiter = ';')
btd

btd = btd.drop(['day_of_week'], axis=1)

for col in btd.select_dtypes(include='object').columns:
    print(col)
    print(btd[col].unique())

btd1=btd.rename(columns={"y": "subscribe"})
btd1

btd1['subscribe'] = btd1['subscribe'].replace({'no':0,'yes':1})

btd1

features_na = [features for features in btd1.columns if btd1[features].isnull().sum() > 0]
for feature in features_na:
    print(feature, np.round(btd1[feature].isnull().mean(), 4),  ' % missing values')
else:
    print("No missing value found")

categorical_features=[feature for feature in btd1.columns if ((btd1[feature].dtypes=='O') & (feature not in ['subscribe']))]
categorical_features

for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(btd1[feature].unique())))

plt.figure(figsize=(15,80))
plotnumber =1
for categorical_feature in categorical_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.countplot(y=categorical_feature,data=btd1)
    plt.xlabel(categorical_feature)
    plt.title(categorical_feature)
    plotnumber+=1
plt.show()

btd1["default"].isna()

numerical_features = [feature for feature in btd1.columns if ((btd1[feature].dtypes != 'O') & (feature not in ['subscribe']))]
print('Number of numerical variables: ', len(numerical_features))

btd1[numerical_features].head()

discrete_feature=[feature for feature in numerical_features if len(btd1[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))

for categorical_feature in categorical_features:
    print(btd1.groupby(['subscribe',categorical_feature]).size())

continuous_features=[feature for feature in numerical_features if feature not in discrete_feature+['subscribe']]
print("Continuous feature Count {}".format(len(continuous_features)))

plt.figure(figsize=(20,60), facecolor='white')
plotnumber =1
for feature in continuous_features:
    ax = plt.subplot(12,3,plotnumber)
    sns.boxplot(x="subscribe", y= btd1[feature], data=btd1)
    plt.xlabel(feature)
    plotnumber+=1
plt.show()

sns.countplot(x='subscribe',data=btd1)
plt.show()

btd1['subscribe'].groupby(btd1['subscribe']).count()

btd1.groupby(['subscribe','default']).size()

btd1.drop(['default'],axis=1, inplace=True)
btd1.groupby(['subscribe','pdays']).size()

btd1.groupby(['subscribe','duration'],sort=True)['duration'].count()
btd1.groupby(['subscribe','campaign'],sort=True)['campaign'].count()

# excluding outliers
btd2 = btd1[btd1['campaign'] < 33]
btd2.groupby(['subscribe','campaign'],sort=True)['campaign'].count()

cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
for col in  cat_columns:
    btd2 = pd.concat([btd2.drop(col, axis=1),pd.get_dummies(btd2[col], prefix=col, prefix_sep='_',drop_first=True, dummy_na=False)], axis=1)

btd2['housing'] = btd2['housing'].map({'yes': 1, 'no': 0})
btd2['housing']=btd2['housing'].replace({'unknown' :np.NaN})

btd2.replace({True: 1, False: 0})

btd2['loan'] = btd2['loan'].map({'yes': 1, 'no': 0})
btd2['loan']=btd2['loan'].replace({'unknown' :np.NaN})

btd2

btd2.dropna(subset=['housing'], inplace=True)
btd2.dropna(subset=['loan'], inplace=True)

btd2

btd2.isnull().sum().sum()

for col in btd2.select_dtypes(include='float').columns:
    print(col)
    print(btd2[col].unique())

btd2.head().replace({True: 1, False: 0})

X = btd2.drop(['subscribe'],axis=1)
y = btd2['subscribe']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

y.value_counts().plot.pie(autopct='%.2f')

len(X_train)

len(X_test)

# OverSampling
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state=42, sampling_strategy= 0.7)
X_res,y_res=smk.fit_resample(X,y)
custom_colors = ['blue', 'green']
ax = y_res.value_counts().plot.pie(autopct = '%1.5f%%', colors=custom_colors)

X_res.shape,y_res.shape

from collections import Counter
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_res)))

os =  RandomOverSampler(random_state=42)
X_train_res, y_train_res = os.fit_resample(X,y)

X_train_res.shape, y_train_res.shape

y_res.value_counts()

# Slicing the data into 5 parts and showing mean value
from sklearn.model_selection import cross_val_score
model_score =cross_val_score(estimator=RandomForestClassifier(),X=X_train_res, y=y_train_res, cv=5)
print(model_score)
print(model_score.mean())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_res = le.fit_transform(y_train_res)

# slicing for XGBClassifier algorithm
from xgboost import XGBClassifier
model_score =cross_val_score(estimator=XGBClassifier(),X=X_train_res, y=y_train_res, cv=5)
print(model_score)
print(model_score.mean())

# to get best parameters, performing hyperparameter technique for gridsearchcv
model_param = {
    'RandomForestClassifier':{
        'model':RandomForestClassifier(),
        'param':{
            'n_estimators': [10, 50, 100, 130], 
            'criterion': ['gini', 'entropy'],
            'max_depth': range(2, 4, 1), 
            'max_features': ['auto', 'log2']
        }
    },
    'XGBClassifier':{
        'model':XGBClassifier(objective='binary:logistic'),
        'param':{
           'learning_rate': [0.5, 0.1, 0.01, 0.001],
            'max_depth': [3, 5, 10, 20],
            'n_estimators': [10, 50, 100, 200]
        }
    }
}

# gridsearch for best parameters
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42, stratify=y)
rfc = RandomForestClassifier()

param_grid = { 
'max_features': ['auto', 'sqrt', 'log2'],
'max_depth' : [4,5,6,7,8],
'criterion' :['gini', 'entropy']
             }
grid_search_model = GridSearchCV(rfc, param_grid=param_grid)

grid_search_model.fit(x_train, y_train)

print('Best Parameters are:',grid_search_model.best_params_)

from sklearn.metrics import roc_auc_score ,mean_squared_error,accuracy_score,classification_report,roc_curve,confusion_matrix

model = RandomForestClassifier()
model.fit(X_train_res, y_train_res)
y_scores = model.predict(X_res)
auc = roc_auc_score(y_res, y_scores)
# print('Classification Report:')
print(classification_report(y_res,y_scores))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_res, y_scores)
print('ROC_AUC_SCORE is',roc_auc_score(y_res, y_scores))
    
plt.plot(false_positive_rate, true_positive_rate)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()

model_xgb = XGBClassifier(objective='binary:logistic',learning_rate=0.1,max_depth=10,n_estimators=100)

model_xgb.fit(X_train_res,y_train_res)

model_xgb.score(X_res,y_res)

X_train_res, X_res, y_train_res, y_test_res = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_res, y_train_res)
y_pred = rfc.predict(X_res)
rfc_importances = pd.Series(rfc.feature_importances_, index=X.columns).sort_values().tail(10)
rfc_importances.plot(kind='bar')
plt.show()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_res,model_xgb.predict(X_train_res))
cm

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True Value')
plt.show()