---
layout: single
permalink: /model/
title: "Modeling the Data"
---

# Pipelining and Applying Algorithms

## Importing Libraries


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel

```


```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
```

## Custom Classes


```python
class FilterNAs(BaseEstimator, TransformerMixin):
    def __init__(self, drop_thresh = .5):
        self.d_per = drop_thresh
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        new_thresh = self.d_per * X.shape[0]
        X.dropna(thresh=new_thresh, inplace=True, axis = 1)
        return X
    
class DropTextCol(BaseEstimator, TransformerMixin):
    def __init__(self, drop_thresh = .2):
        self.d_per = drop_thresh
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        df_obj = X.select_dtypes(include = ['object'])
        too_many = df_obj.apply(lambda x: (x.nunique() / df.shape[0])).sort_values(ascending=False)
        too_many = too_many[too_many >= self.d_per].index.values.tolist()
        X.drop(too_many, axis=1, inplace = True)
        return X    

# Copied from Machine Learning Project Checklist from the book Hands-On Machine Learning with Scikit-Learn
# & TensorFlow by Aurélien Géron. Page 67.

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, strings = False):
        self.attribute_names = attribute_names
        self.strings = strings
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if(self.strings):
            return X[self.attribute_names].values.astype('str')
        return X[self.attribute_names].values
    
# class ToNumpyArray(BaseEstimator, TransformerMixin):      
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X, y=None):
#         return np.c_[X]
#     def names(self, X, y=None):
#         return X.columns.tolist()   
```

## Scikit Learn CategoricalEncoder
*This Scikit Learn Class is only in the github dev version and will be implemented at a later date in the standard version.*

*Link to code: https://github.com/scikit-learn/scikit-learn/blob/47ce5e1/sklearn/preprocessing/data.py#L2871. Idea taken from 02_end_to_end_machine_learning_project. Part of the Machine Learning and Deep Learning in python using Scikit-Learn and TensorFlow jupyter notebooks.*


```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        if self.categories != 'auto':
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet "
                                     "supported")

        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                if self.handle_unknown == 'error':
                    valid_mask = np.in1d(Xi, self.categories[i])
                    if not np.all(valid_mask):
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(self.categories[i])

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            Xi = X[:, i]
            valid_mask = np.in1d(Xi, self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    Xi = Xi.copy()
                    Xi[~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(Xi)

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        feature_indices = np.cumsum(n_values)

        indices = (X_int + feature_indices[:-1]).ravel()[mask]
        indptr = X_mask.sum(axis=1).cumsum()
        indptr = np.insert(indptr, 0, 0)
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csr_matrix((data, indices, indptr),
                                shape=(n_samples, feature_indices[-1]),
                                dtype=self.dtype)
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

    def inverse_transform(self, X):
        check_is_fitted(self, 'categories_')
        X = check_array(X, accept_sparse='csr')

        n_samples, _ = X.shape
        n_features = len(self.categories_)
        n_transformed_features = sum([len(cats) for cats in self.categories_])

        # validate shape of passed X
        msg = ("Shape of the passed X data is not correct. Expected {0} "
               "columns, got {1}.")
        if self.encoding == 'ordinal' and X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))
        elif (self.encoding.startswith('onehot')
                and X.shape[1] != n_transformed_features):
            raise ValueError(msg.format(n_transformed_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        if self.encoding == 'ordinal':
            for i in range(n_features):
                labels = X[:, i].astype('int64')
                X_tr[:, i] = self.categories_[i][labels]

        else:  # encoding == 'onehot' / 'onehot-dense'
            j = 0
            found_unknown = {}

            for i in range(n_features):
                n_categories = len(self.categories_[i])
                sub = X[:, j:j + n_categories]

                # for sparse X argmax returns 2D matrix, ensure 1D array
                labels = np.asarray(_argmax(sub, axis=1)).flatten()
                X_tr[:, i] = self.categories_[i][labels]

                if self.handle_unknown == 'ignore':
                    # ignored unknown categories: we have a row of all zero's
                    unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                    if unknown.any():
                        found_unknown[i] = unknown

                j += n_categories

            # if ignored are found: potentially need to upcast result to
            # insert None values
            if found_unknown:
                if X_tr.dtype != object:
                    X_tr = X_tr.astype(object)

                for idx, mask in found_unknown.items():
                    X_tr[mask, idx] = None

        return X_tr
```

## Reading in Data


```python
df = pd.read_csv('LendingClub2012to2013.csv', skiprows=1, low_memory=False)
df = df.dropna(subset = ['loan_status'])
df = df.dropna(how='all', axis = 0)
df_org = df.copy()


Y = df['loan_status']
X = df.drop('loan_status',axis=1)
```

## Pipelines


```python
# Custom DataFrame pipeline

df_pipline = Pipeline([
    ('FilterNAs', FilterNAs()),
    ('DropingText', DropTextCol()),    
])
```


```python
X = df_pipline.fit_transform(X)

text_attribs = X.select_dtypes(include = ['object']).columns.tolist()
num_attribs = X.select_dtypes(include = ['float64']).columns.tolist()
```


```python
# Feature Union transformation pipeline

text_pipeline = Pipeline([
('selector', DataFrameSelector(text_attribs, strings = True)),
('categorical_encoder', CategoricalEncoder()), 
('imputer', Imputer(strategy="median"))
])

num_pipeline = Pipeline([
('selector', DataFrameSelector(num_attribs)),
('imputer', Imputer(strategy="median")),
('robust_scaler', RobustScaler()) # Due to the presence of numerous outliers, the robust scaler more adequate than standard
# ('var_threshold', VarianceThreshold(threshold=(.9 * (1 - .9))))  # Got rid of this because it actually lowers overall precision and recall.
])

double_pipeline = FeatureUnion(transformer_list=[
("num_pipeline", num_pipeline),
("text_pipeline", text_pipeline)
])
```


```python
X = double_pipeline.fit_transform(X)
X = X.toarray()
```

## Testing/ Train Split

Hold-out is 20% of the data, Validation is 10% of the remaining data, Training is 90% of the remaining data.


```python
# Generating the final test set:
X_train, X_test_final_test, Y_train, Y_final_test = train_test_split(X,Y, test_size = .2)

# Generating the intermediate set: 
X_train, X_test, Y_train, Y_test = train_test_split(X_train,Y_train, test_size = .1)

print("Final test size: " + str(Y_final_test.shape[0]))
print("Intermediate test size: " + str(Y_test.shape[0]))
print("Training size: " + str(Y_train.shape[0]))
```

    Final test size: 37637
    Intermediate test size: 15055
    Training size: 135489
    

# Code used to generate models and results
*Computer intensive due to the fact that there are hundreds of variables.* 

*At the bottom are machine learning models that are much less computer intensive that peformed the final model testing.*

## Fine tuning the machine learning models
*The one of the models that perform well is the Random Forest. Under this  model, there is already very high precision and recall already for many classes of the target variable. Through tuning the models and perhaps modifying the target variables, we may be able to use this model for feature selection.*


```python
ml_algo = [GaussianNB(), DecisionTreeClassifier(),
            RandomForestClassifier(), LogisticRegression()]
```


```python
for algo in ml_algo:
    Model = algo
    Model = Model.fit(X_train,Y_train)
    predict = Model.predict(X_test)
    
    print("============================",str(algo),"============================")
    print("Cross Validation: ", cross_val_score(Model, X_train,Y_train,cv=5), '\n')
    print("Classification Report: ",'\n', classification_report(predict,Y_test), '\n')
    print("Accuary Score: ",accuracy_score(predict,Y_test), '\n')
    print("Confusion Matrix:",'\n',  confusion_matrix(predict, Y_test), '\n')
    print("=========================================================================================", '\n')
```

    ============================ GaussianNB(priors=None) ============================
    Cross Validation:  [0.71821268 0.70867159 0.70432151 0.72991327 0.71378483] 
    
    Classification Report:  
                         precision    recall  f1-score   support
    
           Charged Off       0.89      0.90      0.90      2273
               Current       0.13      0.93      0.23       174
               Default       1.00      0.00      0.00      1864
            Fully Paid       0.75      1.00      0.86      8606
       In Grace Period       0.57      0.02      0.04       620
     Late (16-30 days)       0.19      0.00      0.01      1039
    Late (31-120 days)       0.70      0.10      0.18       479
    
           avg / total       0.75      0.72      0.64     15055
     
    
    Accuary Score:  0.7198937230156094 
    
    Confusion Matrix: 
     [[2045    0    0  228    0    0    0]
     [   0  161    0   12    0    1    0]
     [  82    1    2 1773    0    1    5]
     [  40    1    0 8565    0    0    0]
     [  38  369    0  180   13   10   10]
     [  55  571    0  397    7    3    6]
     [  25  133    0  268    3    1   49]] 
    
    ========================================================================================= 
    
    ============================ DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best') ============================
    Cross Validation:  [0.99081249 0.99143911 0.99095841 0.99077321 0.99128991] 
    
    Classification Report:  
                         precision    recall  f1-score   support
    
           Charged Off       0.99      0.99      0.99      2293
               Current       0.97      0.98      0.98      1226
               Default       0.50      0.50      0.50         2
            Fully Paid       1.00      1.00      1.00     11415
       In Grace Period       0.17      0.12      0.14        34
     Late (16-30 days)       0.06      0.12      0.08         8
    Late (31-120 days)       0.86      0.78      0.82        77
    
           avg / total       0.99      0.99      0.99     15055
     
    
    Accuary Score:  0.9914314181335104 
    
    Confusion Matrix: 
     [[ 2264     0     0    29     0     0     0]
     [    0  1202     0     0    16     4     4]
     [    0     0     1     0     0     0     1]
     [   21     0     0 11394     0     0     0]
     [    0    20     0     0     4     7     3]
     [    0     5     0     0     0     1     2]
     [    0     9     1     0     3     4    60]] 
    
    ========================================================================================= 
    
    ============================ RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False) ============================
    Cross Validation:  [0.98723341 0.98896679 0.98656678 0.98571692 0.98759919] 
    
    Classification Report:  
                         precision    recall  f1-score   support
    
           Charged Off       0.97      1.00      0.98      2212
               Current       0.99      0.94      0.97      1297
               Default       0.00      0.00      0.00         0
            Fully Paid       1.00      0.99      1.00     11520
       In Grace Period       0.00      0.00      0.00         2
     Late (16-30 days)       0.00      0.00      0.00         2
    Late (31-120 days)       0.31      1.00      0.48        22
    
           avg / total       0.99      0.99      0.99     15055
     
    
    Accuary Score:  0.987910993025573 
    
    Confusion Matrix: 
     [[ 2207     0     0     4     0     0     1]
     [    0  1225     1     0    22    10    39]
     [    0     0     0     0     0     0     0]
     [   78    10     1 11419     1     6     5]
     [    0     0     0     0     0     0     2]
     [    0     1     0     0     0     0     1]
     [    0     0     0     0     0     0    22]] 
    
    ========================================================================================= 
    
    

    C:\Users\jakes\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\metrics\classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
      'recall', 'true', average, warn_for)
    

    ============================ LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False) ============================
    Cross Validation:  [0.99357981 0.99287823 0.99143817 0.99151135 0.96969921] 
    
    Classification Report:  
                         precision    recall  f1-score   support
    
           Charged Off       0.98      1.00      0.99      2249
               Current       0.99      0.93      0.96      1311
               Default       0.00      0.00      0.00         0
            Fully Paid       1.00      1.00      1.00     11464
       In Grace Period       0.00      0.00      0.00         6
     Late (16-30 days)       0.00      0.00      0.00         2
    Late (31-120 days)       0.27      0.83      0.41        23
    
           avg / total       0.99      0.99      0.99     15055
     
    
    Accuary Score:  0.9899701095981401 
    
    Confusion Matrix: 
     [[ 2243     0     0     6     0     0     0]
     [    0  1225     2     0    23    11    50]
     [    0     0     0     0     0     0     0]
     [   42     5     0 11417     0     0     0]
     [    0     3     0     0     0     3     0]
     [    0     1     0     0     0     0     1]
     [    0     2     0     0     0     2    19]] 
    
    ========================================================================================= 
    
    

    C:\Users\jakes\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\metrics\classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
      'recall', 'true', average, warn_for)
    

## Grid Search


```python
param_grid = {
    'n_estimators': [15, 23, 30, 45, 60, 75],
     'max_features': ['auto' , None, .5,.7],
     'min_samples_leaf' : [2,5,10,50,120],
     'max_depth': [None, 10, 20,50,100],
     'n_jobs' : [-1]
    }

final_grid = {
    'n_estimators': [25],
     'max_features': [.5,None,'sqrt'],
     'min_samples_leaf' : [2,10,20],
     'max_depth': [None,30],
     'n_jobs' : [-1]
    }


best_grid = {
    'n_estimators': [25],
     'max_features': [None],
     'min_samples_leaf' : [10],
     'max_depth': [None],
     'n_jobs' : [-1]
    }


```


```python
forest = RandomForestClassifier()
grid_search = GridSearchCV(forest, final_grid, cv=5)
grid_search.fit(X_test,Y_test)
```

## Saving Model


```python
joblib.dump(grid_search.best_estimator_, "final_model.pkl")
```

## Loading the Best Model


```python
final_model = joblib.load("final_model.pkl")
final_model
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=10, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=25, n_jobs=-1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)



### *Getting the variable importance*
*Through this code we can get the variable importance of our decision tree.*

*Basic idea taken from Hands-On Machine Learning page 74 and https://stackoverflow.com/questions/36633460/how-to-get-feature-names-selected-by-feature-elimination-in-sklearn-pipeline.*


```python
# Getting the names of the columns

text_col_names = list(np.hstack((text_pipeline.named_steps['categorical_encoder'].categories_)))
num_col_names = list(np.asarray(num_attribs))
attributes = np.concatenate([num_col_names,text_col_names])
```


```python
# Making a dataframe of importance

model = final_model

feature_importance = zip(attributes.tolist(),model.feature_importances_.tolist())
feature_importance = np.asarray(list(feature_importance))
feature_importance = np.sort(feature_importance, axis = 0)[::-1]
feature_importance = pd.DataFrame(feature_importance, columns = ['Feature','Tree_Importance_Score'])
feature_importance.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Tree_Importance_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>wedding</td>
      <td>9.907714074093709e-06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>w</td>
      <td>9.723394957926462e-07</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vacation</td>
      <td>9.619829870131448e-07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>total_rev_hi_lim</td>
      <td>9.564526420689163e-06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>total_rec_prncp</td>
      <td>9.520262367635957e-07</td>
    </tr>
    <tr>
      <th>5</th>
      <td>total_rec_late_fee</td>
      <td>9.491119386912804e-07</td>
    </tr>
    <tr>
      <th>6</th>
      <td>total_rec_int</td>
      <td>9.43946480146061e-06</td>
    </tr>
    <tr>
      <th>7</th>
      <td>total_pymnt_inv</td>
      <td>9.386789949662516e-06</td>
    </tr>
    <tr>
      <th>8</th>
      <td>total_pymnt</td>
      <td>9.266861532305998e-06</td>
    </tr>
    <tr>
      <th>9</th>
      <td>total_il_high_credit_limit</td>
      <td>9.12268594982573e-07</td>
    </tr>
    <tr>
      <th>10</th>
      <td>total_bc_limit</td>
      <td>8.860984618665859e-07</td>
    </tr>
    <tr>
      <th>11</th>
      <td>total_bal_ex_mort</td>
      <td>8.665862926011948e-06</td>
    </tr>
    <tr>
      <th>12</th>
      <td>total_acc</td>
      <td>8.576324971072865e-07</td>
    </tr>
    <tr>
      <th>13</th>
      <td>tot_hi_cred_lim</td>
      <td>8.56190836110275e-07</td>
    </tr>
    <tr>
      <th>14</th>
      <td>tot_cur_bal</td>
      <td>8.55569842128399e-05</td>
    </tr>
    <tr>
      <th>15</th>
      <td>tot_coll_amt</td>
      <td>8.426843274140549e-07</td>
    </tr>
    <tr>
      <th>16</th>
      <td>tax_liens</td>
      <td>8.411840850625117e-07</td>
    </tr>
    <tr>
      <th>17</th>
      <td>small_business</td>
      <td>8.401010104036136e-07</td>
    </tr>
    <tr>
      <th>18</th>
      <td>revol_bal</td>
      <td>8.241690403417078e-06</td>
    </tr>
    <tr>
      <th>19</th>
      <td>renewable_energy</td>
      <td>8.218242241209607e-06</td>
    </tr>
  </tbody>
</table>
</div>



### Saving to excel


```python
df = pd.DataFrame(text_col_names, index = text_attribs)
filepath = 'my_excel_file.xlsx'
df.set_index(text_attribs)
df.to_excel(filepath)
```

# Applying the Final Modeling

### Feature Selection 

The tuned Random Forest is in charge of picking the variables.


```python
print(X_train.shape)
model = SelectFromModel(final_model, prefit=True)
X_train = model.transform(X_train)
X_test_final_test = model.transform(X_test_final_test)
print(X_train.shape)
```

    (135489, 3010)
    (135489, 20)
    


```python
ml_algo = [GaussianNB(), DecisionTreeClassifier(),
            RandomForestClassifier(), LogisticRegression()]
```


```python
for algo in ml_algo:
    Model = algo
    Model = Model.fit(X_train,Y_train)
    predict = Model.predict(X_test_final_test)
    
    print("============================",str(algo),"============================")
#     print("Cross Validation: ", cross_val_score(Model, X_train,Y_train,cv=10), '\n')
    print("Classification Report: ",'\n', classification_report(predict,Y_final_test), '\n')
#     print("Accuary Score: ",accuracy_score(predict, Y_final_test), '\n')
    print("Confusion Matrix:",'\n',  confusion_matrix(predict, Y_final_test), '\n')
    print("=========================================================================================", '\n')
```

    ============================ GaussianNB(priors=None) ============================
    Classification Report:  
                         precision    recall  f1-score   support
    
           Charged Off       0.91      0.96      0.93      5247
               Current       0.97      0.98      0.97      3062
               Default       1.00      0.18      0.31        22
            Fully Paid       0.99      0.98      0.99     29025
       In Grace Period       0.28      0.19      0.23       108
     Late (16-30 days)       0.32      0.18      0.23        51
    Late (31-120 days)       0.59      0.65      0.61       122
    
           avg / total       0.97      0.97      0.97     37637
     
    
    Confusion Matrix: 
     [[ 5037     0     0   210     0     0     0]
     [    0  3000     0     4    39     7    12]
     [    0     0     4     4     0     0    14]
     [  493     2     0 28530     0     0     0]
     [    1    43     0     6    21    10    27]
     [    0    18     0     7    14     9     3]
     [    5    30     0     5     1     2    79]] 
    
    ========================================================================================= 
    
    ============================ DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best') ============================
    Classification Report:  
                         precision    recall  f1-score   support
    
           Charged Off       0.99      0.99      0.99      5514
               Current       0.97      0.98      0.98      3080
               Default       1.00      0.44      0.62         9
            Fully Paid       1.00      1.00      1.00     28789
       In Grace Period       0.23      0.19      0.21        88
     Late (16-30 days)       0.21      0.24      0.23        25
    Late (31-120 days)       0.76      0.78      0.77       132
    
           avg / total       0.99      0.99      0.99     37637
     
    
    Confusion Matrix: 
     [[ 5484     0     0    30     0     0     0]
     [    0  3011     0     0    41     8    20]
     [    0     0     4     0     0     0     5]
     [   52     1     0 28736     0     0     0]
     [    0    56     0     0    17     8     7]
     [    0     8     0     0    11     6     0]
     [    0    17     0     0     6     6   103]] 
    
    ========================================================================================= 
    
    ============================ RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False) ============================
    Classification Report:  
                         precision    recall  f1-score   support
    
           Charged Off       0.99      1.00      1.00      5502
               Current       1.00      0.98      0.99      3160
               Default       0.75      0.60      0.67         5
            Fully Paid       1.00      1.00      1.00     28801
       In Grace Period       0.17      0.42      0.25        31
     Late (16-30 days)       0.25      0.28      0.26        25
    Late (31-120 days)       0.78      0.93      0.85       113
    
           avg / total       1.00      1.00      1.00     37637
     
    
    Confusion Matrix: 
     [[ 5496     0     0     6     0     0     0]
     [    0  3083     0     0    47     9    21]
     [    0     1     3     0     0     0     1]
     [   40     1     0 28760     0     0     0]
     [    0     4     0     0    13    10     4]
     [    0     3     0     0    11     7     4]
     [    0     1     1     0     4     2   105]] 
    
    ========================================================================================= 
    
    ============================ LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False) ============================
    Classification Report:  
                         precision    recall  f1-score   support
    
           Charged Off       0.96      1.00      0.98      5312
               Current       0.99      0.98      0.98      3145
               Default       0.00      0.00      0.00         0
            Fully Paid       1.00      0.99      1.00     29005
       In Grace Period       0.24      0.40      0.30        45
     Late (16-30 days)       0.07      0.12      0.09        17
    Late (31-120 days)       0.76      0.91      0.83       113
    
           avg / total       0.99      0.99      0.99     37637
     
    
    Confusion Matrix: 
     [[ 5297     0     0    15     0     0     0]
     [    0  3072     0     3    41     8    21]
     [    0     0     0     0     0     0     0]
     [  239    14     2 28748     0     1     1]
     [    0     5     0     0    18    14     8]
     [    0     2     0     0    11     2     2]
     [    0     0     2     0     5     3   103]] 
    
    ========================================================================================= 
    
    

    C:\Users\jakes\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\metrics\classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
      'recall', 'true', average, warn_for)
    
