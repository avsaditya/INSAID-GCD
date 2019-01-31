
# coding: utf-8

# # Model evaluation on Insurance Data - Aditya - GCD - Term 2

# ## Table of Content
# 
# 1. [Problem Statement](#section1)<br>
# 2. [Data Loading and Description](#section2)<br>
# 3. [Exploratory Data Analysis](#section3)<br>
# 4. [Random Forest Classifier](#section4)<br>
# 5. [Model evaluation](#section5)<br>
#     - 5.1 [Model evaluation using accuracy score](#section501)<br>
#     - 5.2 [Model Evaluation using confusion matrix](#section502)<br>
#     - 5.3 [Model evaluation using precision score](#section503)<br>
#     - 5.4 [Model evaluation using recall score](#section504)<br>
#     - 5.5 [Model evaluation using f1_score](#section505)<br>
#     - 5.6 [Model evaluation using ROC_AUC curve](#section506)<br>
#     - 5.7 [Choosing better model using precision score](#section507)<br>

# <a id = section1></a>

# ## 1. Problem Statement
# 
# Given the dataset containing __Life Insurance Data__, use multiple models to predict the response which is a measure of risk in 8 level. Evaluate the model using possible __model evaluation techniques__. 
# 
# Steps to be followed:
# a. Data Cleaning and EDA.
# b. Try out various machine learning models using train test and model evaluation
# techniques and choose the best model for output prediction. You may also use
# RandomisedSearchCV or GridSearchCV for hyperparameter tuning of the
# estimator at your own discretion.

# <a id = section2></a>

# ## 2. Data Loading and Description

# Insurance has become an indispensable part of our lives in recent years and people are paying more attention on it. For this project, the data comes from prudential life insurance on kaggle with over a hundred variables describing attributes of life insurance applicants.. The challenge part for them is that the application process time is antiquated and the goal for this project is to help them to enhance the efficiency of processing time as well as reduce labor intensive for new and existing customers.
# The task is to accurately predict the "Response" variable for each Id in the test set. "Response" is an ordinal measure of risk that has 8 levels.
# 
# The dataset consists of __59K rows__.<br/>
# Below is a table having brief description of features present in the dataset.

# # Data fields
# 
# 
# |Variable   |	Description                                               |
# |---------------------------------| --------------------------------------------------------| 
# | Id	                          |   A unique identifier associated with an application.|
# | Product_Info_1-7                | A set of normalized variables relating to the product applied for  |
# | Ins_Age | Normalized age of applicant|
# | Ht | Normalized height of applicant |
# | Wt | Normalized weight of applicant |
# | BMI | Normalized BMI of applicant |
# | Employment_Info_1-6 | A set of normalized variables relating to the employment history of the applicant. |
# | InsuredInfo_1-6 | A set of normalized variables providing information about the applicant. |
# | Insurance_History_1-9 | A set of normalized variables relating to the insurance history of the applicant. |
# | Family_Hist_1-5 | A set of normalized variables relating to the family history of the applicant. |
# | Medical_History_1-41 | A set of normalized variables relating to the medical history of the applicant. |
# | Medical_Keyword_1-48 | A set of dummy variables relating to the presence of/absence of a medical keyword being associated with the application. |
# | Response	| This is the target variable, an ordinal variable relating to the final decision associated with an application |
# 
# The following variables are all __categorical (nominal)__:
# 
# Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41
# 
# The following variables are __continuous__:
# 
# Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5
# 
# The following variables are __discrete__:
# 
# Medical_History_1, Medical_History_10, Medical_History_15, Medical_History_24, Medical_History_32
# 
# Medical_Keyword_1-48 are __dummy variables__.
# 

# __Importing Packages__

# In[241]:


from collections import Counter

import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# #### Importing the Dataset

# In[242]:


insurance = pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-2/master/Projects/insurance_data.csv')
insurance.head()


# <a id = section3></a>

# ## 3. Exploratory Data Analysis

# #### Check the shape of the dataset

# In[243]:


insurance.shape


# #### Check the columns present in the dataset

# In[244]:


insurance.columns


# #### Check the descriptive statistics of the dataset

# In[245]:


insurance.describe()


# #### Check the info of the dataset

# In[246]:


insurance.info()


# #### Check the missing values present in the dataset. 

# In[247]:


insurance.isnull().sum()[insurance.isnull().sum() !=0]


# In[248]:


missing = insurance.isnull().sum()[insurance.isnull().sum() !=0]
missing = pd.DataFrame(missing.reset_index())
missing.rename(columns={'index':'features',0:'missing_count'}, inplace = True)
missing['missing_count_percentage'] = ((missing['missing_count'])/insurance.shape[0])*100
plt.figure()
sns.barplot(y = missing['features'], x = missing['missing_count_percentage'])


# #### Impute the missing values - Analyze the outliers to choose either Median or Mean

# In[249]:


plt.plot(figsize=(15,10))
sns.boxplot(insurance['Employment_Info_1'])


# In[250]:


# Employment_Info_1 has lots of outliers - Median
insurance['Employment_Info_1'].fillna(insurance['Employment_Info_1'].median(),inplace=True) 


# In[251]:


sns.boxplot(insurance['Employment_Info_4'])


# In[252]:


# Employment_Info_4 has most of the values centered close to zero, huge presence of outliers 
insurance['Employment_Info_4'].fillna(insurance['Employment_Info_4'].median(),inplace=True)


# In[253]:


sns.boxplot(insurance['Employment_Info_6'])


# In[254]:


# Employment_Info_6 has no outliers - Mean should do the job here
insurance['Employment_Info_6'].fillna(insurance['Employment_Info_6'].mean(),inplace=True)


# In[255]:


# Though Insurance_History_5 has high number of missing values, instead of dropping it, let us impute as this is insurance dataset
sns.boxplot(insurance['Insurance_History_5'])


# In[256]:


# Insurance_History_5 also has most of the values centered close to zero, huge presence of outliers 
insurance['Insurance_History_5'].fillna(insurance['Insurance_History_5'].median(),inplace=True)


# In[257]:


sns.boxplot(insurance['Medical_History_1'])


# In[258]:


# Medical_History_1 also has presence of outliers with majority of data from 20 - 150
insurance['Medical_History_1'].fillna(insurance['Medical_History_1'].median(),inplace=True)


# In[259]:


#lets drop 8 columns with very high number of missing values 
insurance.drop(['Medical_History_10','Medical_History_15','Medical_History_24','Medical_History_32','Family_Hist_2','Family_Hist_3','Family_Hist_4','Family_Hist_5'], axis=1, inplace = True)


# In[260]:


insurance.isnull().sum()[insurance.isnull().sum() !=0]


# ### There is one Column which has string type data, let us convert it into numeric data

# In[261]:


insurance['Product_Info_2'].unique()


# In[262]:


le = LabelEncoder()
insurance['Product_Info_2'] = le.fit_transform(insurance['Product_Info_2'])


# In[263]:


insurance.head()


# ## Some actual EDA on the data set to understand more

# In[264]:


CATEGORICAL_COLUMNS = ["Product_Info_1", "Product_Info_2", "Product_Info_3", "Product_Info_5", "Product_Info_6",                       "Product_Info_7", "Employment_Info_2", "Employment_Info_3", "Employment_Info_5", "InsuredInfo_1",                       "InsuredInfo_2", "InsuredInfo_3", "InsuredInfo_4", "InsuredInfo_5", "InsuredInfo_6", "InsuredInfo_7",                       "Insurance_History_1", "Insurance_History_2", "Insurance_History_3", "Insurance_History_4", "Insurance_History_7",                       "Insurance_History_8", "Insurance_History_9", "Family_Hist_1"]

CONTINUOUS_COLUMNS = ["Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI",
                      "Employment_Info_1", "Employment_Info_4", "Employment_Info_6",
                      "Insurance_History_5"]

MEDICAL_COLUMNS = ["Medical_Keyword_{}".format(i) for i in range(1, 48)]


# In[265]:


def plot_categoricals(data):
    ncols = len(data.columns)
    fig = plt.figure(figsize=(5 * 5, 5 * (ncols // 5 + 1)))
    for i, col in enumerate(data.columns):
        cnt = Counter(data[col])
        keys = list(cnt.keys())
        vals = list(cnt.values())
        plt.subplot(ncols // 5 + 1, 5, i + 1)
        plt.bar(range(len(keys)), vals, align="center")
        plt.xticks(range(len(keys)), keys)
        plt.xlabel(col, fontsize=18)
        plt.ylabel("frequency", fontsize=18)
    fig.tight_layout()
    plt.show()

plot_categoricals(insurance[CATEGORICAL_COLUMNS])


# In[266]:


def plot_histgrams(data):
    ncols = len(data.columns)
    fig = plt.figure(figsize=(5 * 5, 5 * (ncols // 5 + 1)))
    for i, col in enumerate(data.columns):
        X = data[col].dropna()
        plt.subplot(ncols // 5 + 1, 5, i + 1)
        plt.hist(X, bins=20, alpha=0.5,                  edgecolor="black", linewidth=2.0)
        plt.xlabel(col, fontsize=18)
        plt.ylabel("frequency", fontsize=18)
    fig.tight_layout()
    plt.show()

plot_histgrams(insurance[CONTINUOUS_COLUMNS])


# In[267]:


plot_categoricals(insurance[MEDICAL_COLUMNS])


# ## Correlation between variables

# In[268]:


axis1 = plt.subplots(1,1,figsize=(10,5))
sns.countplot(x='Response',data=insurance)


# ### Age - Response Correlation

# In[269]:


facet = sns.FacetGrid(insurance, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")
facet.map(sns.kdeplot,'Ins_Age')
facet.set(xlim=(0, insurance['Ins_Age'].max()))
facet.add_legend()


# In[270]:


sns.distplot(insurance["Ins_Age"],bins=10,kde=True)


# ### Employment Info

# In[271]:


fig, axis1 = plt.subplots(1,1,figsize=(15,5))
sns.countplot(x='Employment_Info_2', hue="Response", data=insurance, ax=axis1, hue_order=range(1,9))


# In[272]:


fig, axis1 = plt.subplots(1,1,figsize=(15,5))
sns.countplot(x='Employment_Info_3', hue="Response", data=insurance, ax=axis1, hue_order=range(1,9))


# In[273]:


fig, axis1 = plt.subplots(1,1,figsize=(15,5))
sns.countplot(x='Employment_Info_5', hue="Response", data=insurance, ax=axis1, hue_order=range(1,9))


# ## Ht Wt and BMI

# In[274]:


facet = sns.FacetGrid(insurance, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")
facet.map(sns.kdeplot,'Ht')
facet.set(xlim=(0, insurance['Ht'].max()))
facet.add_legend()


# In[275]:


facet = sns.FacetGrid(insurance, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")
facet.map(sns.kdeplot,'Wt')
facet.set(xlim=(0, insurance['Wt'].max()))
facet.add_legend()


# In[276]:


facet = sns.FacetGrid(insurance, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")
facet.map(sns.kdeplot,'BMI')
facet.set(xlim=(0, insurance['BMI'].max()))
facet.add_legend()


# ### Insurance Info

# In[277]:


fig, axis1 = plt.subplots(1,1,figsize=(20,5))
sns.countplot(x='InsuredInfo_6', hue="Response", data=insurance, 
              ax=axis1, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")


# In[278]:


facet = sns.FacetGrid(insurance, hue="Response",aspect=4, hue_order=[1,2,3,4,5,6,7,8], palette="RdBu")
facet.map(sns.kdeplot,'Insurance_History_5')
facet.set(xlim=(0, 0.004))
facet.add_legend()


# <a id = section4></a>

# ## 4. Models Comparison

# #### Preparing X and y using pandas

# In[279]:


X = insurance.drop(['Response'], axis=1)
X.head()


# In[280]:


y = insurance["Response"]
y.head()


# ####  Spliting X and y into train and test dataset.

# In[281]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)


# #### Checking the shape of X and y of train dataset

# In[282]:


print(X_train.shape)
print(y_train.shape)


# #### Checking the shape of X and y of test dataset

# In[283]:


print(X_test.shape)
print(y_test.shape)


# ### Linear Regression

# In[284]:


linreg = LinearRegression()
linreg = linreg.fit(X_train, y_train) 


# In[285]:


y_pred_test = linreg.predict(X_test)


# In[286]:


MAE_test = metrics.mean_absolute_error(y_test, y_pred_test)
print('MAE for test set is {}'.format(MAE_test))


# In[287]:


MSE_test = metrics.mean_squared_error(y_test, y_pred_test)
print('MSE for test set is {}'.format(MSE_test))


# In[288]:


RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
print('RMSE for test set is {}'.format(RMSE_test))


# In[289]:


yhat = linreg.predict(X_test)
SS_Residual = sum((y_test-yhat)**2)
SS_Total = sum((y_test-np.mean(y_test))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(r_squared, adjusted_r_squared)


# ### Logistic Regression

# In[290]:


logreg = LogisticRegression()
logreg = logreg.fit(X_train, y_train)


# In[291]:


y_pred_train = model.predict(X_train)  
y_pred_test = model.predict(X_test)
print('Accuracy score for test data is:', accuracy_score(y_test,y_pred_test))
print (confusion_matrix(y_test, y_pred_test))


# ### Decision Tree Classifier

# In[292]:


param_grid = {'max_depth':range(1, 20, 2)}
DT = DecisionTreeClassifier()
clf_DT = GridSearchCV(DT, param_grid, cv = 10, scoring='accuracy', n_jobs = -1).fit(X_train,y_train)


# In[293]:


y_pred = clf_DT.predict(X_test)
accuracy_score(y_test,y_pred)


# In[294]:


print('Confusion matrix for test data with DT is:\n',confusion_matrix(y_test, y_pred))


# ### Random Forest Classifier

# #### Instantiating Random Forest Classifier using scikit learn with default parameters.

# In[295]:


model1 = RandomForestClassifier(random_state = 0)


# #### Instantiating Random Forest Classifier using scikit learn with:
# - random_state = 0,
# - max_depth = 5, 
# - min_samples_leaf = 5,
# - min_samples_split = 7,
# - min_weight_fraction_leaf = 0.0,
# - n_estimators = 12, 
# - n_jobs = -1

# In[310]:


model2 = RandomForestClassifier(
                                random_state = 0,
                                max_depth = 8, 
                                min_samples_leaf = 5,
                                min_samples_split = 7,
                                min_weight_fraction_leaf = 0.0,
                                max_features = 'sqrt',
                                n_estimators = 50, 
                                oob_score = True,
                                n_jobs = -1,
                                ) 


# #### Fitting the model on X_train and y_train

# In[311]:


model1.fit(X_train,y_train)


# In[312]:


model2.fit(X_train,y_train)


# #### Using the model for prediction

# In[313]:


prediction1 = pd.DataFrame()
prediction1 = model1.predict(X_test)


# In[314]:


prediction2 = pd.DataFrame()
prediction2 = model2.predict(X_test)


# <a id = section5></a>

# ## 5. Model evaluation 

# <a id = section501></a>

# ### 5.1 Model evaluation using accuracy score

# In[315]:


print('Accuracy score for test data with DT is:',accuracy_score(y_test, y_pred))
print('Accuracy score for test data with model 1 is:',accuracy_score(y_test, prediction1))
print('Accuracy score for test data with model 2 is:',accuracy_score(y_test, prediction2))


# __Accuracy score__ of using Decision Trees is slightly greater than that of model 1 and model 2.<br/>
# Lets see some other evaluation techniques, to compare the two models.

# <a id = section502></a>

# ### 5.2 Model evaluation using confusion matrix

# In[316]:


print ("================================================================")
print ("================================================================")
conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred))
conf_mat.index = ["Actual_{}".format(i) for i in range(1, 9)]
conf_mat.columns = ["Predicted_{}".format(i) for i in range(1, 9)]
print('Confusion matrix for test data with DT is:\n',conf_mat)
conf_mat = pd.DataFrame(confusion_matrix(y_test, prediction1))
print ("================================================================")
print ("================================================================")
conf_mat.index = ["Actual_{}".format(i) for i in range(1, 9)]
conf_mat.columns = ["Predicted_{}".format(i) for i in range(1, 9)]
print('Confusion matrix for test data with model 1 is:\n',conf_mat)
conf_mat = pd.DataFrame(confusion_matrix(y_test, prediction2))
print ("================================================================")
print ("================================================================")
conf_mat.index = ["Actual_{}".format(i) for i in range(1, 9)]
conf_mat.columns = ["Predicted_{}".format(i) for i in range(1, 9)]
print('Confusion matrix for test data with model 2 is:\n',conf_mat)
print ("================================================================")
print ("================================================================")


# The confusion matrix looks little confusing.
# But upon study I found that since this is not a binary data but something that has 8 level prediction; the false positives and true positives are given per response level.
# Comparing confusion matrix for the two models: 
# - In Model 2, there seem to be a less number of False Positives, but when it comes to Risk Assessment at level 8, more tendency is towards that.
# - In Model 1 and DT, it is more spread out with Decision Trees doing a better job
# 
# Calculating Recall and precision score for a clearer picture of the scenario.

# <a id = section503></a>

# ### 5.3. Model evaluation using precision score

# In[317]:


print ("================================================================")
print ("Averaging method is None")
print ("================================================================")
precision = precision_score(y_test,y_pred, average=None)
print('Precision score for test data using DT is:', precision)
precision1 = precision_score(y_test,prediction1, average=None)
print('Precision score for test data using model1 is:', precision1)
precision2 = precision_score(y_test,prediction2, average=None)
print('Precision score for test data using model2 is:', precision2)
print ("================================================================")
print ("Averaging method is weighted")
print ("================================================================")
precision = precision_score(y_test,y_pred, average='weighted')
print('Precision score for test data using DT is:', precision)
precision1 = precision_score(y_test,prediction1, average='weighted')
print('Precision score for test data using model1 is:', precision1)
precision2 = precision_score(y_test,prediction2, average='weighted')
print('Precision score for test data using model2 is:', precision2)


# __Precision score for Decision Trees is little better than that of model1 and model2 __. 

# <a id = section504></a>

# ### 5.4 Model evaluation using recall score

# In[318]:


print ("================================================================")
print ("Averaging method is None")
print ("================================================================")
print('Recall score for test data using DT is:',recall_score(y_test,y_pred, average=None))   
print('Recall score for test data using model1 is:',recall_score(y_test,prediction1, average=None))   
print('Recall score for test data using model2 is:',recall_score(y_test,prediction2, average=None))
print ("================================================================")
print ("Averaging method is weighted")
print ("================================================================")
print('Recall score for test data using DT is:',recall_score(y_test,y_pred, average='weighted'))  
print('Recall score for test data using model1 is:',recall_score(y_test,prediction1, average='weighted'))   
print('Recall score for test data using model2 is:',recall_score(y_test,prediction2, average='weighted'))


# Recall score of DT is more than that of model1 and model2.

# <a id = section505></a>

# ### 5.5 Model evaluation using F1_score

# In[319]:


print ("================================================================")
print ("Averaging method is None")
print ("================================================================")
print('F1_score for test data using DT is:',f1_score(y_test, y_pred, average=None))
print('F1_score for test data using model1 is:',f1_score(y_test, prediction1, average=None))
print('F1_score for test data using model2 is:',f1_score(y_test, prediction2, average=None))
print ("================================================================")
print ("Averaging method is weighted")
print ("================================================================")
print('F1_score for test data using DT is:',f1_score(y_test, y_pred, average='weighted'))
print('F1_score for test data using model1 is:',f1_score(y_test, prediction1, average='weighted'))
print('F1_score for test data using model2 is:',f1_score(y_test, prediction2, average='weighted'))


# F1_score for __DT__ is much __higher__ than that of model 1 and model 2.

# <a id = section506a>

# ### 5.6 Model evaluation using ROC_AUC curve

# In[320]:


def class_report(y_true, y_pred, y_score=None, average='micro'):
    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    #Value counts of predictions
    labels, cnt = np.unique(
        y_pred,
        return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index,
        columns=labels)

    support = class_report_df.loc['support']
    total = support.sum() 
    class_report_df['avg / total'] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df['pred'] = pred_cnt
    class_report_df['pred'].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), 
                y_score[:, label_it])

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == 'micro':
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), 
                    y_score[:, 1].ravel())
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                        lb.transform(y_true).ravel(), 
                        y_score.ravel())

            roc_auc["avg / total"] = auc(
                fpr["avg / total"], 
                tpr["avg / total"])

        elif average == 'macro':
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([
                fpr[i] for i in labels]
            ))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df['AUC'] = pd.Series(roc_auc)

    return class_report_df


# ## Decision Tree Classifier

# In[321]:


report_with_auc = class_report(
    y_true=y_test, 
    y_pred=clf_DT.predict(X_test), 
    y_score=clf_DT.predict_proba(X_test))
print(report_with_auc)


# ## Random Forest - Model 1

# In[322]:


report_with_auc = class_report(
    y_true=y_test, 
    y_pred=model1.predict(X_test), 
    y_score=model1.predict_proba(X_test))
print(report_with_auc)


# ## Random Forest - Model 2

# In[323]:


report_with_auc = class_report(
    y_true=y_test, 
    y_pred=model2.predict(X_test), 
    y_score=model2.predict_proba(X_test))
print(report_with_auc)


# <a id = section5.7></a>

# ### 5.7 Choosing better model using all factors

# We have compared the performance of the models using various model evaluation techinques.<br/>
# Our objective is to __precisely predict Response__ so that life insurance applications can be scrutinized easily. Judging all the factors including AUC, __Decision Tree Clasifier model__ is more suitable for this datset. But __Random Forest Classifier Model 2__ also does a very good job.
