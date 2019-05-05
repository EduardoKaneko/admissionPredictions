#%% [markdown]
# # <font color=#005ce8> Predicting Graduate Admissions </font>
# <br>
# <p> Autor: Eduardo Kaneko </p>
# <p> Date: 23.02.2019 </p>
#%% [markdown]
# ## Problem Description
# ______
# <p> This dataset is created for prediction of graduate admissions from an Indian perspective. </p>
# 
# *A computer program is said to learn from **experience E** with respect to some class of **tasks T** and **performance measure P**, if its performance at tasks in T, as measured by P, improves with experience E.*
# <p> 
# 
# **Task (T)**: Predict the student's chance of being approved or not. <br> <br>
# **Experience (E)**: A bunch of students with some parameters where one of them is a chance of admit (ranging from 0 to 1).<br> <br>
# **Performance (P)**: Classification accuracy, the number of students approved correctly out of all students considered as a percentage.
# </p>
#%% [markdown]
# ## Content Description
# ______
# <p> The dataset contains several parameters which are considered important during the application for Masters Programs. The parameters included are : <br>  
#     1. GRE Scores ( out of 340 ) <br>
#     2. TOEFL Scores ( out of 120 ) <br>
#     3. University Rating ( out of 5 ) <br>
#     4. Statement of Purpose and Letter of Recommendation Strength ( out of 5 ) <br>
#     5. Undergraduate GPA ( out of 10 ) <br>
#     6. Research Experience ( either 0 or 1 ) <br> 
#     7. Chance of Admit ( ranging from 0 to 1 ). </p>
#%% [markdown]
# ## Acknowledgements
# ______
# <p> This dataset is inspired by the UCLA Graduate Dataset. The test scores and GPA are in the older format. The dataset is owned by Mohan S Acharya. </p>
#%% [markdown]
# ## Inspiration
# ______
# <p> This dataset was built with the purpose of helping students in shortlisting universities with their profiles. The predicted output gives them a fair idea about their chances for a particular university.</p>
# 
# <p>Besides that, I am solving the problem as a learning exercise. It's my first machine learning project by myself. So, I don't want use the most suitable method to solve the problem, but, instead, I want to explore methods that I'm not familiar with and I want to improve my knoweldge with methods that I already know in order to increase and learn new skills. </p>
#%% [markdown]
# ## Citations
# ______
# <p> Mohan S Acharya, Asfia Armaan, Aneeta S Antony : A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019</p>
#%% [markdown]
# ## <font> Content: </font>
# 
# * [1. Importing the libraries](#import)
# * [2. Loading and Assessing the data](#load)
# * [3. Statistical Looking](#sl)
# * [4. Splitting Data for Training and Testing](#split)
# * [5. Linear Regression](#lr)
# * [6. Statistical Information with StatsModel](#stats)
# * [7. Logistic Regression Model](#logmodel)
# * [8. Support Vector Machine](#svm)
# * [9. Decision Tree](#dt)
# * [10.KNN Model](#kn)
# * [11.Comparision Between Models](#comp)
#%% [markdown]
# <h3><a id="import" class="anchor"><font color=#005ce8>1 - Importing the libraries</font></a></h3>

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#%% [markdown]
# <h3><a id="load" class="anchor"><font color=#005ce8>2 - Loading and Assessing the data</font></a></h3>

#%%
df= pd.read_csv('Admission_Predict.csv')


#%%
df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})


#%%
df.head()


#%%
# Looking to the data
print("Rows: {}".format(df.shape[0]))
print("Columns: {}".format(df.shape[1]))


#%%
df.info()

#%% [markdown]
# *There is no problem of completeness.*

#%%
df.describe()

#%% [markdown]
# <h3><a id="sl" class="anchor"><font color=#005ce8>3 - Statistical Looking</font></a></h3>
#%% [markdown]
# **(3.1) Check if there are linear correlations between the parameters**

#%%
sns.pairplot(df);

#%% [markdown]
# Seens that GRE_SCORE; TOEFL_SCORE and CGPA have high correlation with 'Chance of Admit', let's take a nearest look.

#%%
g = sns.pairplot(df, x_vars=["GRE Score", "TOEFL Score", "CGPA"],
                     y_vars=["Chance of Admit"])

#%% [markdown]
# We can observe a *strong strength* and a *positive direction* between `GRE Score`, `TOEFL Score` and `CGPA` with `Chance of Admit`.

#%%
plt.figure(figsize=(16,8))
sns.heatmap(df.corr(), annot=True);

#%% [markdown]
# The term "correlation" refers to a mutual relationship or association between quantities. As the correlation coefficient `r` of `GRE`, `TOEFL` and `CGPA` with `Chance of Admit` is 0.8, 0.79 and 0.87. We can conclude that there are a truly positive and strong correlation between the parameters and `Chance of Admit`.
#%% [markdown]
# **(3.2) Distribution of Parameters:**

#%%
plt.figure(figsize=(15,12))
plt.subplot(2, 2, 1)
sns.distplot(df['GRE Score'], color='Pink')
plt.title("GRE Distribution")
plt.grid(alpha=0.5)


plt.subplot(2, 2, 2)
sns.distplot(df['TOEFL Score'], color='Grey')
plt.title("TOEFL Distribution")
plt.grid(alpha=0.5)


plt.subplot(2, 2, 3)
sns.distplot(df['CGPA'], color='Green')
plt.title("CGPA Distribution")
plt.grid(alpha=0.5)


plt.subplot(2, 2, 4)
sns.distplot(df['University Rating'], color='Orange')
plt.title("University Rating Distribution")
plt.grid(alpha=0.5)

#%% [markdown]
# **(3.3) Analysis on Research Column:**

#%%
plt.figure(figsize=(8,4))
sns.countplot(y=df['Research'], palette="muted")
plt.grid(alpha=0.8)
plt.title("Counting of Research")
plt.xlabel('Students')
plt.show()


#%%
researchers = (df['Research'] == 1).sum()
non_researchers = (df['Research'] == 0).sum()
research_percentage = (researchers/len(df)*100)

print("Number of Researchers: {}".format(researchers))
print("Number of Non-researchers: {}".format(non_researchers))
print("Percentage of students with research: {}%".format(research_percentage))

#%% [markdown]
# **(3.4) University Rating:**

#%%
plt.figure(figsize=(8,4))
plt.title("Counting of University Rating")
sns.countplot(y=df['University Rating'], palette="muted")
plt.grid(alpha=0.8)
plt.xlabel('Students')
plt.show()


#%%
plt.figure(figsize=(10,6))
university_influence = df[df["Chance of Admit"] >= 0.75]["University Rating"].value_counts()
university_influence.plot(kind='barh')
plt.title("University Rating of 75% being approved chance")
plt.grid(alpha=0.5)
plt.xlabel('Student Count')
plt.ylabel('University Rating')
plt.show()


#%%
rating_one = ((university_influence.iloc[4]/((df['University Rating']==1).sum()))*100)
rating_two = ((university_influence.iloc[3]/((df['University Rating']==2).sum()))*100)
rating_three = ((university_influence.iloc[2]/((df['University Rating']==3).sum()))*100)
rating_four = ((university_influence.iloc[0]/((df['University Rating']==4).sum()))*100)
rating_five = ((university_influence.iloc[1]/((df['University Rating']==5).sum()))*100)


#%%
print("Percentage of >75% being approved by university rating")
print('University Rating 1: {0: .2f}%'.format(rating_one))
print('University Rating 2: {0: .2f}%'.format(rating_two))
print('University Rating 3: {0: .2f}%'.format(rating_three))
print('University Rating 4: {0: .2f}%'.format(rating_four))
print('University Rating 5: {0: .2f}%'.format(rating_five))

#%% [markdown]
# **(3.5) Mean and Standard Deviation of `GRE`, `TOEFL` and `CGPA`:** <br> <br>
# Standard Deviation is a statistical term used to measure the amount of variability or dispersion around an average. Technically it is a measure of volatility. Dispersion is the difference between the actual and the average value. The larger this dispersion or variability is, the higher is the standard deviation.

#%%
gre_avg = df['GRE Score'].mean()
gre_std = df['GRE Score'].std()
print("Maximum GRE Score : ", np.max(df['GRE Score']))
print("Average GRE Score : ",gre_avg)
print("Standard Deaviation : ",gre_std)


#%%
diff = df['GRE Score']-gre_avg
df['SD_GRE'] = diff/gre_std


#%%
toefl_avg = df['TOEFL Score'].mean()
toefl_std = df['TOEFL Score'].std()
print("Maximum TOEFL Score : ", np.max(df['TOEFL Score']))
print("Average TOEFL Score : ",toefl_avg)
print("Standard Deaviation : ",toefl_std)


#%%
diff = df['TOEFL Score']-toefl_avg
df['SD_TOEFL'] = diff/toefl_std


#%%
cgpa_avg = df['CGPA'].mean()
cgpa_std = df['CGPA'].std()
print("Maximum CGPA Score : ", np.max(df['CGPA']))
print("Average CGPA Score : ",cgpa_avg)
print("Standard Deaviation : ",cgpa_std)


#%%
diff = df['CGPA']-cgpa_avg
df['SD_CGPA'] = diff/cgpa_std


#%%
df.head()


#%%
# Plotting new parameters against Chance of Admit
sns.pairplot(df, 
             x_vars=['GRE Score','TOEFL Score','CGPA','SD_GRE','SD_TOEFL','SD_CGPA'], 
             y_vars='Chance of Admit');


#%%
# Constructing Heatmap of Corelation for new parameters against 'Chance of Admit':
plt.figure(figsize=(16,8))
sns.heatmap(df.corr(), annot=True);

#%% [markdown]
# <h3><a id="split" class="anchor"><font color=#005ce8>4 - Splitting Data for Training and Testing</font></a></h3>

#%%
X = df.drop(['Chance of Admit'], axis=1)
y = df['Chance of Admit']


#%%
# Importing module and splitting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    test_size = 0.2,
                                                    random_state = 42)

#%% [markdown]
# <h3><a id="lr" class="anchor"><font color=#005ce8>5 - Linear Regression</font></a></h3>
#%% [markdown]
# **(5.1) Importing the library and fitting the model:**

#%%
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

#%% [markdown]
# **(5.2) The Co-Efficients for the following parameters:**

#%%
coef = pd.DataFrame(lr.coef_, X_test.columns, columns = ['Co-efficiente'])
coef

#%% [markdown]
# From above we can infer that :
# 
# - If GRE Score increases by 1 then Chance of Admit will be affected by 0.001943 <br>
# - If TOEFL increases by 1 then Chance of Admit will be affected by 0.003455 <br>
# - If University Rating increases by 1 then Chance of Admit will be affected by 0.010820 <br>

#%%
pred_lr = lr.predict(X_test)


#%%
len(X_test)

#%% [markdown]
# **(5.3) Plotting Actual VS Predicted Values:**

#%%
fig = plt.figure(figsize=(16,8))
c = [i for i in range(1,81,1)]
plt.plot(c, y_test, color = 'orange', linewidth = 2.5, label='Test')
plt.plot(c, pred_lr, color = 'purple', linewidth = 2.5, label='Predicted')
plt.grid(alpha = 0.5)
plt.legend()
fig.suptitle('Actual vs Predicted');

#%% [markdown]
# **(5.4) Calculating Error Terms:**

#%%
from sklearn.metrics import mean_squared_error, r2_score


#%%
mse = mean_squared_error(y_test, pred_lr)
r_squared_score = r2_score(y_test, pred_lr)


#%%
print('Mean Square Error = {0: .5f}'.format(mse))
print('R_Square Score = {0: .5f}'.format(r_squared_score))


#%%
fig = plt.figure(figsize=(12,6))
plt.plot(c,y_test-pred_lr, color = 'orange', linewidth = 2.5)
plt.grid(alpha = 0.5)
fig.suptitle('Error Terms')

#%% [markdown]
# <h3><a id="stats" class="anchor"><font color=#005ce8>6 - Statistical Information with StatsModel</font></a></h3>

#%%
import statsmodels.api as sm


#%%
X_train_sm = X_train
X_train_sm = sm.add_constant(X_train_sm)
lml = sm.OLS(y_train, X_train_sm).fit()
lml.params


#%%
print(lml.summary())

#%% [markdown]
# ### Inferences
# ______
# <p> We can infer:
# 
# - For each unit increase on `GRE Score`, as long as all the other variables stay the same, the `Chance of Admit` will increase by **0.0006**. <br>
# - For each unit increase on `TOEFL Score`, as long as all the other variables stay the same, the `Chance of Admit` will increase by **0.0019**. <br> 
# - For each unit increase on `University Rating`, as long as all the other variables stay the same, the `Chance of Admit` will increase by **0.0108**. <br> 
# - For each unit increase on `CGPA Score`, as long as all the other variables stay the same, the `Chance of Admit` will increase by **0.0223**. </p>
# 
# <p> Re-Valuating the Data:
# 
# - If 'p > 0.05' for a 95% level of confidence:
# 
#     - {$Ho$} : Value is not significant
#     - {$H1$} : Value is significant Since in GRE p(0.065) > 0.05 so 'we fail to reject Ho' </p>

#%%
X_new = df.drop(['Serial No.','University Rating','SOP','Chance of Admit'], axis=1)
y_new = df['Chance of Admit']


#%%
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, 
                                                                    y_new, 
                                                                    train_size = 0.7, 
                                                                    random_state = 42)


#%%
lr.fit(X_train_new, y_train_new)


#%%
y_pred_new = lr.predict(X_test_new)


#%%
len(X_test_new)


#%%
# Actual vs Predicted after removing GRE
fig = plt.figure(figsize=(16,6))
c = [i for i in range(1,121,1)]
plt.plot(c, y_test_new, color = 'orange', linewidth = 2.5, label='Test')
plt.plot(c, y_pred_new, color = 'purple', linewidth = 2.5, label='Predicted')
plt.grid(alpha = 0.3)
plt.legend()
fig.suptitle('Actual vs Predicted');


#%%
mse_new = mean_squared_error(y_test_new, y_pred_new)
r_square_score_new = r2_score(y_test_new, y_pred_new)
print('Mean Square Error = ',mse_new)
print('R_Square Score = ',r_square_score_new)


#%%
fig = plt.figure(figsize=(16,6))
plt.plot(c,y_test_new-y_pred_new, color = 'orange', linewidth = 2.5)
plt.grid(alpha = 0.3)
fig.suptitle('Error Terms');

#%% [markdown]
# <h3><a id="logmodel" class="anchor"><font color=#005ce8>7 - Logistic Regression Model</font></a></h3>
#%% [markdown]
# **(7.1) Importing the model:**

#%%
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

#%% [markdown]
# **Observations**
#%% [markdown]
# Since Logistic Regression is a Classification model, 'Continuous Data' will not help to classify the output. Hence we need to categorize the data into:
# - Label 1 for Chance of Admit greater or equal to 80% <br>
# - Label 0 for Chance of Admit lesser than 80%

#%%
y_train_label = [1 if each > 0.8 else 0 for each in y_train]
y_test_label  = [1 if each > 0.8 else 0 for each in y_test]


#%%
logmodel.fit(X_train, y_train_label)


#%%
pred_log = logmodel.predict(X_test)


#%%
from sklearn.metrics import classification_report
print(classification_report(y_test_label, pred_log))


#%%
from sklearn.metrics import confusion_matrix
cm_log = confusion_matrix(y_test_label, pred_log)


#%%
plt.figure(figsize=(8, 5))
sns.heatmap(cm_log, annot=True)
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.ylabel("Actual");

#%% [markdown]
# **(7.2) Looking for the main metrics:**

#%%
y_pred = logmodel.predict(X_test)


#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = (accuracy_score(y_test_label, y_pred)*100)

precision = (precision_score(y_test_label, y_pred)*100)

recall = (recall_score(y_test_label, y_pred)*100)

f1 = (f1_score(y_test_label, y_pred)*100)

print("Accuracy:{0: .2f}%".format(accuracy))
print("Precision:{0: .2f}%".format(precision))
print("Recall:{0: .2f}%".format(recall))
print("F1:{0: .2f}%".format(f1))

#%% [markdown]
# **(7.3) Explanation about the metrics used:**
#%% [markdown]
# #### Accuracy Score:
# <p>Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations. One may think that, if we have high accuracy then our model is best. In ou case, the accuracy is <b>0.9625.</b> <br>
# $(TP+TN)/(TP+FP+FN+TN)$ </p>
# 
# #### Precision Score:
# <p>Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. In our case, the precision is <b>0.9643.</b> <br>
# $(TP)/(TP+FP)$
# </p>
# 
# #### Recall Score:
# <p>(Sensitivity) Recall is the ratio of correctly predicted positive observations to the all observations in actual class.In our case, the recall is <b>0.9310.</b><br>
# $TP/(TP+FN)$
# </p>
# 
# #### F1 Score:
# <p>F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. In our case, F1 Score is <b>0.9474.</b> <br>
# $2*(Recall * Precision) / (Recall + Precision)$
# </p>
# 
# 
#  
#%% [markdown]
# <h3><a id="svm" class="anchor"><font color=#005ce8>8 - Support Vector Machine</font></a></h3>

#%%
from sklearn.svm import SVC
svc = SVC()


#%%
svc.fit(X_train, y_train_label)


#%%
pred_svm = svc.predict(X_test)


#%%
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test_label, pred_svm)


#%%
plt.figure(figsize=(8,5))
sns.heatmap(cm_svm, annot=True)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual");


#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = (accuracy_score(y_test_label, pred_svm)*100)

precision = (precision_score(y_test_label, pred_svm)*100)

recall = (recall_score(y_test_label, pred_svm)*100)

f1 = (f1_score(y_test_label, pred_svm)*100)

print("Accuracy:{0: .2f}%".format(accuracy))
print("Precision:{0: .2f}%".format(precision))
print("Recall:{0: .2f}%".format(recall))
print("F1:{0: .2f}%".format(f1))

#%% [markdown]
# <h3><a id="dt" class="anchor"><font color=#005ce8>9 - Decision Tree</font></a></h3>

#%%
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X_train, y_train_label)


#%%
pred_dt = dt_reg.predict(X_test)


#%%
from sklearn.metrics import confusion_matrix
cm_dt = confusion_matrix(y_test_label, pred_dt)


#%%
plt.figure(figsize=(8,5))
sns.heatmap(cm_dt, annot=True)
plt.title("DecisionTree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual");


#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = (accuracy_score(y_test_label, pred_dt)*100)

precision = (precision_score(y_test_label, pred_dt)*100)

recall = (recall_score(y_test_label, pred_dt)*100)

f1 = (f1_score(y_test_label, pred_dt)*100)

print("Accuracy:{0: .2f}%".format(accuracy))
print("Precision:{0: .2f}%".format(precision))
print("Recall:{0: .2f}%".format(recall))
print("F1:{0: .2f}%".format(f1))

#%% [markdown]
# <h3><a id="kn" class="anchor"><font color=#005ce8>10 - KNN Model</font></a></h3>

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


#%%
import math
math.sqrt(len(y_test_label))


#%%
knnclf = KNeighborsClassifier(n_neighbors=10, p=2, metric='euclidean')


#%%
knnclf.fit(X_train, y_train_label)


#%%
pred_knn = knnclf.predict(X_test)


#%%
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test_label, pred_knn)


#%%
plt.figure(figsize=(8,5))
sns.heatmap(cm_knn, annot=True)
plt.title("KNeighbors Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual");


#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = (accuracy_score(y_test_label, pred_knn)*100)

precision = (precision_score(y_test_label, pred_knn)*100)

recall = (recall_score(y_test_label, pred_knn)*100)

f1 = (f1_score(y_test_label, pred_knn)*100)

print("Accuracy:{0: .2f}%".format(accuracy))
print("Precision:{0: .2f}%".format(precision))
print("Recall:{0: .2f}%".format(recall))
print("F1:{0: .2f}%".format(f1))

#%% [markdown]
# <h3><a id="comp" class="anchor"><font color=#005ce8>11 - Comparision Between Models</font></a></h3>

#%%
x = ["KNN","SVM","Logistic_Reg"]
y = np.array([accuracy_score(y_test_label, pred_knn),accuracy_score(y_test_label, pred_svm),accuracy_score(y_test_label, pred_log)])
plt.barh(x,y, color='#225b46')
plt.xlabel("Accuracy Score")
plt.ylabel("Classification Models")
plt.title("Best Accuracy Score")
plt.grid(alpha=0.5)
plt.show()


#%%



