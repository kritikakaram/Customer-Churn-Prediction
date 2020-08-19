
# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
color = sns.color_palette()
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier

# Reading the dataset
data = pd.read_csv("C:/Users/KRITIKA/Documents/CS-513 KDD/PROJECT/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data.head(10)
data.columns
len(data.columns)
data.info()

# Converting the column "TotalCharges" from object to float.
data["TotalCharges"]= pd.to_numeric(data["TotalCharges"],errors="coerce")#.fillna(0,downcast="infer")

data.isnull().sum()
data.head()

# Exploratory Data Analysis
# Droping the column "customerID"
data.drop("customerID",axis=1,inplace =True)

# Converting the column "TotalCharges" from object to float.
data["TotalCharges"]= pd.to_numeric(data["TotalCharges"],errors="coerce")#.fillna(0,downcast="infer")

#Finding the median of the column TotalCharges
a = data["TotalCharges"].median()
# replacing na values in TotalCharges with Median
data["TotalCharges"].fillna(a, inplace = True) 

# Replacing "No internet service" and "No phone service" with "No"
data.replace("No internet service","No",inplace = True)
data["MultipleLines"].replace("No phone service","No",inplace = True)

# Binning the Tenure column
data.tenure.value_counts()
data["tenure"]= pd.cut(data["tenure"],bins=5)
tenure_churn=pd.crosstab(data["tenure"],data["Churn"]).apply(lambda x: (x/x.sum()*100),axis=1)
tenure_churn.plot.bar()

# Gender
gender_count= data.gender.value_counts()
sns.barplot(gender_count.index, gender_count.values)

# Relation between Churn and Gender
gender_churn= pd.crosstab(data["gender"],data["Churn"])
gender_churn.plot.bar()

# Now to get relation between each feature and the Target variable "Churn"
a=data.columns
a=list(a)
a.pop()  # Removing the churn column
for i in (a):
    if data[i].dtype == "object" or data[i].dtype=="int64":
        df=pd.crosstab(data[i],data["Churn"]).apply(lambda x: (x/x.sum()*100),axis=1)
        print(i, "vs Churn")
        print(df)
        df.plot.bar()
        plt.show()
        print("----------------------")
        print("----------------------")
        print("----------------------")


# Churn  
churn_count= data.Churn.value_counts()
sns.barplot(churn_count.index,churn_count.values)

# Separating the features and target columns
x= data.iloc[:,:-1].values
y=data.iloc[:,-1].values
col= data.columns
col=col.drop("Churn")

# Encoding the categorical data
from sklearn.preprocessing import LabelEncoder
LE_x= LabelEncoder()
for i in range(len(col)-1):
    x[:,i]=LE_x.fit_transform(x[:,i])
LE_y= LabelEncoder()
y=LE_y.fit_transform(y)


z=pd.DataFrame(x,columns=col)
# OneHot Encoding the required columns
X=pd.get_dummies(z ,columns=["tenure","InternetService","Contract","PaymentMethod"])    
X["Churn"]=pd.DataFrame(y)
Y=X.iloc[:,-1]
X.drop("Churn",axis=1,inplace = True)

# Spliting the data into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.3)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train.iloc[:,13:15]= sc.fit_transform(X_train.iloc[:,13:15])
X_test.iloc[:,13:15]= sc.transform(X_test.iloc[:,13:15])

# Dropping the column "TotalCharges" as it is higly correlated with "Monthly Charges" and "Tenure"
X_train.drop("MonthlyCharges",axis=1,inplace= True)
X_test.drop("MonthlyCharges",axis=1,inplace= True)



#############################################################33
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()


#GRID SEARCH

n_estimators = [100, 300, 500]
max_depth = [4, 5,7]
min_samples_split = [2, 5, 6]
min_samples_leaf = [1, 2, 3] 
criterion = ['gini','entropy']


from sklearn.model_selection import GridSearchCV
hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf, criterion=criterion)

gridF = GridSearchCV(model, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(X_train, Y_train)

best_param= gridF.best_params_


forestOpt = RandomForestClassifier(random_state = 1,criterion='entropy', max_depth = 7, class_weight='balanced',    n_estimators = 100, min_samples_split = 6, min_samples_leaf = 2)
                                   
modelOpt = forestOpt.fit(X_train, Y_train)
y_pred = modelOpt.predict(X_test)

print(modelOpt.feature_importances_)

y_pred_prob =modelOpt.predict_proba(X_test)[:, 1]



from sklearn.metrics import roc_auc_score

# Calculate roc auc
roc_value = roc_auc_score(Y_test, y_pred_prob)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

cm = (confusion_matrix(Y_test,y_pred))
cm_list=cm.tolist()
cm_list[0].insert(0,'Real True')
cm_list[1].insert(0,'Real False')
#plot_confusion_matrix(cm)



print(classification_report(Y_test,y_pred))
print(accuracy_score(Y_test, y_pred))

#ROC
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(Y_test, y_pred_prob)
plot_roc_curve(fpr, tpr)


#matplotlib inline
#do code to support model
#"data" is the X dataframe and model is the SKlearn object

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(data.columns, modelOpt.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Feature-importance'})
importances.sort_values(by='Feature-importance').plot(kind='bar', rot=85)
