#!/usr/bin/python3
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('AdviceRobo_Test_data.csv')
print(df.shape)
df.info()
# there are 24 variables in the dataset and 30000 observations

#Do the data need cleaning?
#are there any missing values?
df.isnull().sum()
#The data are clean, there are no missing values!

# In the data there are no categorical data values, therefore no transformation is necessary.

# Our independent variables are the first columns and the last one is our dependent value, the default or not default column.
# Let's call those X and Y 
Y = df['Y']
X = df.iloc[:, 1:24]

#what is the ration between default and non default?
class_names = {0:'Not Default', 1:'Default'}
print(df.Y.value_counts().rename(index = class_names))
# let's plot it
sns.countplot(df["Y"])
labels = ['Non-Default', 'Default']
plt.title('Number of Default and Non-Default Cases')
#plt.xticks(df["Y"], labels, rotation='vertical')
#plt.xticks(rotation='vertical')
plt.savefig('defaultornot.png')
plt.show()



# Let's see which are the most informative features in the dataset, as those that explain most of the variance:
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k = 23)
#New dataframe with the selected features for later use in the classifier. fit() method works too, if you want only the feature names and their corresponding scores
X_new = selector.fit_transform(X, Y)
names = X.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
#Sort the dataframe for better visualization
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
print(ns_df_sorted)
# Clearly the most influencial parameter to detect a client that has more chances to default is the history of past payment.
# X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is:


# We will test now some supervised machine learning binary classification methods to see which one performs best here 
# Some fitting methods that we will test here are the Logistic Regression, the k Nearest Neighbors, the Random Forests and the Support Vector Machine

#First we split the data into a train set and a test set in order to test both the train and the test accuracy.
# Will be 80% train data 20% test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)

#We need to bring all features to the same level of magnitudes. 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# All the models will be evauated for their accuracy in order to select the best one
# the Confusion Matrix will be used for the model evaluations
from sklearn.metrics import classification_report, confusion_matrix

# Let's start with the first model, the Logistic Regression algorithm
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,Y_train)
print (LR)
Y_pred = LR.predict(X_test)
cmlr = confusion_matrix(Y_test, Y_pred)
print ('Classification report Logistic Regression')
print (classification_report(Y_test, Y_pred))


# k Nearest Neighbors algorithm
#We can calculate the accuracy of KNN for different Ks in order to select the one that provides the most accurate results.
from sklearn import metrics
from  sklearn.neighbors import KNeighborsClassifier
####
Ks = 12
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,Y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(Y_test, yhat)   
    std_acc[n-1]=np.std(yhat==Y_test)/np.sqrt(yhat.shape[0])
   
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(Y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(Y_test, yhat))

#### Plot  model accuracy  for Different number of Neighbors 
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.savefig('bestkNN.png')
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

####
# we run the kNN algorithm with k = 8 and print the confusion matrix
classifier = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)
#predict the test set results and check the accuracy with each of our model:
Y_pred = classifier.predict(X_test)
#test the accuracy with a confusion matrix
from sklearn.metrics import confusion_matrix
cmknn = confusion_matrix(Y_test, Y_pred)
print ('Classification report kNN')
print (classification_report(Y_test, Y_pred))


#Random Forest Classification algorithm

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
cmrf = confusion_matrix(Y_test, Y_pred)
print ('Classification report Random Forest')
print (classification_report(Y_test, Y_pred))

# Support Vector Machine algorithm
from sklearn import svm
classifier = svm.SVC(kernel='rbf')
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
cmsvm = confusion_matrix(Y_test, Y_pred)
print ('Classification report Support Vector Machine')
print (classification_report(Y_test, Y_pred))


# a visual inspection of the confusion matrixes gives relatively similar accuracy between the different methods
# in the precision and recall (false positive and negatives) -non normalized- that follows in the code below. In the code above,
#the harmonic average F1 of the precision and recall (normalized) will gives the overall estimation of the accuracy. 

# It seems that the overall best accuracy comes from the Support Vector Machine algorithm.
# Let's plot the CM to discuss it

#Make a function that plots the confusion matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plot_confusion_matrix(cmknn, classes=['Non-Default','Default'],normalize= False,  title='Confusion matrix - Support Vector Machine')
plt.savefig('cm_SVM.png')
plt.show()
