#import libraries and read in the csv file
import warnings
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

df = pd.read_csv("/Users/rachaelkondrat/vhdl/ProjectData/weatherAUS.csv")
print("Amount of Data Points and Variables",df.shape)

df.head()

#print a summary of the data
print("A Summary of the Data:" )
df = df.dropna()
df.describe()

#scale the data

from sklearn import preprocessing

numerical = [var for var in df.columns if df[var].dtype == "float64"]
for col in numerical:
    df[col] =preprocessing.scale(df[col])
df.head()

#convert all data into positive numbers
#scale all of the numbers

z=np.abs(df._get_numeric_data())

#keep all values under 3 times the standard deviation to removethe 
#outliers
df = df[(z<3).all(axis=1)]

print(z)
df.shape

#replace all no's with 0 and yes's with 1

df['RainToday'].replace({'No' : 0, 'Yes': 1},inplace=True)
df['RainTomorrow'].replace({'No' : 0, 'Yes': 1},inplace=True)

X = df[["MaxTemp", ]]
y = df[["Rainfall"]]

plt.rcParams.update({'font.size': 35})

fig, ax = plt.subplots(figsize=(30, 25))
plt.scatter(X,y, color = 'darkblue', alpha = 0.3)

ax.set(title = "Rainfall Likelihood in Relation to the Maimum Tempuerature")
plt.xlabel("MaxTemp")
plt.ylabel("Rainfall")

X = df[["Evaporation", ]]
y = df[["Rainfall"]]

plt.rcParams.update({'font.size': 35})

fig, ax = plt.subplots(figsize=(30, 25))
plt.scatter(X,y, color = 'darkblue', alpha = 0.3)

ax.set(title = "Rainfall Likelihood in Relation to Evaporation Rates")
plt.xlabel("Evaporation")
plt.ylabel("Rainfall")

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111,projection= "3d")
x1 = df["MaxTemp"]
x2 = df["Evaporation"]

plt.rcParams.update({'font.size': 20})


ax.scatter(y,x1,x2, c="r", marker="o")

ax.set(title = "Rainfall Likelihood in Relation to Max Tempuratures and Evaporation Rates")
ax.set_xlabel("Rainfall")
ax.set_ylabel("MaxTemp")
ax.set_zlabel("Evaporation")

#set up ML training variables
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

regr = linear_model.Ridge(alpha = 0.5)

X =df.loc[:, df.columns != "RainTomorrow"]

y = df.RainTomorrow

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =34)

logReg = LogisticRegression()
logReg.fit(X_train,y_train)

#evaluate the model

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean

#set random to avoid bias

#useKFold: Provides train/test indices to split data in train/test sets
cv = KFold(n_splits = 5, random_state =1, shuffle = True )

#use cross_val_scores to Evaluate a score by cross-validation

scores = cross_val_score(logReg, X, y, scoring = "accuracy", cv = cv)
average_score = mean(scores)

print("overall score", average_score)

# this will display the predictions based on the previous data if it will rain 
#the upcoming day
y_pred=logReg.predict(X_test[0:50])
y_pred

#ask for the users input for what day they want to predict
val = input("Please enter a value 0 through 40 to see if it will flood the next upcoming day: ")
print("You selected day: " , val)
input_int = int(val)

if y_pred[input_int] == 0:
    print("It will not flood tomorrow")
if y_pred[input_int] == 1:
    print("It will flood tomorrow")
