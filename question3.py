import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv("C:/Users/vinil/Downloads/datasets(2)/datasets/Iris.csv")
print(df.head())

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
A_train_std = stdsc.fit_transform(df.iloc[:,range(0,4)].values)
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
B = class_le.fit_transform(df['Species'].values)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
A_train_lda = lda.fit_transform(A_train_std,B)
data=pd.DataFrame(A_train_lda)
data['class']=B
data.columns=["LD1","LD2","class"]
print(data.head())


markers = ['s', 'x', 'o']
colors = ['r', 'b', 'g']
sns.lmplot(x="LD1", y="LD2", data=data, hue='class', markers=markers, fit_reg=False, legend=False)
plt.legend(loc='upper center')
plt.show()
