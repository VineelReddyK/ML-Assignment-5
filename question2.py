from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC, LinearSVC
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

df= pd.read_csv("C:/Users/vinil/Downloads/datasets(2)/datasets/pd_speech_features.csv")
print(df.head())

print(df.shape)

print(df['class'].value_counts())


A = df.drop('class',axis=1).values
B = df['class'].values
scaler = StandardScaler()
A_Scale = scaler.fit_transform(A)
pca2 = PCA(n_components=3)
principalComponents = pca2.fit_transform(A_Scale)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, df[['class']]], axis = 1)
print(finalDf.head())

A_train, A_test, B_train, B_test = train_test_split(A_Scale,B, test_size=0.3,random_state=0)
svc = SVC(max_iter=1000)
svc.fit(A_train, B_train)
B_pred = svc.predict(A_test)
acc_svc = round(svc.score(A_train, B_train) * 100, 2)
print("svm accuracy =", acc_svc)
