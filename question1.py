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

df= pd.read_csv("C:/Users/vinil/Downloads/datasets(2)/datasets/CC.csv")
df.head()
print(df.shape)

print(df['TENURE'].value_counts())

A = df.iloc[:,[1,2,3,4]]
B = df.iloc[:,-1]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['CUST_ID'] = le.fit_transform(df.CUST_ID.values)

pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(A)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['TENURE']]], axis = 1)
print(finalDf.head())

from sklearn.cluster import KMeans
nclusters = 2 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(A)

# predict the cluster for each data point
B_cluster_kmeans = km.predict(A)
from sklearn import metrics
score = metrics.silhouette_score(A, B_cluster_kmeans)
print(score)

scaler = StandardScaler()
a_Scale = scaler.fit_transform(A)

pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(a_Scale)

principalDf1 = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf1 = pd.concat([principalDf1, df[['TENURE']]], axis = 1)
finalDf1.head()

from sklearn.cluster import KMeans
nclusters = 2 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(a_Scale)

# predict the cluster for each data point
B_cluster_kmeans = km.predict(a_Scale)
from sklearn import metrics
score = metrics.silhouette_score(a_Scale, B_cluster_kmeans)
print(score)