#METRICS
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits = 10)

#make dataframe for images, for txt already exist
df = pd.read_csv(r"D:\SPAM DATASET\datasets\text_dataset\spam_fusion_dataset.csv", encoding='latin-1')
df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis=1, inplace=True)
#use axis = 1 when dropping columns

df.columns = ["data","target"]
y = df['data']
X = df['target']


for train_set, test_set in kf.split(X=X,y=y):
    print(train_set)
    print(test_set)
    pipe.fit(X.loc[train_set],y[test_set])
    
    

