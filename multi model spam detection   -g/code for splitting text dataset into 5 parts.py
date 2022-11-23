
import pandas as pd

# csvfile = open(r"D:\dev\projects\SPAM research\DATASETS\text_dataset\spam_fusion_dataset.csv", 'r').readlines()
# k = 1
# for i in range(len(csvfile)):
#      filename = r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset" + str(k)
#      if i % 1115 == 0:
#          open(str(filename) + '.csv', 'w+').writelines(csvfile[i:i+1000])
#          k += 1
         
         
         
df = pd.read_csv(r"D:\dev\projects\SPAM research\DATASETS\text_dataset\spam_fusion_dataset.csv", encoding='latin-1')

df1 = df.iloc[:1115,:]
df2 = df.iloc[1115:2230,:]
df3 = df.iloc[2230:3345,:]
df4 = df.iloc[3345:4460,:]
df5 = df.iloc[4460:5574,:]

df1.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\file1.csv",header=False,index=False)
df2.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\file2.csv",header=False,index=False)
df3.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\file3.csv",header=False,index=False)
df4.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\file4.csv",header=False,index=False)
df5.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\file5.csv",header=False,index=False)


fold1_train = pd.concat([df2,df3,df4,df5], ignore_index=True)
fold1_test = df1

fold2_train = pd.concat([df1,df3,df4,df5], ignore_index=True)
fold2_test = df2

fold3_train = pd.concat([df2,df1,df4,df5], ignore_index=True)
fold3_test = df3

fold4_train = pd.concat([df2,df3,df1,df5], ignore_index=True)
fold4_test = df4

fold5_train = pd.concat([df2,df3,df4,df1], ignore_index=True)
fold5_test = df5


fold1_train.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold1\fold1_train.csv",header=False,index=False)
fold1_test.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold1\fold1_test.csv",header=False,index=False)
fold2_train.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold2\fold2_train.csv",header=False,index=False)
fold2_test.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold2\fold2_test.csv",header=False,index=False)
fold3_train.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold3\fold3_train.csv",header=False,index=False)
fold3_test.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold3\fold3_test.csv",header=False,index=False)
fold4_train.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold4\fold4_train.csv",header=False,index=False)
fold4_test.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold4\fold4_test.csv",header=False,index=False)
fold5_train.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold5\fold5_train.csv",header=False,index=False)
fold5_test.to_csv(r"D:\dev\projects\SPAM research\DATASETS\5 fold text dataset\fold5\fold5_test.csv",header=False,index=False)






