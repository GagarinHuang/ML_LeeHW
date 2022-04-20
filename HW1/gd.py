import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#model:y = w1x1 + w2x2 + w3x3 + ... + w162 x162 + b
def GD(X, Y, w, lr, iteration):
    list_cost = []
    for i in range(iteration + 1):
        #loss = (Y - Xw)^T * (Y - Xw)
        hypo = np.dot(X,w)
        loss = hypo - Y
        cost = np.dot(loss.T, loss) / (2 * len(X))
        # print('--------------------------cost-------------------------------')
        #print(cost)
        list_cost.append(cost)
        grad = np.dot(X.T, loss) / len(X)
        w = w - lr * grad
        #print('--------------------------grade-------------------------------')
        #print(grad)
        #if i == 1:
            #break
        if i % 10000 == 0:
            print("i: %i, Loss: %f" % (i, cost))
    return w, list_cost

#csv文件读取，前9小时为x，后1小时为y，自变量9*18
traindf = pd.read_csv("./data/train.csv", encoding='big5', engine='python')
#parse data
x = []
y = []

for i in range(240):
    x.append([])

for index, row in traindf.iterrows():
    for col in range(0,9):
        if row[str(col)] != "NR":
            x[index // 18].append(float(row[str(col)]))
        else:
            x[index // 18].append(0)
    if index % 18 == 17:
        x[index // 18].append(1)
    if index % 18 == 9:
        y.append(float(row['9']))
trainX = np.array(x)
trainY = np.array(y)

#initial value
#向量(w;b)
w = np.zeros(163) #18*9个w，1个b
lr = 0.000004 #learning rate
iteration = 100000 #迭代次数

#不断更新(w;b), 计算Loss值
result = GD(trainX, trainY, w, lr, iteration)

#test data
testdf = pd.read_csv("./data/test.csv", encoding='big5', engine='python', header=None)
x = []
y = []
for i in range(240):
    x.append([])
for index, row in testdf.iterrows():
    for i in range(2, 11):
        if (row[i] != "NR"):
            x[index // 18].append(float(row[i]))
        else:
            x[index // 18].append(0)
    if index % 18 == 17:
        x[index // 18].append(1)
testX = np.array(x)
#testY = np.ones(240)
testY = np.dot(testX, result[0])
testY = np.around(testY, decimals=0)

#output to ans.csv
ids = []
for i in range(0,240):
    ids.append(["".join(["id_", str(i)])])
ids = np.array(ids)
testY = testY.reshape(240,1)
testY = np.concatenate((ids, testY),axis=1)
resultdf = pd.DataFrame(testY, columns =["id", "value"])
resultdf.to_csv('./data/result.csv', index=False)

#plot the picture
ansdf = pd.read_csv("./data/ans.csv", encoding='big5', engine='python')
resultdf['answer'] = ansdf['value']
column1 = np.array(resultdf['value'].astype(float)).reshape(240,1)
column2 = np.array(resultdf['answer'].astype(float)).reshape(240,1)
out = np.concatenate((column1, column2),axis=1)
df = pd.DataFrame(out, columns=['Estimate', 'Current'], index=np.arange(0, 240, 1))
df.plot()
plt.show()