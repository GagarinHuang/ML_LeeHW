[**HomeWork Description**](https://docs.google.com/presentation/d/1r2u-xVytctdRSbaCAHwWlHIBkmJ50Stnpj1hqi9pFXs/edit?usp=sharing)   


## Predicting PM2.5  
model: linear regression
根据xi:前9个小时的18种空气成分值
预测y:第10个小时的PM2.5

## Data   
- train.csv:每个月前20天每个小时的气象资料（每小时有18种测资）共12个月
- test.csv: 排除train.csv剩余的资料 取连续9小时的资料当作feature 预测第10小时的PM2.5值 总共240笔
- sampleSubmission.py:结果predict.csv的格式
- ans.csv:test的answer  

## Result
predict.csv:模型预测结果
Result_Figure.png:实际值与预估值对比图

使用simple GD方式，极限learning rate ≈ 0.000004, iteration = 100000次内无法得到最小loss值
