## playground-series-s3e12
## Kidney Stone Prediction based on Urine Analysis基于尿液分析的肾结石预测
特征描述
* gravity,比重，尿液相对于水的密度；
* ph,pH，氢离子的负对数；
* osmo,渗透压(mOsm)，生物学和医学中使用但物理化学中不使用的单位,渗透压与溶液中分子的浓度成正比
* cond,电导率(mMho milliMho)。1 Mho 是 1欧姆的倒数。电导率与溶液中带电离子的浓度成正比
* urea,尿素浓度，单位为毫摩尔每升；
* calc,钙浓度 (CALC)，单位为毫摩尔升。


### MLP(x5)
dropout=0.2,lr=0.001
* 200 epoch , Loss 0.1436, score 0.7493
* 400 epoch , Loss 0.0671, score 0.8293
* 600 epoch , Loss 0.0402, score 0.8053
* 800 epoch , Loss 0.0303, score 0.7960
###加入残差 dropout=0.2,lr=0.001 RELU
* 400 epoch , Loss 0.1053
* 800 epoch , Loss 0.0510, score 0.8173
* 1200 epoch ,Loss 0.0509, score 0.8333
###加入残差 dropout=0.2,lr=0.0001 RELU
* 2000 epoch ,Loss 0.0025, score 0.8533
### 注意力机制 heads=2,dropout=0.2,lr=0.0001 RELU
* 2000 epoch ,Loss 0.0033, score 0.8653
* ### 注意力机制 heads=4,dropout=0.2,lr=0.0001 RELU
* 3000 epoch ,Loss*1000 0.00000008, score 0.7480
* 1000 epoch ,Loss*1000 0.00015600, score 0.7501