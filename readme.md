### Tiny dataset
+ Epoch #1 in Test evaluation
  
| Methods       | logloss  | AUC      |
|---------------|----------|----------|
| DNN           | 0.688668 | 0.825521 |
| DNN+Cartesian | 0.686849 | 0.927083 |

+ Epoch #3 in Test evaluation

| Methods       | logloss  | AUC      |
|---------------|----------|----------|
| DNN           | 0.676594 | 1.000000 |
| DNN+Cartesian | 0.675608 | 1.000000 |

### Tiny dataset
+ 用户和广告只随机采样了10%
+ 交叉特征时，如果两个特征维度均超过1000个就不交叉
+ Epoch #1 in Test evaluation
  
| Methods       | logloss  | AUC      |
|---------------|----------|----------|
| DNN           | 0.343912 | 0.567566 |
| DNN+Cartesian | 0.200720 | 0.518092 |

+ Epoch #3 in Test evaluation
  
| Methods       | logloss  | AUC      |
|---------------|----------|----------|
| DNN           | 0.237509 | 0.554009 |
| DNN+Cartesian | 0.232280 | 0.557579 |
