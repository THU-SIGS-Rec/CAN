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

### Ali dataset
+ 用户和广告只随机采样了10%
+ 交叉特征时，如果两个特征维度均超过1000个就不交叉
+ Epoch #1 in Test evaluation
  
| Methods       | logloss  | AUC      |
|---------------|----------|----------|
| DNN           | 0.200720 | 0.518092 |
| DNN+Cartesian | 0.343912 | 0.567566 |

+ Epoch #3 in Test evaluation
  
| Methods       | logloss  | AUC      |
|---------------|----------|----------|
| DNN           | 0.232280 | 0.557579 |
| DNN+Cartesian | 0.237509 | 0.554009 |


### Avazu_x4 dataset
+ 删掉了device_id和device_ip，因为这两个太大了没办法交叉
+ Epoch #1 in Test evaluation

| Methods       | logloss  | AUC      |
|---------------|----------|----------|
| DNN           | 0.391069 | 0.759401 |
| DNN+Cartesian | 0.386576 | 0.767494 |

### Avazu_x4_10 dataset
+ 在原来基础抽取1/200的数据，
+ Epoch #1 in Test evaluation

| Methods       | logloss  | AUC      | gAUC     |
|---------------|----------|----------|----------|
| DNN           | 0.417918 | 0.706982 | 0.702964 |
| DNN+Cartesian | 0.405675 | 0.734526 | 0.730774 |

+ Epoch #3 in Test evaluation

| Methods       | logloss  | AUC      | gAUC     |
|---------------|----------|----------|----------|
| DNN           | 0.405261 | 0.734623 | 0.731886 |
| DNN+Cartesian | 0.405675 | 0.734526 | 0.730774 |