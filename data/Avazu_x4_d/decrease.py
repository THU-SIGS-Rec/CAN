import pandas

if __name__ == "__main__":
    train = pandas.read_csv('../Avazu_x4/train.csv')
    train = train.sample(train.shape[0]//200)
    train.to_csv("train.csv")
    
    valid = pandas.read_csv('../Avazu_x4/valid.csv')
    valid = valid.sample(valid.shape[0]//200)
    valid.to_csv("valid.csv")
    
    test = pandas.read_csv('../Avazu_x4/test.csv')
    test = test.sample(test.shape[0]//200)
    test.to_csv("test.csv")