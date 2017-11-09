from sklearn.linear_model import LinearRegression
def studentReg(train_set,train_tag):
    reg = LinearRegression()
    reg.fit(train_set,train_tag)

    return reg