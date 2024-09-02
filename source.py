import numpy as np
import pandas as pd
df = pd.read_csv("yunus.csv")
X = df[['intercept', 'i', 'j', 'k']].values
y = df['l'].values
XT = X.T
XTX = XT.dot(X)
XTy = XT.dot(y)
beta = np.linalg.inv(XTX).dot(XTy)
print("Beta values:", beta)
