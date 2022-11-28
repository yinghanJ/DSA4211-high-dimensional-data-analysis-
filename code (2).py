import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import csv
from math import sqrt
from sklearn.decomposition import KernelPCA, PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, cross_val_score,train_test_split 
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler,PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from mpl_toolkits.mplot3d import Axes3D
from plotnine import *
import warnings
warnings.filterwarnings('ignore')


train_data=pd.read_csv("C:/Users/64242/Desktop/DSAProject/train-xy.csv")
test_data=pd.read_csv("C:/Users/64242/Desktop/DSAProject/test-x.csv")
validation_data=pd.read_csv("C:/Users/64242/Desktop/DSAProject/validationx.csv")

train_data=train_data.fillna(train_data.mean()).values
test_data=test_data.fillna(test_data.mean()).values
validation_data=validation_data.fillna(validation_data.mean()).values
print(train_data.shape)
print(test_data.shape)
print(validation_data.shape)

train_label=train_data[:,0]
train_var=train_data[:,1:]

print(train_var.shape)
print(train_label.shape)

#heatmap
f, ax = plt.subplots(figsize=(100, 60))
corr = train_data.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('x&y Correlation Heatmap', fontsize=14)
plt.savefig('C:/Users/64242/Desktop/DSAProject/x&y Correlation Heatmap.png')

#PCA
pca = PCA(n_components=3)
pca.fit(train_var)
train_var = pca.transform(train_var)
test_data = pca.transform(test_data)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="3d")
ax.scatter(train_var[:,0], train_var[:,1], train_var[:,2])
plt.savefig("fig1.pdf")
plt.show()
tmp_df = pd.DataFrame({"X1": train_var[:,0], "X2": train_var[:,1], "X3": train_var[:,2]})
fig = px.scatter_3d(tmp_df, x="X1", y="X2", z="X3", size_max=18)
fig.show()

#lasso
ridge=Lasso(alpha=0.5)
ridge.fit(train_var,train_label)
pred_test=ridge.predict(test_data)
pred_validation=ridge.predict(validation_data)
print('coefficient: ',ridge.coef_)
print('intercept: ',ridge.intercept_)

pred_test_value=pd.DataFrame(pred_test)
save_pth=r'C:\Users\64242\Desktop\DSAProject\lasso_test-result.csv'
pred_test_value.to_csv(save_pth,header=False,index=False)

pred_validation_value=pd.DataFrame(pred_validation)
save_pth=r'C:\Users\64242\Desktop\DSAProject\lasso_validation-result.csv'
pred_validation_value.to_csv(save_pth,header=False,index=False)

print("mean_absolute_error:", mean_absolute_error(train_label, pred_validation_value))
print("mean_squared_error:", mean_squared_error(train_label, pred_validation_value))
print("rmse:", sqrt(mean_squared_error(train_label, pred_validation_value)))
print("r2 score:", r2_score(train_label, pred_validation_value))

#ridge
ridge=Ridge(alpha=0.5)
ridge.fit(train_var,train_label)
pred_test=ridge.predict(test_data)
pred_validation=ridge.predict(validation_data)
print('coefficient: ',ridge.coef_)
print('intercept: ',ridge.intercept_)

pred_test_value=pd.DataFrame(pred_test)
save_pth=r'C:\Users\64242\Desktop\DSAProject\ridge_test-result.csv'
pred_test_value.to_csv(save_pth,header=False,index=False)

pred_validation_value=pd.DataFrame(pred_validation)
save_pth=r'C:\Users\64242\Desktop\DSAProject\ridge_validation-result.csv'
pred_validation_value.to_csv(save_pth,header=False,index=False)

print("mean_absolute_error:", mean_absolute_error(train_label, pred_validation_value))
print("mean_squared_error:", mean_squared_error(train_label, pred_validation_value))
print("rmse:", sqrt(mean_squared_error(train_label, pred_validation_value)))
print("r2 score:", r2_score(train_label, pred_validation_value))

# elasticnet
ridge=ElasticNet(alpha=0.5)
ridge.fit(train_var,train_label)
pred_test=ridge.predict(test_data)
pred_validation=ridge.predict(validation_data)
print('coefficient: ',ridge.coef_)
print('intercept: ',ridge.intercept_)

pred_test_value=pd.DataFrame(pred_test)
save_pth=r'C:\Users\64242\Desktop\DSAProject\elasticnet_test-result.csv'
pred_test_value.to_csv(save_pth,header=False,index=False)

pred_validation_value=pd.DataFrame(pred_validation)
save_pth=r'C:\Users\64242\Desktop\DSAProject\elasticnet_validation-result.csv'
pred_validation_value.to_csv(save_pth,header=False,index=False)

print("mean_absolute_error:", mean_absolute_error(train_label, pred_validation_value))
print("mean_squared_error:", mean_squared_error(train_label, pred_validation_value))
print("rmse:", sqrt(mean_squared_error(train_label, pred_validation_value)))
print("r2 score:", r2_score(train_label, pred_validation_value))

#svm
svm_reg = LinearSVR(epsilon=1.5)
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
for model in models:
    model.fit(train_var, train_label)
models = [svm_reg, svm_poly_reg]

y_pred1 = model.predict(test_data)

pred_test_value=pd.DataFrame(y_pred1)
save_pth=r'C:\Users\64242\Desktop\DSAProject\svm-test-result.csv'
pred_test_value.to_csv(save_pth,header=False,index=False)
