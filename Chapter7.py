# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # 一般化線形混合モデル（GLMM）
# ## 一般化線形混合モデルで説明できない時

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy import stats
from motofunctions import iofunctions
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns

# %% [markdown]
# ## INPUT 

# %%

# 全カラムを表示させる
pd.set_option('display.max_columns', None)
df_inp = pd.read_csv("data7.csv")
print(df_inp)

fig=plt.figure(figsize=(6,6),dpi=120)
fig.patch.set_facecolor('white')
ax=fig.add_subplot(111)
ax = sns.swarmplot(x='x',y='y',data=df_inp[["x","y"]] )
ax.set(xlabel='x',ylabel='y')
ax.set_axisbelow(True)
ax.grid(axis='y')
plt.show()

# %%
# 著者サイトから3ファイル(data4a.csvとdata4b.csv, )をダウンロード
#response = requests.get('http://hosho.ees.hokudai.ac.jp/~kubo/stat/iwanamibook/fig/glmm/data.csv')
#with open ('data7.csv', 'wb') as f:
    #f.write(response.content)
    #f.close()


# %%
result = smf.glm(formula="y+I(N-y)~x", data=df_inp, family=sm.families.Binomial()).fit()
result.summary()


# %%
from scipy.stats import binom
#
y=np.arange(9)
logistic  = lambda beta1, beta2, x:1/(1+math.exp(-(beta1+beta2*x)))

fig=plt.figure()
def plotfig():
    i=[1]
    def inner():
        q= logistic(result.params["Intercept"], result.params["x"],i[0])
        y=range(9)

        num_individuals=df_inp[df_inp["x"]==i[0]]
        num_y=[sum(k==y[j] for k in num_individuals["y"]) for j in y]

        ax1=fig.add_subplot(111)
        ln1=ax1.plot(y,num_y, label="actually")
        plt.xlabel("y")
        plt.ylabel("number of individuals")

        ax2=ax1.twinx()
        ln2=ax2.plot(y, [binom.pmf(y_i,8,q) for y_i in y], label="predicted",color="orange")
        plt.ylabel("pmf")
        h1,l1=ax1.get_legend_handles_labels()
        h2,l2=ax2.get_legend_handles_labels()
        ax1.legend(h1+h2,l1+l2,loc="upper right")
        i[0]+=1
        return ln1
    return inner

plo=plotfig()
lns=[]

for i in range(9):
    lns.append(plo())

#%%
import matplotlib.animation as animation
ani = animation.ArtistAnimation(fig, lns, interval=100)
plt.show()

print(lns)