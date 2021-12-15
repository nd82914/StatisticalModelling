# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## 準備
# 著者サイトで公開されているデータを使います。
# 
# http://hosho.ees.hokudai.ac.jp/~kubo/ce/IwanamiBook.html
# %%
import zipfile
import math
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3
pd.set_option('display.max_columns', None)
#%%
data_file = "./chap9.csv"
df_inp = pd.read_csv(data_file)
df_inp

# %%
fig=plt.figure()
ax1=fig.add_subplot(111)
sc=ax1.scatter(x="x",y="y",data=df_inp[["x","y"]])
# %% [markdown]
# ## 9.1 例題：種子数のポアソン回帰(個体差なし)
# %%
# statsmodelsを使って最尤推定でパラメータを求める
import statsmodels.formula.api as smf

# scipy 1.0.0にはstats.chisqprobが無いため、statsmodels 0.8.0ではfitでエラーとなる。
# そのため、scipy.stats.chisqprobを再定義している。
# cf. https://github.com/statsmodels/statsmodels/issues/3931
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
data=df_inp[["x","y"]]
result = smf.poisson('y ~ x', data=data).fit()
result.summary()

# %% [markdown]
# ## 9.3 無情報事前分布

# %%
# N(0, 1)とN(0, 100)の確率密度を比較
from scipy.stats import norm

x = np.arange(-10.00, 10.01, 0.01)

plt.plot(x, [norm.pdf(x_i, 0, 1) for x_i in x], label='N(0, 1)')
plt.plot(x, [norm.pdf(x_i, 0, 100) for x_i in x], label='N(0, 100)')
plt.legend()
plt.xlabel('beta_1, beta_2')
plt.ylabel('p(beta_*)')
plt.show()

# %% [markdown]
# ## 9.4 ベイズ統計モデルの事後分布の推定



# %%
# np.float128ではエラーとなったのでfloat64に型変換
data['x'] = data['x'].astype(np.float64)
data['y'] = data['y'].astype(np.float64)


# %%
with pymc3.Model() as model:
    # 事前分布をN(0, 100)の正規分布で設定
    beta1 = pymc3.Normal('beta1', mu=0, sd=100)
    beta2 = pymc3.Normal('beta2', mu=0, sd=100)
    
    # 線形予測子θをβ1+β2xで設定
    theta = beta1 + beta2*data['x'].values
    
    # ログリンク関数(log(μ)=θ⇔μ=exp(θ))を設定し、ポアソン分布で推定する
    y = pymc3.Poisson('y', mu=np.exp(theta), observed=data['y'].values)


# %%
# ハミルトニアンモンテカルロ法
with model:
    # 101個目から3個置きでサンプルを取得するチェインを3つ作る
    trace = pymc3.sample(1500, step=pymc3.HamiltonianMC(), tune=100, njobs=3, random_seed=0)[::3]
    
_ = pymc3.traceplot(trace)

pymc3.summary(trace)


# %%
print('Trace type:', type(trace)) # Trace type: <class 'pymc3.backends.base.MultiTrace'>
print('Trace length:', len(trace)) # Trace length: 500
print('trace[0]:', trace[0]) # trace[0]: {'beta1': 2.0772965015391716, 'beta2': -0.02971672503615687}


# %%
# メトロポリス法
with model:
    # 101個目から3個置きでサンプルを取得するチェインを3つ作る
    trace_Metropolis = pymc3.sample(1500, step=pymc3.Metropolis(), tune=100, njobs=3, random_seed=0)[::3]

_ = pymc3.traceplot(trace_Metropolis)

pymc3.summary(trace_Metropolis)


# %%
# NUTS(デフォルト)でサンプリング
with model:
    # 101個目から3個置きでサンプルを取得するチェインを3つ作る
    trace_NUTS = pymc3.sample(1500, step=pymc3.NUTS(), tune=100, njobs=3, random_seed=0)[::3]
    
_ = pymc3.traceplot(trace_NUTS)

pymc3.summary(trace_NUTS)

# %% [markdown]
# ## 9.5 MCMCサンプルから事後分布を推定

# %%
# ハミルトニアンモンテカルロ法のサンプリング過程を表示(再掲)
_ = pymc3.traceplot(trace)


# %%
# 同時事後分布p(β1,β2|Y)のプロット
beta1_averages = np.zeros(len(trace), dtype=np.float64)
beta2_averages = np.zeros(len(trace), dtype=np.float64)

for i in trace.chains:
    # 各サンプル列のパラメータの平均値を計算
    beta1_averages += trace.get_values('beta1', chains=i) / trace.nchains
    beta2_averages += trace.get_values('beta2', chains=i) / trace.nchains

plt.plot(beta1_averages, label='beta1')
plt.plot(beta2_averages, label='beta2')
plt.legend()


# %%
# 同時事後分布p(β1,β2|Y)を散布図でプロット
plt.scatter(beta1_averages, beta2_averages, alpha=0.2)
plt.xlabel('beta1')
plt.ylabel('beta2')


# %%
# ハミルトニアンモンテカルロ法の事後分布の統計量を表示(再掲)
pymc3.summary(trace)


# %%

