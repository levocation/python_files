from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['x', 'y'])
df.loc[0] = [2, 3]
df.loc[1] = [2, 11]
df.loc[2] = [3, 7]
df.head(3)

sb.lmplot(x='x', y='y', data=df, fit_reg=False, scatter_kws={'s': 3})