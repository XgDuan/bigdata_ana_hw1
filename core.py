import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
import warnings
np.random.seed(233)
warnings.filterwarnings("ignore")
plt.style.use('classic')


def norm_test(data, label, categories, thres=0.05, transform=None):
    df = pd.DataFrame(columns=['category', 'num-of-data', 'mean', 'p-value', 'normality', 'std-dev', 'homogeneity'])
    for cat in categories:
        data_cat = data[data['群类别'] == cat][label]
        if transform is not None:
            data_cat = [transform(x) for x in data_cat]
        p_value = stats.shapiro(data_cat)[1]
        mean = np.mean(data_cat)
        std_dev = np.sqrt(np.sum((data_cat - mean)**2) / (len(data_cat) - 1))
        df = df.append({'category':cat, 'num-of-data':len(data_cat), 'mean': mean,
                        'p-value': p_value, 'normality': p_value > 0.05, 'std-dev':std_dev},
                       ignore_index=True)
    homogeneity = max(df['std-dev']) < 2 * min(df['std-dev'])
    df = df.append({'homogeneity': 'True' if homogeneity else 'False'}, ignore_index=True)
    return df.fillna('')

def anova(data):
    data = data[:-1]  # remove the final line
    df = pd.DataFrame(columns=['Source', 'SS', 'df', 'MS', 'F', 'p'])
    grand_mean = np.sum(data['num-of-data'] * data['mean']) / np.sum(data['num-of-data'])
    SS_b = np.sum(data['num-of-data'] * (data['mean'] - grand_mean)**2)
    df_b = len(data) - 1
    MS_b = SS_b / df_b
    SS_w = np.sum(data['std-dev'] ** 2 * (data['num-of-data'] - 1))
    df_w = np.sum(data['num-of-data']) - len(data)
    MS_w = SS_w / df_w
    F = MS_b / MS_w
    p = 1 - stats.f(df_b, df_w).cdf(F)
    df = df.append({'Source': 'Between', 'SS': SS_b, 'df': df_b, 'MS': MS_b, 'F': F, 'p': p}, ignore_index=True)
    df = df.append({'Source': 'Within', 'SS': SS_w, 'df': df_w, 'MS': MS_w}, ignore_index=True)
    df = df.append({'Source': 'Total', 'SS': SS_b + SS_w, 'df': df_b + df_w}, ignore_index=True)
    return df.fillna('')