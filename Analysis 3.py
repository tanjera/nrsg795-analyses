import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats
from scipy.stats import sem
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import f_oneway

import statsmodels.api as sm

# import main data file
df = pd.read_excel(r'C:\Users\Ibi\Google Drive\School, UMSON\2023.FA, NRSG 795 (Biostatistics for Evidence Based Practice)\Analysis 3\NRSG795_Fall23 Analysis 3.xlsx',
                   sheet_name="NRSG795_myclients")

# ------------------------------------------------------------
# Testing Hypotheses 1a & 1b
# ------------------------------------------------------------

# Check assumptions for H1a & H1b
satwt_bfcur = df.loc[(pd.notna(df.satwtbf)) & (pd.notna(df.satcurwt))].loc[:,["satwtbf", "satcurwt"]]

# Pearson r Correlation
pearson_satwt = stats.pearsonr(satwt_bfcur.loc[:, "satwtbf"], satwt_bfcur.loc[:, "satcurwt"])
print("satwt_bfcur Pearson r Correlation: \t\t\tr = %.2f" % pearson_satwt.statistic, "\t\tp = %.4f" % pearson_satwt.pvalue)

# Check assumptions by visualizing histogram of satwtbf and satwtcur
axs = sns.histplot(data=satwt_bfcur.loc[:, "satwtbf"], bins=10)
axs.set_xlabel('Satisfaction with Weight, 5 Years Ago', fontsize=10)
plt.show()
axs.get_figure().clf()

axs = sns.histplot(data=satwt_bfcur.loc[:, "satcurwt"], bins=10)
axs.set_xlabel('Satisfaction with Weight, Current', fontsize=10)
plt.show()
axs.get_figure().clf()

# Linear regression
linreg_satwt = stats.linregress(satwt_bfcur.loc[:, "satwtbf"], y=satwt_bfcur.loc[:, "satcurwt"])
print("linreg_satwt:\t intercept: %.1f" % linreg_satwt.intercept, "\tslope (beta): %.1f" % linreg_satwt.slope)
print(" └→ \t\t\tr: %.2f" % linreg_satwt.rvalue, "\t\tp: %.4f" % linreg_satwt.pvalue)
print(" └→ \t\t\tr^2: %.2f" % math.pow(linreg_satwt.rvalue, 2))

exog = satwt_bfcur.loc[:, "satwtbf"]
endog = satwt_bfcur.loc[:, "satcurwt"]
exog = sm.tools.add_constant(exog)

model = sm.OLS(endog, exog).fit()
print(model.summary())


# Scatterplot for satwt bf vs. cur
axs = sns.scatterplot(x="satwtbf", y="satcurwt", data=satwt_bfcur, palette="tab10", legend=False, alpha=0.3)
axs.set_xlabel('Satisfaction with Weight, 5 Years Ago', fontsize=10)
axs.set_ylabel('Satisfaction with Weight, Current', fontsize=10)
xseq = np.linspace(0, 10, num=100)
axs.plot(xseq, linreg_satwt.intercept + linreg_satwt.slope * xseq, color="#3590ae", lw=3);
plt.show()
axs.get_figure().clf()