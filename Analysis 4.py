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
df = pd.read_excel(r'C:\Users\Ibi\Google Drive\School, UMSON\2023.FA, NRSG 795 (Biostatistics for Evidence Based Practice)\Analysis 4\NRSG795 Fall23 Analysis 4.xlsx',
                   sheet_name="statsurvey")


# ------------------------------------------------------------
# Part A: Multiple Linear Regression
# ------------------------------------------------------------

# Display pairplot for bivariate analysis
sns.pairplot(data=df, x_vars=["age", "sex", "anxiety", "confid", "fearstat"], y_vars="exam")
#sns.pairplot(data=df, x_vars=["Age", "Sex", "Anxiety", "Confidence", "Fear"], y_vars="Exam Score")
plt.show()
plt.clf()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Print correlation map for bivariate analysis
print("Pandas Dataframe Correlation Map (Pearson r)")
corr_map = df.loc[:, ["exam", "age", "sex", "anxiety", "confid", "fearstat"]].corr()
#corr_map = df.loc[:, ["Exam Score", "Age", "Sex", "Anxiety", "Confidence", "Fear"]].corr()
print(corr_map, "\n\n")

# Display heatmap of correlation map for bivariate analysis
plt.figure(figsize=(7,7))
sns.heatmap(corr_map, annot=True, cmap="RdBu")
plt.show()
plt.clf()

# Setup, fit, and display statistics on multiple linear regression model
exog = df.loc[:, ["age", "sex", "anxiety", "confid", "fearstat"]]
endog = df.loc[:, "exam"]
#exog = df.loc[:, ["Age", "Sex", "Anxiety", "Confidence", "Fear"]]
#endog = df.loc[:, "Exam Score"]
exog = sm.tools.add_constant(exog)

model = sm.OLS(endog, exog).fit()
print(model.summary())




