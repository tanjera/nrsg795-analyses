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

# Descriptive Statistics

print (">>>> Descriptive Statistics")
df_age = df.loc[:, "age"]
print("age: \t\t\tn =", len(df_age), "\t\tMean: %.1f" % df_age.mean(), "\tMedian: %.1f" % df_age.median(),
      "\tSt Dev: %.1f" % df_age.std(), "\tRange: ", df_age.min(), "-", df_age.max())
count_sex = len(df.loc[:, "sex"])
count_male = len(df.loc[df["sex"] == 0].loc[:, "sex"])
print("sex: Male: \t\tn =", count_male, "(%.0f" % (count_male / count_sex * 100), "%) \t/", count_sex)
count_female = len(df.loc[df["sex"] == 1].loc[:, "sex"])
print("sex: Female: \tn =", count_female, "(%.0f" % (count_female / count_sex * 100), "%) \t/", count_sex)
df_anxiety = df.loc[:, "anxiety"]
print("anxiety: \t\tn =", len(df_anxiety), "\t\tMean: %.1f" % df_anxiety.mean(), "\tMedian: %.1f" % df_anxiety.median(), 
      "\tSt Dev: %.1f" % df_anxiety.std(), "\tRange: ", df_anxiety.min(), "-", df_anxiety.max())
df_confid = df.loc[:, "confid"]
print("confid: \t\tn =", len(df_confid), "\t\tMean: %.1f" % df_confid.mean(), "\tMedian: %.1f" % df_confid.median(), 
      "\tSt Dev: %.1f" % df_confid.std(), "\tRange: ", df_confid.min(), "-", df_confid.max())
df_fearstat = df.loc[:, "fearstat"]
print("fearstat: \t\tn =", len(df_fearstat), "\t\tMean: %.1f" % df_fearstat.mean(), "\tMedian: %.1f" % df_fearstat.median(),
      "\tSt Dev: %.1f" % df_fearstat.std(), "\tRange: ", df_fearstat.min(), "-", df_fearstat.max())
df_exam = df.loc[:, "exam"]
print("exam: \t\t\tn =", len(df_exam), "\t\tMean: %.1f" % df_exam.mean(), "\tMedian: %.1f" % df_exam.median(),
      "\tSt Dev: %.1f" % df_exam.std(), "\tRange: ", df_exam.min(), "-", df_exam.max())
count_failcourse = len(df.loc[:, "failcourse"])
count_passed = len(df.loc[df["failcourse"] == 0].loc[:, "failcourse"])
print("failcourse: Passed: \tn =", count_passed, "(%.0f" % (count_passed / count_failcourse * 100), "%) \t/", count_failcourse)
count_failed = len(df.loc[df["failcourse"] == 1].loc[:, "failcourse"])
print("failcourse: Failed: \tn =", count_failed, "(%.0f" % (count_failed / count_failcourse * 100), "%) \t/", count_failcourse)
print("\n")

# Display pairplot for bivariate analysis
"""
sns.pairplot(data=df, x_vars=["age", "sex", "anxiety", "confid", "fearstat"], y_vars="exam")
#sns.pairplot(data=df, x_vars=["Age", "Sex", "Anxiety", "Confidence", "Fear"], y_vars="Exam Score")
plt.show()
plt.clf()
"""

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Print correlation map for bivariate analysis
print(">>>> Pandas Dataframe Correlation Map (Pearson r)")
corr_map = df.loc[:, ["exam", "age", "sex", "anxiety", "confid", "fearstat"]].corr()
#corr_map = df.loc[:, ["Exam Score", "Age", "Sex", "Anxiety", "Confidence", "Fear"]].corr()
print(corr_map, "\n\n")

# Display heatmap of correlation map for bivariate analysis
"""
plt.figure(figsize=(7,7))
sns.heatmap(corr_map, annot=True, cmap="RdBu")
plt.show()
plt.clf()
"""

# Setup, fit, and display statistics on multiple linear regression model
exog = df.loc[:, ["age", "sex", "anxiety", "confid", "fearstat"]]
endog = df.loc[:, "exam"]
exog = sm.tools.add_constant(exog)
model = sm.OLS(endog, exog).fit()
print(">>>> Regression Result with ALL exogenous variables")
print(model.summary())

exog = df.loc[:, ["age", "sex", "confid", "fearstat"]]
endog = df.loc[:, "exam"]
exog = sm.tools.add_constant(exog)
model = sm.OLS(endog, exog).fit()
print("\n\n>>>> Regression Result withOUT anxiety")
print(model.summary())

exog = df.loc[:, ["age", "sex", "anxiety", "fearstat"]]
endog = df.loc[:, "exam"]
exog = sm.tools.add_constant(exog)
model = sm.OLS(endog, exog).fit()
print("\n\n>>>> Regression Result withOUT confid")
print(model.summary())
