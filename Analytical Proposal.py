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
df = pd.read_excel(r'C:\Users\Ibi\Google Drive\School, UMSON\2023.FA, NRSG 795 (Biostatistics for Evidence Based Practice)\Analytical Proposal\ProposalAnalysis_SMIdata.xlsx',
                   sheet_name="SMI_CDMP")


# ------------------------------------------------------------
# Step 1: Descriptive Statistics
# ------------------------------------------------------------

print (">>>> Descriptive Statistics\n")
df_age = df.loc[:, "age"]
print("age: \t\t\t\t\t\tn =", len(df_age), "\t\tMean: %.1f" % df_age.mean(), "\tMedian: %.1f" % df_age.median(),
      "\tSt Dev: %.1f" % df_age.std(), "\tRange: ", df_age.min(), "-", df_age.max())
print("")
count_sex = len(df.loc[:, "sex"])
count_male = len(df.loc[df["sex"] == 0].loc[:, "sex"])
print("sex: Male: \t\t\t\t\tn =", count_male, "(%.0f" % (count_male / count_sex * 100), "%) \t/", count_sex)
count_female = len(df.loc[df["sex"] == 1].loc[:, "sex"])
print("sex: Female: \t\t\t\tn =", count_female, "(%.0f" % (count_female / count_sex * 100), "%) \t/", count_sex)
print("")
count_diag = len(df.loc[:, "diag_condition"])
count_diag_bipolar = len(df.loc[df["diag_condition"] == 1].loc[:, "diag_condition"])
print("diag: Bipolar: \t\t\t\tn =", count_diag_bipolar, "(%.0f" % (count_diag_bipolar / count_diag * 100), "%) \t/", count_diag)
count_diag_schizoph = len(df.loc[df["diag_condition"] == 2].loc[:, "diag_condition"])
print("diag: Schizophrenia: \t\tn =", count_diag_schizoph, "(%.0f" % (count_diag_schizoph / count_diag * 100), "%) \t/", count_diag)
count_diag_mdd = len(df.loc[df["diag_condition"] == 3].loc[:, "diag_condition"])
print("diag: Major Depression: \tn =", count_diag_mdd, "(%.0f" % (count_diag_mdd / count_diag * 100), "%) \t/", count_diag)
print("")
count_bmi = len(df.loc[:, "bmi_cat"])
count_bmi_normal = len(df.loc[df["bmi_cat"] == 0].loc[:, "bmi_cat"])
print("bmi: Normal: \t\t\tn =", count_bmi_normal, "(%.0f" % (count_bmi_normal / count_bmi * 100), "%) \t/", count_bmi)
count_bmi_over = len(df.loc[df["bmi_cat"] == 1].loc[:, "bmi_cat"])
print("bmi: Over/Obese: \t\tn =", count_bmi_over, "(%.0f" % (count_bmi_over / count_bmi * 100), "%) \t/", count_bmi)
print("")
df_aspchange = df.loc[:, "asp2change"]
print("asp2change: \t\t\tn =", len(df_aspchange), "\t\tMean: %.1f" % df_aspchange.mean(), "\tMedian: %.1f" % df_aspchange.median(),
      "\tSt Dev: %.1f" % df_aspchange.std(), "\tRange: ", df_aspchange.min(), "-", df_aspchange.max())
df_startsteps = df.loc[:, "start_steps"]
print("start_steps: \t\t\tn =", len(df_startsteps), "\t\tMean: %.1f" % df_startsteps.mean(), "\tMedian: %.1f" % df_startsteps.median(),
      "\tSt Dev: %.1f" % df_startsteps.std(), "\tRange: ", df_startsteps.min(), "-", df_startsteps.max())
df_sdsteps = df.loc[:, "60d_steps"]
print("60d_steps: \t\t\t\tn =", len(df_sdsteps), "\t\tMean: %.1f" % df_sdsteps.mean(), "\tMedian: %.1f" % df_sdsteps.median(),
      "\tSt Dev: %.1f" % df_sdsteps.std(), "\tRange: ", df_sdsteps.min(), "-", df_sdsteps.max())
print("\n")


# ------------------------------------------------------------
# Step 2, Option 4: Chi-Square & Logistic Regression of SMI by BMI
# ------------------------------------------------------------

print (">>>> Association of Diagnosis with BMI Category")
df_chisq = df.loc[:, ["diag_condition", "bmi_cat"]]
df_chisq["diag_condition"] = df_chisq["diag_condition"].replace(1, "Bipolar").replace(2, "Schizophrenia").replace(3, "Major Depressive Disorder")
df_chisq["bmi_cat"] = df_chisq["bmi_cat"].replace(0, "Normal").replace(1, "Overweight/Obese")
crosstab = pd.crosstab(df_chisq.loc[:, "diag_condition"], df_chisq.loc[:, "bmi_cat"])
print("\nCrosstabulation of Diagnosis x BMI Category:\n", crosstab)

chicont = stats.chi2_contingency(crosstab)
print("\nChi-Square Test via stats.chi2_contingency:\n", chicont, "\n")

exog = df.loc[:, ["dc_diag_2", "dc_diag_3"]]
exog = sm.tools.add_constant(exog)
endog = df.loc[:, "bmi_cat"]
logreg = sm.Logit(endog, exog).fit()
print (logreg.summary())
print("DUMMY CODING KEY:")
print("\tbipolar (1)\t\t\treference group")
print("\tschizophrenia (2)\tdc_diag_2")
print("\tmaj dep d/o (3)\t\tdc_diag_3")

params = logreg.params
conf = logreg.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['5% CI', '95% CI', 'Odds Ratio']
# Odds Ratio (and its CI's) calculated as euler's e ^ model beta
print("\nOdds Ratio w/ CI's (e^b d/t logit model)\n", np.exp(conf), "\n\n")


# ------------------------------------------------------------
# Step 2, Option 5: Correlation & Linear Regression of Asp2Change by # Steps @ 60 Days
# ------------------------------------------------------------

print (">>>> Association of Aspiration to Change with # of Steps at 60 Days\n")

df_linreg = df.loc[:, ["asp2change", "60d_steps"]]

pearson = stats.pearsonr(df_linreg.loc[:, "asp2change"], df_linreg.loc[:, "60d_steps"])
print("Pearson r: \t\t\t\t\tr = %.2f" % pearson.statistic, "\t\tp = %.4f" % pearson.pvalue)
normaltest = stats.normaltest(df_linreg.loc[:, "asp2change"])
print("Normal Test (asp2change): \tf = %.2f" % normaltest.statistic, "\t\tp = %.4f" % normaltest.pvalue)
skew = stats.skew(df_linreg.loc[:, "asp2change"])
print("Skew (asp2change): \t\t\tf = %.2f" % skew)
kurtosis = stats.kurtosis(df_linreg.loc[:, "asp2change"])
print("Kurtosis (asp2change): \t\tf = %.2f" % kurtosis)
normaltest = stats.normaltest(df_linreg.loc[:, "60d_steps"])
print("Normal Test (60d_steps): \tf = %.2f" % normaltest.statistic, "\t\tp = %.4f" % normaltest.pvalue)
skew = stats.skew(df_linreg.loc[:, "60d_steps"])
print("Skew (60d_steps): \t\t\tf = %.2f" % skew)
kurtosis = stats.kurtosis(df_linreg.loc[:, "60d_steps"])
print("Kurtosis (60d_steps): \t\tf = %.2f" % kurtosis)
print("")

"""
axs = sns.histplot(data=df_linreg.loc[:, "asp2change"], bins=10)
axs.set_xlabel('Aspiration to Change', fontsize=10)
plt.show()
axs.get_figure().clf()

axs = sns.histplot(data=df_linreg.loc[:, "60d_steps"], bins=10)
axs.set_xlabel('Steps Taken at 60 Days', fontsize=10)
plt.show()
axs.get_figure().clf()
"""

# Linear regression
exog = df_linreg.loc[:, "asp2change"]
endog = df_linreg.loc[:, "60d_steps"]
exog = sm.tools.add_constant(exog)

linreg = sm.OLS(endog, exog).fit()
print(linreg.summary(), "\n")

heterobp = sm.stats.het_breuschpagan(linreg.resid, linreg.model.exog)
print("Heteroscedasticity (BP): \tf = %.2f" % heterobp[0], "\t\tp = %.4f" % heterobp[1], "\n\n")


# ------------------------------------------------------------
# Step 3: Boxplot
# ------------------------------------------------------------

"""
df_boxplot = df.loc[:, ["diag_condition", "asp2change"]]
df_boxplot = df_boxplot.sort_values("diag_condition")
df_boxplot["diag_condition"] = df_boxplot["diag_condition"].replace(1, "Bipolar").replace(2, "Schizophrenia").replace(3, "Major Depressive Disorder")
sns.set_palette(["#000000"])
axs = sns.boxplot(data=df_boxplot, x="diag_condition", y="asp2change", width=0.65, fill=False)
axs.set_xlabel("Diagnosis", labelpad=10)
axs.set_ylabel("Aspiration to Change Physical Activity", labelpad=10)
plt.yticks(range(0, 51, 5))
plt.show()
plt.clf()
"""

print (">>>> Descriptive Statistics for Boxplot: Aspiration to Change per Diagnosis Group\n")
df_aspchange_bipolar = df.loc[df["diag_condition"] == 1].loc[:, "asp2change"]
print("asp2change / bipolar: \tn =", len(df_aspchange_bipolar), "\t\tMean: %.1f" % df_aspchange_bipolar.mean(), "\tMedian: %.1f" % df_aspchange_bipolar.median(),
      "\tSt Dev: %.1f" % df_aspchange_bipolar.std(),
      "\n\t\t\t\t\t\t\t\t\tRange: ", df_aspchange_bipolar.min(), "-", df_aspchange_bipolar.max(),
      "\t\t\tIQR: %.1f" % (df_aspchange_bipolar.median() - (0.6745 * df_aspchange_bipolar.std())), "-",
      "%.1f" % (df_aspchange_bipolar.median() + (0.6745 * df_aspchange_bipolar.std())))

df_aspchange_schizoph = df.loc[df["diag_condition"] == 2].loc[:, "asp2change"]
print("asp2change / schizoph: \tn =", len(df_aspchange_schizoph), "\t\tMean: %.1f" % df_aspchange_schizoph.mean(), "\tMedian: %.1f" % df_aspchange_schizoph.median(),
      "\tSt Dev: %.1f" % df_aspchange_schizoph.std(),
      "\n\t\t\t\t\t\t\t\t\tRange: ", df_aspchange_schizoph.min(), "-", df_aspchange_schizoph.max(),
      "\t\t\tIQR: %.1f" % (df_aspchange_schizoph.median() - (0.6745 * df_aspchange_schizoph.std())), "-",
      "%.1f" % (df_aspchange_schizoph.median() + (0.6745 * df_aspchange_schizoph.std())))

df_aspchange_mdd = df.loc[df["diag_condition"] == 3].loc[:, "asp2change"]
print("asp2change / mdd: \t\tn =", len(df_aspchange_mdd), "\t\tMean: %.1f" % df_aspchange_mdd.mean(), "\tMedian: %.1f" % df_aspchange_mdd.median(),
      "\tSt Dev: %.1f" % df_aspchange_mdd.std(),
      "\n\t\t\t\t\t\t\t\t\tRange: ", df_aspchange_mdd.min(), "-", df_aspchange_mdd.max(),
      "\t\t\tIQR: %.1f" % (df_aspchange_mdd.median() - (0.6745 * df_aspchange_mdd.std())), "-",
      "%.1f" % (df_aspchange_mdd.median() + (0.6745 * df_aspchange_mdd.std())))


# ------------------------------------------------------------
# Step 4: Additional Visualizations
# ------------------------------------------------------------

"""
# Barplot of Diagnosis by amount Binned by BMI Category
df_barplot = pd.DataFrame(columns=["Diagnosis", "BMI", "Amount"])
for diag in df["diag_condition"].unique():
      for bmi in df["bmi_cat"].unique():
            df_barplot.loc[len(df_barplot.index)] = [diag, bmi, len(df.loc[(df["diag_condition"] == diag) & (df["bmi_cat"] == bmi)])]
df_barplot = df_barplot.sort_values("BMI").sort_values("Diagnosis")
df_barplot["Diagnosis"] = df_barplot["Diagnosis"].replace(1, "Bipolar").replace(2, "Schizophrenia").replace(3, "Major Depressive Disorder")
df_barplot["BMI"] = df_barplot["BMI"].replace(0, "Normal").replace(1, "Overweight/Obese")
axs = sns.barplot(data=df_barplot, x="Diagnosis", y="Amount", hue="BMI")
plt.yticks(range(0, 20, 2))
axs.set_xlabel("Diagnosis", labelpad=10)
axs.set_ylabel("Amount of Clients", labelpad=10)
plt.show()
plt.clf()
"""

# Scatterplot of Asp2change by Steps at 60d w/ Linreg slope
df_linreg = df.loc[:, ["asp2change", "60d_steps"]]
exog = df_linreg.loc[:, "asp2change"]
endog = df_linreg.loc[:, "60d_steps"]
exog = sm.tools.add_constant(exog)
linreg = sm.OLS(endog, exog).fit()

"""
axs = sns.scatterplot(x="asp2change", y="60d_steps", data=df_linreg, palette="tab10", legend=False, alpha=0.75)
axs.set_xlabel('Aspiration to Change Scores', fontsize=10)
axs.set_ylabel('Steps Taken at 60 Days', fontsize=10)
xseq = np.linspace(20, 50, num=100)
axs.plot(xseq, linreg.params["const"] + linreg.params["asp2change"] * xseq, color="#3590ae", lw=1.5);
plt.show()
axs.get_figure().clf()
"""







