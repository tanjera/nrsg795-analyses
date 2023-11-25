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

print (">>>> Descriptive Statistics")
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
print("\nOdds Ratio w/ CI's (e^b d/t logit model)\n", np.exp(conf), "\n")


# ------------------------------------------------------------
# Step 3: Boxplot
# ------------------------------------------------------------

"""
df_boxplot = df.loc[:, ["diag_condition", "asp2change"]]
df_boxplot["diag_condition"] = df_boxplot["diag_condition"].replace(1, "Bipolar").replace(2, "Schizophrenia").replace(3, "Major Depressive Disorder")
sns.set_palette(["#000000"])
axs = sns.boxplot(data=df_boxplot, x="diag_condition", y="asp2change", width=0.65, fill=False)
axs.set_xlabel("Diagnosis", labelpad=10)
axs.set_ylabel("Aspiration to Change Physical Activity", labelpad=10)
plt.yticks(range(0, 51, 5))
plt.show()
plt.clf()
"""

print (">>>> Descriptive Statistics for Boxplot: Aspiration to Change per Diagnosis Group")
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





