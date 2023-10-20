import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pingouin as pg

from scipy import stats
from scipy.stats import sem
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import f_oneway

# import main data file
df = pd.read_excel(r'C:\Users\Ibi\Google Drive\School, UMSON\2023.FA, NRSG 795 (Biostatistics for Evidence Based Practice)\Analysis 3\NRSG795_Fall23 Analysis 3.xlsx',
                   sheet_name="NRSG795_myclients")

# ------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\nDESCRIPTIVE STATISTICS:\n")

total_records = len(df)
print("Total records:", total_records, "\n")

df_age = df.loc[(pd.notna(df.age))].loc[:, "age"]
print("df.age: \t\tn =", len(df_age), "\t\tMean: %.1f" % df_age.mean(), "\tMedian: %.1f" % df_age.median(),
      "\tSt Dev: %.1f" % df_age.std(), "\tRange: ", df_age.min(), "-", df_age.max(),
      "\tSEM: %.2f" % sem(df_age))

df_educ = df.loc[(pd.notna(df.educ))].loc[:, "educ"]
print("df.educ: \t\tn =", len(df_educ), "\t\tMean: %.1f" % df_educ.mean(), "\tMedian: %.1f" % df_educ.median(),
      "\tSt Dev: %.1f" % df_educ.std(), "\tRange: ", df_educ.min(), "-", df_educ.max(),
      "\tSEM: %.2f" % sem(df_educ))

df_satcurwt = df.loc[(pd.notna(df.satcurwt))].loc[:, "satcurwt"]
print("df.satcurwt: \tn =", len(df_satcurwt), "\t\tMean: %.1f" % df_satcurwt.mean(), "\tMedian: %.1f" % df_satcurwt.median(),
      "\tSt Dev: %.1f" % df_satcurwt.std(), "\tRange: ", df_satcurwt.min(), "-", df_satcurwt.max(),
      "\tSEM: %.2f" % sem(df_satcurwt))

df_satwtbf = df.loc[(pd.notna(df.satwtbf))].loc[:, "satwtbf"]
print("df.satwtbf: \tn =", len(df_satwtbf), "\t\tMean: %.1f" % df_satwtbf.mean(), "\tMedian: %.1f" % df_satwtbf.median(),
      "\tSt Dev: %.1f" % df_satwtbf.std(), "\tRange: ", df_satwtbf.min(), "-", df_satwtbf.max(),
      "\tSEM: %.2f" % sem(df_satwtbf))

df_health = df.loc[(pd.notna(df.health))].loc[:, "health"]
print("df.health: \t\tn =", len(df_health), "\t\tMean: %.1f" % df_health.mean(), "\tMedian: %.1f" % df_health.median(),
      "\tSt Dev: %.1f" % df_health.std(), "\tRange: ", df_health.min(), "-", df_health.max(),
      "\tSEM: %.2f" % sem(df_health))

df_qolcur = df.loc[(pd.notna(df.qolcur))].loc[:, "qolcur"]
print("df.qolcur: \t\tn =", len(df_qolcur), "\t\tMean: %.1f" % df_qolcur.mean(), "\tMedian: %.1f" % df_qolcur.median(),
      "\tSt Dev: %.1f" % df_qolcur.std(), "\tRange: ", df_qolcur.min(), "-", df_qolcur.max(),
      "\tSEM: %.2f" % sem(df_qolcur))

df_qolbf = df.loc[(pd.notna(df.qolbf))].loc[:, "qolbf"]
print("df.qolbf: \t\tn =", len(df_qolbf), "\t\tMean: %.1f" % df_qolbf.mean(), "\tMedian: %.1f" % df_qolbf.median(),
      "\tSt Dev: %.1f" % df_qolbf.std(), "\tRange: ", df_qolbf.min(), "-", df_qolbf.max(),
      "\tSEM: %.2f" % sem(df_qolbf))

df_energy = df.loc[(pd.notna(df.energy))].loc[:, "energy"]
print("df.energy: \t\tn =", len(df_energy), "\t\tMean: %.1f" % df_energy.mean(), "\tMedian: %.1f" % df_energy.median(),
      "\tSt Dev: %.1f" % df_energy.std(), "\tRange: ", df_energy.min(), "-", df_energy.max(),
      "\tSEM: %.2f" % sem(df_energy))

df_pain = df.loc[(pd.notna(df.pain))].loc[:, "pain"]
print("df.pain: \t\tn =", len(df_pain), "\t\tMean: %.1f" % df_pain.mean(), "\tMedian: %.1f" % df_pain.median(),
      "\tSt Dev: %.1f" % df_pain.std(), "\tRange: ", df_pain.min(), "-", df_pain.max(),
      "\tSEM: %.2f" % sem(df_pain))

print("")

df_gender__all = df.loc[(pd.notna(df.gender))].loc[:, "gender"]
print("df.gender, n =", len(df_gender__all))
df_gender__male = df.loc[(df.gender == 0)].loc[:, "gender"]
print("- male: \t\tn =", len(df_gender__male), "\t\t%%: %.1f" % (len(df_gender__male) / len(df_gender__all) * 100))
df_gender__female = df.loc[(df.gender == 1)].loc[:, "gender"]
print("- female: \t\tn =", len(df_gender__female), "\t\t%%: %.1f" % (len(df_gender__female) / len(df_gender__all) * 100))

print("")

df_marital__all = df.loc[(pd.notna(df.marital))].loc[:, "marital"]
print("df.marital, n =", len(df_marital__all))
df_marital__never = df.loc[(df.marital == 1)].loc[:, "marital"]
print("- never: \t\tn =", len(df_marital__never), "\t\t%%: %.1f" % (len(df_marital__never) / len(df_marital__all) * 100))
df_marital__married = df.loc[(df.marital == 2)].loc[:, "marital"]
print("- married: \t\tn =", len(df_marital__married), "\t\t%%: %.1f" % (len(df_marital__married) / len(df_marital__all) * 100))
df_marital__other = df.loc[(df.marital >= 3) & (df.marital <= 6)].loc[:, "marital"]
print("- other: \t\tn =", len(df_marital__other), "\t\t%%: %.1f" % (len(df_marital__other) / len(df_marital__all) * 100))

print("")

df_smoke__all = df.loc[(pd.notna(df.smoke))].loc[:, "smoke"]
print("df.smoke, n =", len(df_smoke__all))
df_smoke__never = df.loc[(df.smoke == 0)].loc[:, "smoke"]
print("- never: \t\tn =", len(df_smoke__never), "\t\t%%: %.1f" % (len(df_smoke__never) / len(df_smoke__all) * 100))
df_smoke_quit = df.loc[(df.smoke == 1)].loc[:, "smoke"]
print("- quit: \t\tn =", len(df_smoke_quit), "\t\t%%: %.1f" % (len(df_smoke_quit) / len(df_smoke__all) * 100))
df_smoke_still = df.loc[(df.smoke == 2)].loc[:, "smoke"]
print("- still: \t\tn =", len(df_smoke_still), "\t\t%%: %.1f" % (len(df_smoke_still) / len(df_smoke__all) * 100))

print("")

df_depressed__all = df.loc[(pd.notna(df.depressed))].loc[:, "depressed"]
print("df.depressed, n =", len(df_depressed__all))
df_depressed__no = df.loc[(df.depressed == 0)].loc[:, "depressed"]
print("- no: \t\tn =", len(df_depressed__no), "\t\t%%: %.1f" % (len(df_depressed__no) / len(df_depressed__all) * 100))
df_depressed__yes = df.loc[(df.depressed == 1)].loc[:, "depressed"]
print("- yes: \t\tn =", len(df_depressed__yes), "\t\t%%: %.1f" % (len(df_depressed__yes) / len(df_depressed__all) * 100))

print("")

df_exer__all = df.loc[(pd.notna(df.exer))].loc[:, "exer"]
print("df.exer, n =", len(df_exer__all))
df_exer__rarely = df.loc[(df.exer == 1)].loc[:, "exer"]
print("- rarely: \t\tn =", len(df_exer__rarely), "\t\t%%: %.1f" % (len(df_exer__rarely) / len(df_exer__all) * 100))
df_exer__sometimes = df.loc[(df.exer == 2)].loc[:, "exer"]
print("- sometimes: \tn =", len(df_exer__sometimes), "\t\t%%: %.1f" % (len(df_exer__sometimes) / len(df_exer__all) * 100))
df_exer__often = df.loc[(df.exer == 3)].loc[:, "exer"]
print("- often: \t\tn =", len(df_exer__often), "\t\t%%: %.1f" % (len(df_exer__often) / len(df_exer__all) * 100))
df_exer__routinely = df.loc[(df.exer == 4)].loc[:, "exer"]
print("- routinely: \tn =", len(df_exer__routinely), "\t\t%%: %.1f" % (len(df_exer__routinely) / len(df_exer__all) * 100))

print("")

df_eat__all = df.loc[(pd.notna(df.eat))].loc[:, "eat"]
print("df.eat, n =", len(df_eat__all))
df_eat__rarely = df.loc[(df.eat == 1)].loc[:, "eat"]
print("- rarely: \t\tn =", len(df_eat__rarely), "\t\t%%: %.1f" % (len(df_eat__rarely) / len(df_eat__all) * 100))
df_eat__sometimes = df.loc[(df.eat == 2)].loc[:, "eat"]
print("- sometimes: \tn =", len(df_eat__sometimes), "\t\t%%: %.1f" % (len(df_eat__sometimes) / len(df_eat__all) * 100))
df_eat__often = df.loc[(df.eat == 3)].loc[:, "eat"]
print("- often: \t\tn =", len(df_eat__often), "\t\t%%: %.1f" % (len(df_eat__often) / len(df_eat__all) * 100))
df_eat__routinely = df.loc[(df.eat == 4)].loc[:, "eat"]
print("- routinely: \tn =", len(df_eat__routinely), "\t\t%%: %.1f" % (len(df_eat__routinely) / len(df_eat__all) * 100))

print("")

# ------------------------------------------------------------
# Testing Hypotheses 1a & 1b
# ------------------------------------------------------------

# Check assumptions for H1a & H1b: multivariate normality w/ pngn.multivariate_normality()
satwt_bfcur = df.loc[(pd.notna(df.satwtbf)) & (pd.notna(df.satcurwt))].loc[:,["satwtbf", "satcurwt"]]

# Pearson r Correlation
pearson_satwt = stats.pearsonr(satwt_bfcur.loc[:, "satwtbf"], satwt_bfcur.loc[:, "satcurwt"])
print("satwt_bfcur Pearson r Correlation: \t\t\tr = %.2f" % pearson_satwt.statistic, "\t\tp = %.4f" % pearson_satwt.pvalue)

# Henze-Zirkler multivariate normality test
hzmn_satwt = pg.multivariate_normality(satwt_bfcur)
print("satwt_bfcur Henze-Zirkler multi normality: \tt = %.2f" % hzmn_satwt.hz, "\t\tp = %.4f" % hzmn_satwt.pval)

# Check assumptions by visualizing histogram of satwtbf and satwtcur
axs = sns.histplot(data=satwt_bfcur.loc[:, "satwtbf"], bins=10)
plt.show()
axs.get_figure().clf()

axs = sns.histplot(data=satwt_bfcur.loc[:, "satcurwt"], bins=10)
plt.show()
axs.get_figure().clf()

# Linear regression of satwt bf vs. cur
linreg_satwt = stats.linregress(satwt_bfcur.loc[:, "satwtbf"], satwt_bfcur.loc[:, "satcurwt"])

# Scatterplot for satwt bf vs. cur
axs = sns.scatterplot(x="satwtbf", y="satcurwt", data=satwt_bfcur, palette="tab10", legend=False, alpha=0.3)
axs.set_xlabel('Satisfaction with Weight, 5 Years Ago', fontsize=12)
axs.set_ylabel('Satisfaction with Weight, Current', fontsize=12)
xseq = np.linspace(0, 10, num=100)
axs.plot(xseq, linreg_satwt.intercept + linreg_satwt.slope * xseq, color="#3590ae", lw=3);
plt.show()
axs.get_figure().clf()