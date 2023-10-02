import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats
from scipy.stats import shapiro
from scipy.stats import levene


# import main data file
df = pd.read_excel(r'C:\Users\Ibi\Google Drive\School, UMSON\2023.FA, NRSG 795 (Biostatistics for Evidence Based Practice)\Analysis 2\NRSG795 Fall 23 Analysis 2.xlsx',
                   sheet_name="smokeintervention")

# ------------------------------------------------------------
# PART 1: DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\nPART 1: DESCRIPTIVE STATISTICS:\n")

total_records = len(df)
print("Total records:", total_records)

g1 = df[(df.group == 1)]
g2 = df[(df.group == 2)]

g1_age = g1.loc[:, "age"]
print("G1 Age: \t\tn =", len(g1_age), "\t\tMean: %.1f" % g1_age.mean(), "\tMedian: %.1f" % g1_age.median(),
      "\tSt Dev: %.1f" % g1_age.std(), "\tRange: ", g1_age.min(), "-", g1_age.max())

g1_male = g1.loc[(df.sex == 1)].loc[:, "sex"]
print("G1 Male: \t\tn =", len(g1_male), "\t\t%%: %.1f" % (len(g1_male) / len(g1) * 100))
g1_female = g1.loc[(df.sex == 2)].loc[:, "sex"]
print("G1 Female: \t\tn =", len(g1_female), "\t\t%%: %.1f" % (len(g1_female) / len(g1) * 100))

g1_dep1 = g1.loc[:, "dep1"]
print("G1 Dep1: \t\tn =", len(g1_dep1), "\t\tMean: %.1f" % g1_dep1.mean(), "\tMedian: %.1f" % g1_dep1.median(),
      "\tSt Dev: %.1f" % g1_dep1.std(), "\tRange: ", g1_dep1.min(), "-", g1_dep1.max())

g1_anx1 = g1.loc[:, "anx1"]
print("G1 Anx1: \t\tn =", len(g1_anx1), "\t\tMean: %.1f" % g1_anx1.mean(), "\tMedian: %.1f" % g1_anx1.median(),
      "\tSt Dev: %.1f" % g1_anx1.std(), "\tRange: ", g1_anx1.min(), "-", g1_anx1.max())

g1_smok1 = g1.loc[:, "smok1"]
print("G1 Smok1: \t\tn =", len(g1_smok1), "\t\tMean: %.1f" % g1_smok1.mean(), "\tMedian: %.1f" % g1_smok1.median(),
      "\tSt Dev: %.1f" % g1_smok1.std(), "\tRange: ", g1_smok1.min(), "-", g1_smok1.max())

print("")

g2_age = g2.loc[:, "age"]
print("G2 Age: \t\tn =", len(g2_age), "\t\tMean: %.1f" % g2_age.mean(), "\tMedian: %.1f" % g2_age.median(),
      "\tSt Dev: %.1f" % g2_age.std(), "\tRange: ", g2_age.min(), "-", g2_age.max())

g2_male = g2.loc[(df.sex == 1)].loc[:, "sex"]
print("G2 Male: \t\tn =", len(g2_male), "\t\t%%: %.1f" % (len(g2_male) / len(g2) * 100))
g2_female = g2.loc[(df.sex == 2)].loc[:, "sex"]
print("G2 Female: \t\tn =", len(g2_female), "\t\t%%: %.1f" % (len(g2_female) / len(g2) * 100))

g2_dep1 = g2.loc[:, "dep1"]
print("G2 Dep1: \t\tn =", len(g2_dep1), "\t\tMean: %.1f" % g2_dep1.mean(), "\tMedian: %.1f" % g2_dep1.median(),
      "\tSt Dev: %.1f" % g2_dep1.std(), "\tRange: ", g2_dep1.min(), "-", g2_dep1.max())

g2_anx1 = g2.loc[:, "anx1"]
print("G2 Anx1: \t\tn =", len(g2_anx1), "\t\tMean: %.1f" % g2_anx1.mean(), "\tMedian: %.1f" % g2_anx1.median(),
      "\tSt Dev: %.1f" % g2_anx1.std(), "\tRange: ", g2_anx1.min(), "-", g2_anx1.max())

g2_smok1 = g2.loc[:, "smok1"]
print("G2 Smok1: \t\tn =", len(g2_smok1), "\t\tMean: %.1f" % g2_smok1.mean(), "\tMedian: %.1f" % g2_smok1.median(),
      "\tSt Dev: %.1f" % g2_smok1.std(), "\tRange: ", g2_smok1.min(), "-", g2_smok1.max())

# ------------------------------------------------------------
# PART 2: COMPARING TWO INDEPENDENT GROUP MEANS
# ------------------------------------------------------------

print("\n\nPART 2: COMPARING TWO INDEPENDENT GROUP MEANS:\n")

# Test for normality of distribution to determine which homogeneity test to use
# Shapiro-Wilk Test: if p > 0.05, H0 (normal distribution) is accepted
shapiro_g1_smok1 = shapiro(g1_smok1)
print("G1 Smok1 Shapiro-Wilk Test: \t\tt = %.2f" % shapiro_g1_smok1.statistic, "\t\tp = %.4f" % shapiro_g1_smok1.pvalue)
shapiro_g2_smok1 = shapiro(g2_smok1)
print("G2 Smok1 Shapiro-Wilk Test: \t\tt = %.2f" % shapiro_g2_smok1.statistic, "\t\tp = %.4f" % shapiro_g2_smok1.pvalue)

print("")
# Test for homogeneity of variance
# Since g1_smok1 and g2_smok2 are normally distributed (as shown by Shapiro-Wilk test),
# Will use levene's Test; If p > 0.05, H0 (equal/homogenous variances) is accepted
levene_smok1 = levene(g1_smok1, g2_smok1)
print("Smok1 Levene's Variance: \t\t\tt = %.2f" % levene_smok1.statistic, "\t\tp = %.4f" % levene_smok1.pvalue)

print("")
# Test for comparison of independent group means
# Independent t Test: H0 (population means are equal) versus H1 (p < 0.05, population means are not equal)
# Note: test is inherently 2-tailed (outputs positive and negative results); for a 1-tailed test, p/2 < alpha
#   would accept the H1 and reject H0
# Also for 1-tailed test, need to set alternative hypothesis in function!!
ind_ttest_smok1 = stats.ttest_ind(g1_smok1, g2_smok1)
print("Smok1 Independent t Test: \t\t\tt = %.2f" % ind_ttest_smok1.statistic, "\t\tp = %.4f" % ind_ttest_smok1.pvalue)
print(" └→ \t", ind_ttest_smok1.confidence_interval(confidence_level=0.95))

# ------------------------------------------------------------
# PART 3: COMPARING MEANS OF A SINGLE GROUP, PRE-TEST POST-TEST
# ------------------------------------------------------------

print("\n\nPART 3: COMPARING MEANS OF A SINGLE GROUP, PRE-TEST POST-TEST:\n")

print("G1 Smok1: \t\tn =", len(g1_smok1), "\t\tMean: %.1f" % g1_smok1.mean(), "\tMedian: %.1f" % g1_smok1.median(),
      "\tSt Dev: %.1f" % g1_smok1.std(), "\tRange: ", g1_smok1.min(), "-", g1_smok1.max())
g1_smok2 = g1.loc[:, "smok2"]
print("G1 Smok2: \t\tn =", len(g1_smok2), "\t\tMean: %.1f" % g1_smok2.mean(), "\tMedian: %.1f" % g1_smok2.median(),
      "\tSt Dev: %.1f" % g1_smok2.std(), "\tRange: ", g1_smok2.min(), "-", g1_smok2.max())

print("")

print("G2 Smok1: \t\tn =", len(g2_smok1), "\t\tMean: %.1f" % g2_smok1.mean(), "\tMedian: %.1f" % g2_smok1.median(),
      "\tSt Dev: %.1f" % g2_smok1.std(), "\tRange: ", g2_smok1.min(), "-", g2_smok1.max())
g2_smok2 = g2.loc[:, "smok2"]
print("G2 Smok2: \t\tn =", len(g2_smok2), "\t\tMean: %.1f" % g2_smok2.mean(), "\tMedian: %.1f" % g2_smok2.median(),
      "\tSt Dev: %.1f" % g2_smok2.std(), "\tRange: ", g2_smok2.min(), "-", g2_smok2.max())

print("")

shapiro_g1_smok2 = shapiro(g1_smok2)
print("G1 Smok2 Shapiro-Wilk Test: \t\tt = %.2f" % shapiro_g1_smok2.statistic, "\t\tp = %.4f" % shapiro_g1_smok2.pvalue)
shapiro_g2_smok2 = shapiro(g2_smok2)
print("G2 Smok2 Shapiro-Wilk Test: \t\tt = %.2f" % shapiro_g2_smok2.statistic, "\t\tp = %.4f" % shapiro_g2_smok2.pvalue)

print("")

levene_g1_smok = levene(g1_smok1, g1_smok2)
print("G1_Smok? Levene's Variance: \t\tt = %.2f" % levene_g1_smok.statistic, "\t\tp = %.4f" % levene_g1_smok.pvalue)

levene_g2_smok = levene(g2_smok1, g2_smok2)
print("G2_Smok? Levene's Variance: \t\tt = %.2f" % levene_g2_smok.statistic, "\t\tp = %.4f" % levene_g2_smok.pvalue)

print("")

p1ttest_g1_smok = stats.ttest_rel(g1_smok1, g1_smok2, alternative="greater")
print("G1_Smok? Paired 1-sided t Test: \tt = %.2f" % p1ttest_g1_smok.statistic, "\t\tp = %.4f" % p1ttest_g1_smok.pvalue)
print(" └→ \t", p1ttest_g1_smok.confidence_interval(confidence_level=0.95))

p1ttest_g2_smok = stats.ttest_rel(g2_smok1, g2_smok2, alternative="greater")
print("G2_Smok? Paired 1-sided t Test: \tt = %.2f" % p1ttest_g2_smok.statistic, "\t\tp = %.4f" % p1ttest_g2_smok.pvalue)
print(" └→ \t", p1ttest_g2_smok.confidence_interval(confidence_level=0.95))

