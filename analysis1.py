import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from colorama import Fore, Back, Style
import warnings

warnings.filterwarnings("ignore")

def validate_data(column, missing_code, to_exit: bool = True):
    num_missing = len(df.loc[df[column] == missing_code].loc[:,column])
    num_blank = len(df.loc[df[column] == ''].loc[:,column])    

    if num_missing > 0:        
        print(Fore.BLACK, Back.RED, "WARNING! Column", column, "has", num_missing, "coded missing records!", Style.RESET_ALL)
    if num_blank > 0:        
        print(Fore.BLACK, Back.RED, "WARNING! Column", column, "has", num_blank, "absolutely blank records!", Style.RESET_ALL)

    return


# import main data file
df = pd.read_excel(r'C:\Users\Ibi\Google Drive\School, UMSON\2023.FA, NRSG 795 (Biostatistics for Evidence Based Practice)\Analysis 1\Data Workbook.xlsx', 
                   sheet_name="NRSG795_sum20")

# ------------------------------------------------------------
# DATA VALIDATION
# ------------------------------------------------------------

print("\n")

validate_data("age", 99)
validate_data("gender", 9)
validate_data("marital", 9)
validate_data("workyrs", 99)
validate_data("shift", 9)
validate_data("qol", 99)



# ------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n")

total_records = len(df)
print("Total records:", total_records)

df_age = df.loc[df["age"] != 99].loc[:,"age"]
print("Age: \t\tn =", len(df_age), "\t\tMean: %.1f" % df_age.mean(), "\tMedian: %.1f" % df_age.median(), 
      "\tSt Dev: %.1f" % df_age.std(), "\tRange: ", df_age.min(), "-", df_age.max())

df_workyrs = df.loc[df["workyrs"] != 99].loc[:,"workyrs"]
print("Work Years: \tn =", len(df_workyrs), "\t\tMean: %.1f" % df_workyrs.mean(), "\tMedian: %.1f" % df_workyrs.median(), 
      "\tSt Dev: %.1f" % df_workyrs.std(), "\tRange: ", df_workyrs.min(), "-", df_workyrs.max())

df_qol = df.loc[df["qol"] != 99].loc[:,"qol"]
print("QOL: \t\tn =", len(df_qol), "\t\tMean: %.1f" % df_qol.mean(), "\tMedian: %.1f" % df_qol.median(), 
      "\tSt Dev: %.1f" % df_qol.std(), "\tRange: ", df_qol.min(), "-", df_qol.max())

count_male = len(df.loc[df["gender"] == 0].loc[:,"gender"])
print("Gender: Male: \tn =", count_male, "(%.0f" % (count_male / total_records * 100), "%)")
count_female = len(df.loc[df["gender"] == 1].loc[:,"gender"])
print("Gender: Fale: \tn =", count_female, "(%.0f" % (count_female / total_records * 100), "%)")


count_marital_nm = len(df.loc[df["marital"] == 1].loc[:,"marital"])
print("Marital: Never Married: n =", count_marital_nm, "(%.0f" % (count_marital_nm / total_records * 100), "%)")

count_marital_m = len(df.loc[df["marital"] == 2].loc[:,"marital"])
print("Marital: Married: \tn =", count_marital_m, "(%.0f" % (count_marital_m / total_records * 100), "%)")

count_marital_lwso = len(df.loc[df["marital"] == 3].loc[:,"marital"])
print("Marital: LwSO: \t\tn =", count_marital_lwso, "(%.0f" % (count_marital_lwso / total_records * 100), "%)")

count_marital_sep = len(df.loc[df["marital"] == 4].loc[:,"marital"])
print("Marital: Separated: \tn =", count_marital_sep, "(%.0f" % (count_marital_sep / total_records * 100), "%)")

count_marital_widow = len(df.loc[df["marital"] == 5].loc[:,"marital"])
print("Marital: Widowed: \tn =", count_marital_widow, "(%.0f" % (count_marital_widow / total_records * 100), "%)")

count_marital_div = len(df.loc[df["marital"] == 6].loc[:,"marital"])
print("Marital: Divorced: \tn =", count_marital_div, "(%.0f" % (count_marital_div / total_records * 100), "%)")

count_shift_night = len(df.loc[df["shift"] == 1].loc[:,"shift"])
print("Shift: Nights: \t\tn =", count_shift_night, "(%.0f" % (count_shift_night / total_records * 100), "%)")

count_shift_evening = len(df.loc[df["shift"] == 2].loc[:,"shift"])
print("Shift: Evenings: \tn =", count_shift_evening, "(%.0f" % (count_shift_evening / total_records * 100), "%)")

count_shift_day = len(df.loc[df["shift"] == 3].loc[:,"shift"])
print("Shift: Day: \t\tn =", count_shift_day, "(%.0f" % (count_shift_day / total_records * 100), "%)")

print("\n")

# ------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------





"""
fig, axs = plt.subplots()

# prep data; split quality of life by gender; then retain only qol column
male_qol = df.loc[df["gender"] == 0].loc[:,"qol"]
female_qol = df.loc[df["gender"] == 1].loc[:,"qol"]

# subplot 0: boxplot w/ gender vs qol
axs.boxplot([male_qol, female_qol])
axs.set_xlabel('Gender', fontsize=13)
axs.set_ylabel('Quality of Life', fontsize=13)
axs.set(ylim=(0, 22))
axs.set_yticks([0, 5, 10, 15, 20])
axs.set_xticklabels(["Male", "Female"])
"""


wy_shift_qol = df.loc[df["shift"] != 9].loc[:,["workyrs", "shift", "qol"]]
wy_shift_qol["shift"] = wy_shift_qol["shift"].replace(1, "Night").replace(2, "Evening").replace(3, "Day")

axs = sns.scatterplot(x="workyrs", y="qol", data=wy_shift_qol, hue="shift", palette="tab10", legend=True)
axs.set_xlabel('Years of Work', fontsize=13)
axs.set_ylabel('Quality of Life', fontsize=13)
plt.legend(title="Shift")

#show_plots flag for unmasking console output
show_plots = True

if show_plots == True:
    plt.show()
else:
    print("\nshow_plots is False; plots not shown\n")
