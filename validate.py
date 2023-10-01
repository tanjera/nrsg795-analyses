import pandas as pd
from colorama import Fore, Back, Style

def validate_data(df, column, missing_code, to_exit: bool = True):
    num_missing = len(df.loc[df[column] == missing_code].loc[:,column])
    num_blank = len(df.loc[df[column] == ''].loc[:,column])

    if num_missing > 0:
        print(Fore.BLACK, Back.RED, "WARNING! Column", column, "has", num_missing, "coded missing records!", Style.RESET_ALL)
    if num_blank > 0:
        print(Fore.BLACK, Back.RED, "WARNING! Column", column, "has", num_blank, "absolutely blank records!", Style.RESET_ALL)
    return
