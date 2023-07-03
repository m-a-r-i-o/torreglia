import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression 
import seaborn as sns

def remove_second_header(df):
    if len(df) < 3:  # If there are less than 3 rows, do nothing
        return df
    
    second_row = df.iloc[1]
    third_row = df.iloc[2]

    # Check if all the values in the second row are non-numeric
    second_row_all_nonnumeric = all(not str(x).replace('.', '', 1).isdigit() for x in second_row)
    
    # Check if some values in the third row are numeric
    third_row_some_numeric = any(str(x).replace('.', '', 1).isdigit() for x in third_row)

    if second_row_all_nonnumeric and third_row_some_numeric:
        df = df.drop(df.index[1])
        df.reset_index(drop=True, inplace=True)

    return df

def binrule(u):
    return int(1.5*(1 + np.log(len(u))/np.log(2.0)))

def fit_line_or_poly(x, y, **kwargs):
    """Fit a line or a 2nd degree polynomial, based on the p-value of the 2nd degree coefficient."""
    poly_coef, residuals, rank, singular_values, rcond = np.polyfit(x, y, 2, full=True)
    _, _, r_value, p_value, _ = stats.linregress(x, y)

    x_sorted = sorted(x)

    if p_value < 0.05:  # If the 2nd degree coefficient is significant
        plt.plot(x_sorted, np.poly1d(poly_coef)(x_sorted), color='r', alpha=0.5)  # Plot 2nd degree polynomial
    else:
        plt.plot(x, np.poly1d(poly_coef[1:])(x), color='g', alpha=0.5)  # Plot line

def is_binary(file):
    CHUNKSIZE = 1024
    initial_bytes = file.read(CHUNKSIZE)
    file.seek(0)  # reset file pointer back to the beginning
    return b'\0' in initial_bytes

def sanitize_column_names(df):
    df.columns = df.columns.str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    df.columns = df.columns.str.replace(' ', '_')
    return df

def detect_separator(file):
    first_line = file.readline().decode('utf-8')
    file.seek(0)  # reset file pointer to beginning
    if ',' in first_line:
        return ','
    elif ';' in first_line:
        return ';'
    else:
        return None

def round_to_significant_digits(num, sig_digits):
    """
    Rounds a a number to a specified number of significant digits
    
    Parameters:
    num (float): The number to be rounded.
    sig_digits (int): The number of digits to round to.
    
    Returns:
    float: The rounded number.
    """
    if num != 0:
        return round(num, -int(np.floor(np.log10(abs(num))) - (sig_digits - 1)))
    else:
        return 0  # Can't take the log of 0

def calculate_mutual_info(df):
    """
    Calculate the mutual information between each pair of variables in a DataFrame.
    
    This function calculates the mutual information, a measure of dependency, 
    between each distinct pair of variables in a pandas DataFrame. The function
    assumes that all variables are continuous. It returns a dictionary where
    the keys are tuples representing pairs of variables, and the values are
    the calculated mutual information scores.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame, where each column represents a variable.
    
    Returns:
    dict: A dictionary where the keys are tuples (var1, var2) representing pairs 
    of variables, and the values are the corresponding mutual information scores.
    
    Note:
    This function assumes that all variables in the DataFrame are continuous. If the 
    DataFrame contains categorical variables, they should be suitably encoded before 
    using this function.
    """
    mutual_info_dict = {}
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            mutual_info = mutual_info_regression(df[[cols[i]]], df[cols[j]])
            mutual_info_dict[(cols[i], cols[j])] = mutual_info[0]
    return mutual_info_dict

def get_top_k_pairs(mutual_info_dict, k):
    sorted_pairs = sorted(mutual_info_dict.items(), key=lambda item: item[1], reverse=True)
    return [pair[0] for pair in sorted_pairs[:k]]

def plot_scatter_plots(df, pairs):
    n = len(pairs)
    n = max(2, n)
    fig, axs = plt.subplots(n, figsize=(8, 6*n))  # Set the total figure size and create subplots

    for i, pair in enumerate(pairs):
        sns.regplot(data=df, x=pair[0], y=pair[1], lowess=True, scatter_kws={'color': '#A0A0A0'}, line_kws={'color': '#222222'}, ax=axs[i])
        #sns.regplot(data=df, x=pair[0], y=pair[1], lowess=True, ax=axs[i], color = "#A0A0A0")  # Specify the subplot to draw on
        #sns.scatterplot(data=df, x=pair[0], y=pair[1], ax=axs[i], color = "#A0A0A0")  # Specify the subplot to draw on
        #axs[i].set_title(f'{pair[0]} vs {pair[1]}')

    plt.tight_layout()
    plt.savefig("static/images/mipairsscatter.png", bbox_inches='tight') 
