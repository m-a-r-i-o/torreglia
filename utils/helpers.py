import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression 
import seaborn as sns
from scipy.interpolate import make_interp_spline
from lingam.direct_lingam import DirectLiNGAM
from scipy.stats import entropy
import re
from dateutil.parser import parse
from scipy.stats import chi2_contingency, fisher_exact
from itertools import combinations

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def categorical_bivariate(categorical_df):
    columns = categorical_df.columns
    corr_data = []

    for col1, col2 in combinations(columns, 2):
        contingency_table = pd.crosstab(categorical_df[col1], categorical_df[col2])
        min_value = contingency_table.values.min()

        _, p_val, _, _ = chi2_contingency(contingency_table)
        test_used = 'Chi-Squared'
    
        cramers_v_val = cramers_v(categorical_df[col1], categorical_df[col2])

        corr_data.append([col1, col2, test_used, p_val, cramers_v_val])

    correlation_df = pd.DataFrame(corr_data, columns=['Variable 1', 'Variable 2', 'Test Used', 'p-value', 'Cramér\'s V'])
    return correlation_df

def get_sample_rows(df):
    ksample = 3
    # Get first and last five rows
    first_rows = df.head(ksample)
    last_rows = df.tail(ksample)

    # Make sure there are more than 10 rows, otherwise just return the dataframe
    if len(df) <= 3*ksample:
        return df

    # Exclude first and last five rows when sampling
    middle_rows = df.iloc[ksample:-ksample].sample(ksample)

    # Create a separator row
    separator = pd.DataFrame({col: '...' for col in df.columns}, index=[0])

    # Concatenate the dataframes
    sample_df = pd.concat([first_rows, separator, middle_rows, separator, last_rows], ignore_index=True)

    return sample_df


def read_input_file(file):
    if is_binary(file):
        return render_template('error.html', error_message="The file appears to be binary, not csv")
    print("Detecting csv separator")
    sep = detect_separator(file)
    print("Reading in the file")
    df = pd.read_csv(file, sep=sep, skipinitialspace=True, na_values=['NA', 'na', 'N/A', 'n/a', 'nan', 'NaN', 'NAN'])
    df = df.applymap(lambda x: x.strip() if type(x) is str else x) #strip trailing spaces between commas
    #df = remove_second_header(df)
    return(df)

def is_date(string):
	# First, check if the string is a string data type
    if not isinstance(string, str):
        return False

    date_formats = ["\d{1,2}-\d{1,2}-\d{4}", "\d{4}-\d{1,2}-\d{1,2}", "\d{1,2}:\d{1,2}:\d{4}"]

    if any(re.match(pattern, string) for pattern in date_formats):
        try:
            parse(string)
            return True
        except ValueError:
            return False
    else:
        return False



def is_date_column(df, column):
    non_na_values = df[column].dropna()
    if non_na_values.empty:
        return False
    return non_na_values.apply(is_date).all()

def get_date_columns(df):
    date_columns = [col for col in df.columns if is_date_column(df, col)]
    return date_columns


def calculate_categorical_stats(df, top_n=20):
    stats = []
    categorical_columns = df.columns

    for col in categorical_columns:
        freq = df[col].value_counts().head(top_n)
        rel_freq = df[col].value_counts(normalize=True)
        mode = df[col].mode().iloc[0]
        entropy_value = entropy(rel_freq)
        rel_freq = rel_freq.head(top_n)

        stats.append([col, freq.to_string(header=None).replace('\n', '; '), rel_freq.to_string(header=None).replace('\n', '; '), mode, entropy_value])

    stats_df = pd.DataFrame(stats, columns=['Variable', 'Counts', 'Relative Frequencies', 'Mode', 'Entropy'])

    return stats_df 

def find_constant_columns(df):
    constant_columns = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_columns.append(col)
    return constant_columns

def barchart_for_categorical_vars(df, top_n=20):
    cat_cols = df.columns
    fig, axs = plt.subplots(len(cat_cols), 1, figsize=(10, 5*len(cat_cols)))

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    for i, col in enumerate(cat_cols):
        # Count the column data
        col_count = df[col].value_counts()
        # Keep only top N values
        top_values = col_count[:top_n]
        # Replace the less frequent with 'Other'
        df_tmp = df[col].replace({idx: 'Other' for idx in col_count.index[top_n:]})
        col_count = df_tmp.value_counts()
        # Draw bar plot (horizontal)
        axs[i].barh(col_count.index, col_count.values, color='#A0A0A0')
        axs[i].set_title(f'Distribution of {col}')

    plt.tight_layout()
    plt.savefig("static/images/barchart.png", bbox_inches='tight') 
    plt.close()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_scatter(df):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    z = df.iloc[:, 2]

    ax.scatter(x, y, z, color = "#A0A0A0")

    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_zlabel(df.columns[2])

    plt.savefig("static/images/parcoord.png", bbox_inches='tight') 
    plt.close()

def create_boxplot(df):
    print("Creating boxplot")

    # Reshape the data.
    df_melted = pd.melt(df)

    # Create the box plot.
    plt.figure(figsize=(12,6))  
    sns.boxplot(y="variable", x="value", data=df_melted, color="#A0A0A0")

    # Save the figure
    plt.savefig("static/images/violin_plot.png", bbox_inches='tight') 
    plt.close()

def parallel_coordinate_plot(df):
    # Assume df is your DataFrame
    df = df.astype(float)  # Make sure all data is float
    df_norm = (df - df.min()) / (df.max() - df.min())  # Normalize the data

    x = range(df.shape[1])  # x coordinates for each column

    plt.figure(figsize=(10, 8))

    for i in range(df_norm.shape[0]):
        y = df_norm.iloc[i, :]  # y coordinates for each row

        # Make a spline interpolation
        spl = make_interp_spline(x, y, k=3)  # type: BSpline
        xnew = np.linspace(min(x), max(x), 500)  
        ynew = spl(xnew)
        plt.plot(xnew, ynew, alpha=0.08, color="#222222")

    for i in x:
        plt.axvline(x=i, linestyle='dotted', color='#888888')

    plt.xticks(x, df.columns, rotation=90)
    plt.ylabel('Value (scaled to [0,1])')

    plt.savefig("static/images/parcoord.png", bbox_inches='tight') 
    plt.close()

def preprocessing_df(df):
    init_num_rows = df.shape[0]
    print("Computing NA statistics")

    date_columns = get_date_columns(df)

    # NAs and duplicated
    df_no_nan = df.dropna(axis=0, how='any')
    num_rows_no_nan = df_no_nan.shape[0]

    duplicates = df_no_nan.duplicated()
    num_duplicated_rows = duplicates.sum()

    na_count = df.isna().sum()

    #Sanitize column names
    print("Sanitizing column names")
    df = sanitize_column_names(df)

    constant_columns = find_constant_columns(df)
    df = df.drop(columns=constant_columns)

    # Preprocessing: remove NAs
    print("Removing NAs")
    df.dropna(inplace=True)

    return df, init_num_rows, num_rows_no_nan, num_duplicated_rows, na_count, constant_columns, date_columns

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
    fig, axs = plt.subplots(n, figsize=(8, 6*n))  # Set the total figure size and create subplots

    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    for i, pair in enumerate(pairs):
        # Prepare data for DirectLiNGAM
        X = df[list(pair)].values

        # Instantiate and fit DirectLiNGAM
        model = DirectLiNGAM()
        model.fit(X)

        # Get the causal ordering
        causal_order = model.causal_order_

        # Check if the first variable is the cause (smaller index in causal_order)
        if causal_order[0] < causal_order[1]:
            axs[i].scatter(df[pair[0]], df[pair[1]], color = "#A0A0A0")
            axs[i].set_xlabel(pair[0])
            axs[i].set_ylabel(pair[1])
        else:
            axs[i].scatter(df[pair[1]], df[pair[0]], color = "#A0A0A0")
            axs[i].set_xlabel(pair[1])
            axs[i].set_ylabel(pair[0])

    plt.tight_layout()
    plt.savefig("static/images/mipairsscatter.png", bbox_inches='tight') 
