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
import math
from flask import session
from scipy.stats import mode
from scipy.stats import chisquare
import statsmodels.api as sm
from scipy import interpolate
from sklearn.metrics import r2_score 
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score

def clustering(df):
    df_numeric = df.select_dtypes(include=[np.number])
    scaler = RobustScaler()
    df_numeric_scaled = scaler.fit_transform(df_numeric)
    maxclu = min(df.shape[0], 11)
    ra = range(2, maxclu)
    range_n_clusters = list(ra)  # Adjust as needed
    best_num_clusters = 0
    best_silhouette_score = -1
    silhouette_scores = []
    for n_clusters in range_n_clusters:
        # Fit the KMedoids model
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=0).fit(df_numeric_scaled)
        # Compute the silhouette score
        silhouette_avg = silhouette_score(df_numeric_scaled, kmedoids.labels_)
        silhouette_scores.append(silhouette_avg)
        # Check if this silhouette score is better than the current best
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_num_clusters = n_clusters
    # Now compute KMedoids with the best number of clusters 
    kmedoids = KMedoids(n_clusters=best_num_clusters, random_state=0).fit(df_numeric_scaled)
    # Assign the labels to a new column in your DataFrame
    categories = list(df_numeric.columns)
    df_numeric['cluster'] = kmedoids.labels_
    # Get the medoids
    medoids = df_numeric.iloc[kmedoids.medoid_indices_, :]
    medoids_scaled = pd.DataFrame(df_numeric_scaled).iloc[kmedoids.medoid_indices_, :]

    plt.figure()
    plt.plot(ra, silhouette_scores, color = "#888888")
    plt.xlabel("Number of clusters")
    plt.ylabel("Average silhouette width")
    upload_id = session.get("upload_id")
    plt.savefig(f"static/images/{upload_id}_sil.png", bbox_inches='tight') 
    plt.close()

    # Plotting code
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    fig, axs = plt.subplots(1, best_num_clusters, figsize=(10, 6), subplot_kw=dict(polar=True))
    fig.subplots_adjust(wspace=0.3)

    for i, ax in enumerate(axs.flatten()):
        values = medoids_scaled.iloc[i].values.flatten().tolist()
        values += values[:1] # Repeat the first value to close the circular graph
        ax.fill(angles, values, color='#A0A0A0')
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        #ax.set_title(f'Medoid {i+1}')

    plt.savefig(f"static/images/{upload_id}_medoids.png", bbox_inches='tight') 
    plt.close()

    return medoids, ra, np.array(silhouette_scores)

def perform_second_analysis(df):
    return clustering(df)

def benford_law(series):
    # Calculate Benford's frequency for digits 1-9
    benford_freq = np.array([np.log10(1 + 1/digit) for digit in range(1, 10)])

    # Ensure the series contains finite values, is not zero, and is not null
    valid_values = series[np.isfinite(series) & (series != 0)]

    # Extract first digit of each value
    first_digits = valid_values.dropna().apply(lambda x: int(str(abs(x))[0]))
    
    # Calculate observed frequency of first digits
    observed_freq = first_digits.value_counts(normalize=True).sort_index()

    # Ensure that observed_freq has the same number of elements as benford_freq
    observed_freq_reindexed = observed_freq.reindex(range(1, 10), fill_value=0)

    # Perform chi-square test
    chi_stat, p_val = chisquare(observed_freq_reindexed, f_exp=benford_freq)
    
    # Returns True if the distribution is not significantly different from Benford's law
    return p_val > 0.05



def detect_high_frequency_values(df, threshold=0.01):
    high_freq_cols = []
    n = len(df)
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Count unique values in the column
            value_counts = df[col].value_counts()
            # Find the frequency of the most common value
            max_count = value_counts.max()
            max_freq = max_count / n
            if (max_freq > threshold) and (max_count > 10):
                max_val = value_counts.idxmax()
                high_freq_cols.append((col, max_val, max_freq))
    return high_freq_cols

def detect_rounded_values(df, num_bins=10):
    rounded_cols = []
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            # Extract fractional parts
            fractional_parts = df[col] % 1
            # Compute histogram
            hist, bin_edges = np.histogram(fractional_parts, bins=num_bins)
            # If histogram of fractional parts is approximately uniform, it might be rounded
            if np.std(hist) < len(df) / (4*num_bins):  # some threshold to determine 'uniformity'
                rounded_cols.append(col)
        elif df[col].dtype in ['int64', 'int32']:  # checks if the data type is integer
            rounded_cols.append(col)
    return rounded_cols


def detect_almost_identical_cols(df, threshold=1e-10):
    identical_cols = []
    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2 and all(abs(df[col1]-df[col2]) < threshold):
                identical_cols.append((col1, col2))
    return identical_cols

def column_summary(df):
    almost_identical_cols = detect_almost_identical_cols(df)
    rounding_cols = detect_rounded_values(df)
    high_freq_cols = detect_high_frequency_values(df)

    summary_df = pd.DataFrame(index=df.columns, columns=["Repeated value", "Repetition frequency", "Fraction NAs", "Almost identical to", "Rounded", 'Benford\'s Law'])

    for col in df.columns:
        summary_df.loc[col, "Fraction NAs"] = df[col].isna().mean()

        high_freq = [x for x in high_freq_cols if x[0] == col]
        if high_freq:
            summary_df.loc[col, "Repeated value"] = high_freq[0][1]
            summary_df.loc[col, "Repetition frequency"] = high_freq[0][2]
        else:
            summary_df.loc[col, "Repeated value"] = ''
            summary_df.loc[col, "Repetition frequency"] = ''           
        
        identical_cols = [x for x in almost_identical_cols if x[0] == col]
        if identical_cols:
            summary_df.loc[col, "Almost identical to"] = ''.join(identical_cols[0][1])
        else:
            summary_df.loc[col, "Almost identical to"] = ''

        if col in rounding_cols:
            summary_df.loc[col, "Rounded"] = 'Yes'
        else:
            summary_df.loc[col, "Rounded"] = ''

        # Fill in the 'Benford's Law' column
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            summary_df.at[col, 'Benford\'s Law'] = 'Yes' if benford_law(df[col]) else 'No'


    return summary_df


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
        nlevels = len(np.unique(df[col]))

        maximum_entropy = np.log(nlevels)
        if freq[0] == 1:
            is_name_or_ID = 'Yes'
            stats.append([col, nlevels, '', '', '', '', '', is_name_or_ID])
        else:
            is_name_or_ID = ''
            stats.append([col, nlevels, freq.to_string(header=None).replace('\n', '; '), rel_freq.to_string(header=None).replace('\n', '; '), mode, entropy_value, maximum_entropy, is_name_or_ID])

        stats_df = pd.DataFrame(stats, columns=['Variable', 'Levels', 'Counts', 'Relative Frequencies', 'Mode', 'Entropy', 'Max. Entropy', 'Name/ID?'])

    return stats_df 

def find_constant_columns(df):
    constant_columns = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_columns.append(col)
    return constant_columns

def remove_single_level_columns(df):
    single_level_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    df = df.drop(columns=single_level_cols)
    return df

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
    upload_id = session.get("upload_id")
    plt.savefig(f"static/images/{upload_id}_barchart.png", bbox_inches='tight') 
    plt.close()

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

    upload_id = session.get("upload_id")
    plt.savefig(f"static/images/{upload_id}_parcoord.png", bbox_inches='tight') 
    plt.close()

def create_boxplot(df):
    print("Creating boxplot")

    # Calculate ranges and sort columns by them
    var_ranges = df.max() - df.min()
    sorted_columns = var_ranges.sort_values().index

    # Initialize an empty list for storing sub-dataframes (each representing a subplot)
    dfs = []
    # Start the first group with the column of smallest range
    current_group = [sorted_columns[0]]

    for col in sorted_columns[1:]:
        # If adding a column to the current group wouldn't increase the range of the group more than 5 times
        newrange = max(df[col].max(),df[current_group].max().max()) - min(df[col].min(), df[current_group].min().min())
        oldrange = (df[current_group].max().max() - df[current_group].min().min())
        if newrange < 5 * oldrange:
            # Add it to the current group
            current_group.append(col)
        else:
            # Else start a new group with this column
            dfs.append(df[current_group])
            current_group = [col]

    # Don't forget to add the last group
    dfs.append(df[current_group])

    # Create a subplot for each group
    fig, axes = plt.subplots(len(dfs), 1, figsize=(12, 6 * len(dfs)))

    # If only one subplot, axes won't be an array; convert it to array for consistency
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, df in zip(axes, dfs):
        df_melted = pd.melt(df)
        sns.boxplot(y="variable", x="value", data=df_melted, color="#A0A0A0", ax=ax, width=0.1)
        # Increase the font size of the labels
        ax.tick_params(axis='x', labelsize=20)  # Change the font size of x-axis labels
        ax.tick_params(axis='y', labelsize=14)  # Change the font size of y-axis labels
        ax.set_ylabel("")
        ax.set_xlabel("value", fontsize=18)        
        # Optionally, you can also increase the font size of the tick labels
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    # Save the figure
    plt.tight_layout()
    upload_id = session.get("upload_id")
    plt.savefig(f"static/images/{upload_id}_violin_plot.png", bbox_inches='tight')
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

    upload_id = session.get("upload_id")
    plt.savefig(f"static/images/{upload_id}_parcoord.png", bbox_inches='tight') 
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

def plot_scatter_plots(df, pairs, threshold = 0.8):
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
            x = df[pair[0]]
            y = df[pair[1]]
            axs[i].set_xlabel(pair[0])
            axs[i].set_ylabel(pair[1])
        else:
            x = df[pair[1]]
            y = df[pair[0]]
            axs[i].set_xlabel(pair[1])
            axs[i].set_ylabel(pair[0])
        axs[i].scatter(x, y, color = "#A0A0A0") 
        X = sm.add_constant(x)
        model = sm.OLS(y, X)
        results = model.fit()
        # Determine how good the regression is
        r_squared = r2_score(y, results.predict(X))
        print("Regressing")
        # Check if the fit is good enough
        if r_squared >= threshold:
            print("if")
            # If it is good enough, overplot the regression line
            axs[i].plot(x, results.predict(X), color='#222222', label='Linear Regression')

            # Add equation as legend
            axs[i].text(0.05, 0.95, 'y = {:.2f}x + {:.2f}, R² = {:.2f}'.format(results.params[1], results.params[0], r_squared),
            transform=axs[i].transAxes, verticalalignment='top')


    plt.tight_layout()
    upload_id = session.get("upload_id")
    plt.savefig(f"static/images/{upload_id}_mipairsscatter.png", bbox_inches='tight') 
