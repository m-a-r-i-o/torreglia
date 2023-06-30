from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
from scipy.stats import shapiro, spearmanr
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

def round_to_significant_digits(num, sig_digits):
    if num != 0:
        return round(num, -int(np.floor(np.log10(abs(num))) - (sig_digits - 1)))
    else:
        return 0  # Can't take the log of 0

def sanitize_column_names(df):
    df.columns = df.columns.str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    df.columns = df.columns.str.replace(' ', '_')
    return df

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['datafile']
        if file:
            df = pd.read_csv(file, na_values=['NA', 'na', 'N/A', 'n/a', 'nan', 'NaN', 'NAN'])
            init_num_rows = df.shape[0]
            # NAs and duplicated
            df_no_nan = df.dropna(axis=0, how='any')
            num_rows_no_nan = df_no_nan.shape[0]

            duplicates = df_no_nan.duplicated()
            num_duplicated_rows = duplicates.sum()

            na_count = df.isna().sum()

            #Sanitize column names
            df = sanitize_column_names(df)

            # Preprocessing: remove NAs
            df.dropna(inplace=True)

            # Univariate analysis
            univariate_analysis = {}
            for column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    normal_p = shapiro(df[column])[1]
                    lognormal_p = 1
                    if (min(df[column]) > 0):
                        lognormal_p = shapiro(np.log(df[column]))[1]
                    normal_verdict = ' '
                    lognormal_verdict = ' '
                    if(normal_p < 0.05):
                        normal_verdict = 'No'
                    if(lognormal_p < 0.05):
                        lognormal_verdict = 'No'                    
                    univariate_analysis[column] = {
                        'mean': round_to_significant_digits(df[column].mean(), 2),
                        'std': round_to_significant_digits(df[column].std(), 2),
                        'skewness': round_to_significant_digits(skew(df[column]), 2),
                        'kurtosis': round_to_significant_digits(kurtosis(df[column]), 2),
                        'min': round_to_significant_digits(df[column].min(), 2),
                        'max': round_to_significant_digits(df[column].max(), 2),
                        'median': round_to_significant_digits(df[column].median(), 2),
                        'IQR': round_to_significant_digits(df[column].quantile(0.75) - df[column].quantile(0.25),2),
                        'Normal?': normal_verdict,
                        'p_n': round_to_significant_digits(normal_p, 2),
                        'Lognormal?': lognormal_verdict,
                        'p_logn': round_to_significant_digits(lognormal_p, 2)
                    }

            # Bivariate analysis: correlation matrix using Spearman correlation
            corr_matrix = (df.corr(method='spearman')).applymap(lambda x: round_to_significant_digits(x, 2)).to_html()

            na_count_nice = na_count[na_count>0]
            if(len(na_count_nice) == 0):
                na_count_nice = "None"
            else:
                na_count_nice = na_count_nice.to_dict()
                na_count_nice = ', '.join(f'{k} ({v})' for k, v in na_count_nice.items())

            # Filter out non-numeric columns
            df_numeric = df.select_dtypes(include=['int64', 'float64'])

            # Reshape the data.
            df_melted = pd.melt(df_numeric)

            # Create the violin plot.
            plt.figure(figsize=(12,6))  
            #sns.violinplot(y="variable", x="value", data=df_melted, inner=None)

            # Overlay a strip plot.
            sns.stripplot(y="variable", x="value", data=df_melted, color="k", jitter=True)

            # Save the figure
            plt.savefig("static/images/violin_plot.png", bbox_inches='tight') 
            plt.close()

            if not os.path.exists('static/images/histograms'):
                os.makedirs('static/images/histograms')

            # For each column in the DataFrame, create a histogram and save it as an image.
            for column in df_numeric.columns:
                plt.figure()  # Create a new figure
                sns.histplot(df_numeric[column], kde=False, bins = int(1.5*np.sqrt(df_numeric.shape[0])), color='gray', edgecolor=None)  # Create the histogram
                plt.savefig(f'static/images/histograms/histogram_{column}.png', bbox_inches='tight')  # Save the figure
                plt.close()  # Close the figure

            return render_template(
                'report.html',
                num_initial_instances=init_num_rows,
                num_instances=df.shape[0],
                num_instances_nan = init_num_rows - num_rows_no_nan,
                num_duplicated_instances = num_duplicated_rows,
                num_columns=df.shape[1],
                num_numeric=len(df.select_dtypes(include=['int64', 'float64']).columns),
                num_categorical=len(df.select_dtypes(include=['object']).columns),
                univariate_analysis=pd.DataFrame(univariate_analysis).transpose().to_html(),
                columns_with_NAs=na_count_nice,
                bivariate_analysis=corr_matrix,
                num_columns_list=df_numeric.columns.tolist()  # add this line
            )
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
