from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
from scipy.stats import shapiro, spearmanr
from scipy.stats import skew, kurtosis

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
            df = pd.read_csv(file)
            df = sanitize_column_names(df)

            # NAs and duplicated
            df_no_nan = df.dropna()
            num_rows_no_nan = len(df_no_nan)

            duplicates = df.duplicated()
            num_duplicated_rows = duplicates.sum()

            # Preprocessing: remove NAs
            df.dropna(inplace=True)

            # Univariate analysis
            univariate_analysis = {}
            for column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    univariate_analysis[column] = {
                        'mean': round_to_significant_digits(df[column].mean(), 2),
                        'std_dev': round_to_significant_digits(df[column].std(), 2),
                        'skewness': round_to_significant_digits(skew(df[column]), 2),
                        'kurtosis': round_to_significant_digits(kurtosis(df[column]), 2),
                        'min': round_to_significant_digits(df[column].min(), 2),
                        'max': round_to_significant_digits(df[column].max(), 2),
                        'median': round_to_significant_digits(df[column].median(), 2),
                        'iqr': round_to_significant_digits(df[column].quantile(0.75) - df[column].quantile(0.25),2),
                        'shapiro_test_p_value': round_to_significant_digits(shapiro(df[column])[1], 2),
                    }

            # Bivariate analysis: correlation matrix using Spearman correlation
            corr_matrix = (df.corr(method='spearman')).applymap(lambda x: round_to_significant_digits(x, 2)).to_html()

            return render_template(
                'report.html',
                num_instances=df.shape[0],
                num_instances_not_nan = num_rows_no_nan,
                num_duplicated_instances = num_duplicated_rows,
                num_columns=df.shape[1],
                num_numeric=len(df.select_dtypes(include=['int64', 'float64']).columns),
                num_categorical=len(df.select_dtypes(include=['object']).columns),
                univariate_analysis=pd.DataFrame(univariate_analysis).transpose().to_html(),
                bivariate_analysis=corr_matrix
            )
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
