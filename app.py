from flask import Flask, request, render_template, url_for
import pandas as pd
import numpy as np
from scipy.stats import shapiro, spearmanr
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from diptest import diptest
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helpers import *
import matplotlib as mpl

app = Flask(__name__)

mpl.rcParams['axes.edgecolor'] = '#888888'
mpl.rcParams['axes.labelcolor'] = '#222222'
mpl.rcParams['xtick.color'] = '#555555'
mpl.rcParams['ytick.color'] = '#555555'
mpl.rcParams['text.color'] = '#222222'


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['datafile']
        if file:
            try:
                df = read_input_file(file)
            except Exception as e:
                return render_template('error.html', error_message=str(e))

            sample_df = get_sample_rows(df)
            sampled_df_table = sample_df.to_html(index=False)

            df, init_num_rows, num_rows_no_nan, num_duplicated_rows, na_count, constant_columns, date_columns = preprocessing_df(df)

            # Univariate analysis
            print("Univariate analysis")
            univariate_analysis_0 = {}
            univariate_analysis_1 = {}
            cat_stats = {}
            for column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    print("Testing unimodality")
                    unimodal_p, dip = diptest(df[column])
                    print("Testing normality")
                    normal_p = shapiro(df[column])[1]
                    lognormal_p = 0
                    if (min(df[column]) > 0):
                        print("Testing lognormality")
                        lognormal_p = shapiro(np.log(df[column]))[1]
                    unimodal_verdict = ' '
                    normal_verdict = ' '
                    lognormal_verdict = ' '
                    if(unimodal_p < 0.05):
                        unimodal_verdict = 'No'
                    if(normal_p < 0.05):
                        normal_verdict = 'No'
                    if(lognormal_p < 0.05):
                        lognormal_verdict = 'No'
                    print("Computing summary stats")                    
                    univariate_analysis_0[column] = {
                        'mean': round_to_significant_digits(df[column].mean(), 2),
                        'std': round_to_significant_digits(df[column].std(), 2),
                        'skewness': round_to_significant_digits(skew(df[column]), 2),
                        'kurtosis': round_to_significant_digits(kurtosis(df[column]), 2),
                        'min': round_to_significant_digits(df[column].min(), 2),
                        'median': round_to_significant_digits(df[column].median(), 2),
                        'max': round_to_significant_digits(df[column].max(), 2),
                    }
                    univariate_analysis_1[column] = {
                        'Unimodal': unimodal_verdict,
                        'p_u' : round_to_significant_digits(unimodal_p, 2),
                        'Normal?': normal_verdict,
                        'p_n': round_to_significant_digits(normal_p, 2),
                        'Lognormal?': lognormal_verdict,
                        'p_logn': round_to_significant_digits(lognormal_p, 2)
                    }
            
            df_numeric = df.select_dtypes(include=['int64', 'float64'])
            df_non_numeric = df.select_dtypes(exclude=['int64', 'float64'])
            
            col_summary_table_html = column_summary(df_numeric).to_html()

            cat_bivariate_stats = {}
            df_non_numeric_noID = remove_single_level_columns(df_non_numeric)
            if df_non_numeric_noID.shape[1] > 0:
                barchart_for_categorical_vars(df_non_numeric_noID)
            if df_non_numeric.shape[1] > 0:    
                cat_stats = calculate_categorical_stats(df_non_numeric).to_html(index=False)
                cat_bivariate_stats = categorical_bivariate(df_non_numeric).to_html()

            na_count_nice = na_count[na_count>0]
            if(len(na_count_nice) == 0):
                na_count_nice = ""
            else:
                na_count_nice = na_count_nice.to_dict()
                na_count_nice = ', '.join(f'{k} ({v})' for k, v in na_count_nice.items())

            corr_matrix_0 = {}
            corr_matrix_1 = {}
            if df_numeric.shape[1] > 0:
                print("Bivariate analysis")
                # Bivariate analysis: correlation matrix using Spearman correlation
                print("Computing correlation matrix")
                corr_matrix_0 = (df.corr(method='pearson')).applymap(lambda x: np.round(x, 2)).to_html()
                corr_matrix_1 = (df.corr(method='spearman')).applymap(lambda x: np.round(x, 2)).to_html()

                create_boxplot(df_numeric)

                if not os.path.exists('static/images/histograms'):
                    os.makedirs('static/images/histograms')

                # For each column in the DataFrame, create a histogram and save it as an image.
                print("Creating histograms")
                for column in df_numeric.columns:
                    plt.figure()  # Create a new figure
                    #sns.histplot(df_numeric[column], kde=False, bins = int(1.5*np.sqrt(df_numeric.shape[0])), color='gray', edgecolor=None)  # Create the histogram
                    plt.hist(df_numeric[column], color='#A0A0A0', edgecolor=None, bins=binrule(df_numeric[column]))
                    plt.xlabel(column)
                    plt.ylabel('Counts')
                    plt.savefig(f'static/images/histograms/histogram_{column}.png', bbox_inches='tight')  # Save the figure
                    plt.close()  # Close the figure

                # Calculating mutual information between variables
                try:
                    mutual_info_dict = calculate_mutual_info(df_numeric)
                    k_mi = 10 #maximum number of mutual info plots to display
                    top_pairs = get_top_k_pairs(mutual_info_dict, k_mi)
                    plot_scatter_plots(df, top_pairs)
                except Exception as e:
                    print(str(e))

                #parallel coordinate plot
                if(df_numeric.shape[1] > 3):
                    parallel_coordinate_plot(df_numeric)
                elif(df_numeric.shape[1] == 3):
                    plot_3d_scatter(df_numeric)

            # Create an enhanced pair plot for the numeric variables
            # Create a PairGrid
            #g = PairGrid(df_numeric)

            # Map the custom function to the lower triangle
            #g.map_lower(fit_line_or_poly)

            # Map a histogram to the diagonal
            #g.map_diag(plt.hist)

            # Map a scatterplot to the upper triangle
            #g.map_upper(plt.scatter, s=3)

            #print("Creating pair plot")
            #pair_plot = sns.pairplot(df_numeric)

            # save the figure
            #g.savefig('static/images/pairplot.png')

            if len(constant_columns) == 0:
                constant_columns = ""

            if len(date_columns) == 0:
                date_columns = ""

            #plt.close()
            print("Finishing off")
            return render_template(
                'report.html',
                file=file.filename,
                num_initial_instances=init_num_rows,
                num_instances=df.shape[0],
                num_instances_nan = init_num_rows - num_rows_no_nan,
                num_duplicated_instances = num_duplicated_rows,
                constant_columns = constant_columns,
                date_columns = date_columns,
                num_columns=df.shape[1],
                num_numeric=len(df.select_dtypes(include=['int64', 'float64']).columns),
                num_categorical=len(df.select_dtypes(include=['object']).columns),
                univariate_analysis_0=pd.DataFrame(univariate_analysis_0).transpose().to_html(),
                univariate_analysis_1=pd.DataFrame(univariate_analysis_1).transpose().to_html(),
                columns_with_NAs=na_count_nice,
                bivariate_analysis_0=corr_matrix_0,
                bivariate_analysis_1=corr_matrix_1,                
                num_columns_list=df_numeric.columns.tolist(),
                cat_stats=cat_stats,
                sampled_df_table = sampled_df_table,
                cat_bivariate_stats = cat_bivariate_stats,
                col_summary_table_html = col_summary_table_html
            )
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
