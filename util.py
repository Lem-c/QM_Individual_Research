# This is assist methods file
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.genmod.families import NegativeBinomial

import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_partregress_grid
from scipy import stats

# The dot size used for plotting
dot_size = 1


def format_read_csv(file_: str, sheet_id=1, header_id=0) -> pd.DataFrame:
    file_type = os.path.splitext(file_)[1]

    if file_type == ".xlsx" or file_type == ".xls":
        table_df = pd.read_excel(file_, sheet_name=sheet_id, header=header_id)
    else:
        table_df = pd.read_csv(file_)

    if table_df is None:
        raise FileNotFoundError("Read in xls file fail")

    return table_df


def linear_predict_model(X_train, X_test, y_train, y_test):
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_prediction = model.predict(X_test)
    # Evaluating the model
    mse = mean_squared_error(y_test, y_prediction)
    r2 = r2_score(y_test, y_prediction)
    # Model coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_

    print(f'The MSE is: {mse:.3f} and R-squared is: {r2:.3f}')
    print(f'The coefficient(s) is(are): {coefficients} and intercept is: {intercept}')
    return y_prediction


def simple_linear_regression(df_: pd.DataFrame, col_X: str, col_y: str,
                             isPolynomial=False,
                             polynomialDegree=2,
                             is_plot=True):
    if len(col_X) < 2 or len(col_y) < 1:
        print(col_X, col_y)
        raise Exception("Can't handle null column name to a dataset")

    df_[col_X] = pd.to_numeric(df_[col_X], errors='coerce')
    df_[col_y] = pd.to_numeric(df_[col_y], errors='coerce')

    df_ = df_.dropna(subset=[col_X, col_y])
    for col in df_.columns:
        if df_[col].dtype in ['float64', 'int64']:  # Apply only to numeric columns
            df_ = remove_outliers(df_, col)

    df_X = df_[[col_X]].to_numpy()
    df_y = df_[col_y].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.1, train_size=0.9, random_state=2023)

    if isPolynomial:
        # Reshaping data for the model
        df_y = df_y[:, np.newaxis]
        # Transforming the data to include another axis
        polynomial_features = PolynomialFeatures(degree=polynomialDegree)
        X_poly = polynomial_features.fit_transform(df_X)
        # Not split the dataset
        X_train = X_poly
        X_test = X_poly
        y_train = df_y
        y_test = df_y

    y_prediction = linear_predict_model(X_train, X_test, y_train, y_test)

    if is_plot and not isPolynomial:
        plot_reg(X_test, y_test, y_prediction)
    else:
        # Combine X and Y into a single array for sorting
        combined = np.column_stack((df_X, y_prediction))
        # Sort the array by the first column (X)
        sorted_combined = combined[np.argsort(combined[:, 0])]
        # Extract the sorted X and Y values
        X_sorted = sorted_combined[:, 0]
        Y_sorted = sorted_combined[:, 1]
        plot_reg(X_sorted, df_y, Y_sorted)


def support_vector_regression(df, col_X: str, col_y: str):
    cleaned_data = df.dropna()

    # Defining the independent X and dependent y variables
    X = cleaned_data[[col_X]]
    y = cleaned_data[col_y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=2024)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)  # 'mle' automatically selects the number of components
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = SVR(kernel='rbf')
    model.fit(X_train_pca, y_train)
    y_prediction = model.predict(X_test_pca)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_prediction)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")

    plot_reg(X_test, y_test, y_prediction)


def multi_linear_regression(df, col_X: list, col_y: str):
    print("----Multi-Linear-Regression----")
    cleaned_data = df.dropna()

    # Defining the independent X and dependent y variables
    X = cleaned_data[col_X].to_numpy()
    y = cleaned_data[col_y].to_numpy()

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42)

    linear_predict_model(X_train, X_test, y_train, y_test)


def stats_linear_regression(df, col_X: str, col_y: str,
                            x_axis: str, y_axis: str, title: str,
                            isVarify=True):
    print("----Stats-model-Regression----")
    cleaned_data = df.dropna()

    # Defining the independent X and dependent y variables
    X = cleaned_data[col_X].to_numpy()
    y = cleaned_data[col_y].to_numpy()
    # Add a constant to the independent variable (for the intercept term)
    X_with_constant = sm.add_constant(X)

    # Fit a linear regression model
    model = sm.OLS(y, X_with_constant).fit()
    # Summary of the regression model
    model_summary = model.summary()

    print(model_summary)

    if not isVarify:
        return

    # Extract the residuals
    residuals = model.resid

    # Normality test on the residuals (using the Jarque-Bera test)
    jb_test = stats.jarque_bera(residuals)
    # Homoscedasticity test (using the Breusch-Pagan test)
    bp_test = sm.stats.diagnostic.het_breuschpagan(residuals, X_with_constant)
    print(jb_test, "\n")
    print(bp_test)

    # Plotting residuals to visually inspect for homoscedasticity
    plt.figure(figsize=(10, 6))
    plt.scatter(model.fittedvalues, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.show()


def stats_mutil_linear_regression(df, indi_param_list: list, col_y: str, isVarify=True):

    print("----Multi-Linear-Regression----")
    cleaned_data = df.dropna()

    # Defining the independent X and dependent y variables
    X = cleaned_data[indi_param_list]
    y = cleaned_data[col_y]

    # Assuming `X` is a DataFrame with multiple columns and `y` is the target series

    model = sm.OLS(y, sm.add_constant(X)).fit()
    print(model.summary())

    # Specify the number of columns in the grid
    fig = plt.figure(figsize=(8, 6))
    plot_partregress_grid(model, fig=fig)
    plt.tight_layout()
    plt.show()

    if not isVarify:
        return

    # Extract the residuals
    residuals = model.resid

    # Plotting residuals to visually inspect for homoscedasticity
    plt.figure(figsize=(10, 6))
    plt.scatter(model.fittedvalues, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Fitted model value")
    plt.ylabel("Residual")
    plt.title("Residual vs. Fitted Plot")
    plt.show()


def get_bins_by_value(val_: float, min_: float, max_: float, level: int):
    dif_min_max = int(max_ - min_)
    if dif_min_max < 0:
        exit(-1)
    step = dif_min_max / level

    for i in range(level):
        prev = min_ + step * i
        lstv = min_ + step * (i+1)

        if lstv >= max_:
            return level

        if prev <= val_ <= lstv:
            return i
        else:
            continue

    return -1


def auto_encoding_df(df, col: str, interval: int):
    df_min = np.min(df[col])
    df_max = np.max(df[col])

    result_col = []

    for val_ in df[col]:
        result_col.append(get_bins_by_value(val_, df_min, df_max, interval))

    df.loc[:, col] = result_col


def stats_neg_binom_reg(df, indi_param_list: list, col_y: str):
    print("----neg_binom-Regression----")
    cleaned_data = df.dropna()

    # Defining the independent X and dependent y variables
    X = cleaned_data[indi_param_list]
    X = sm.add_constant(X)
    y = cleaned_data[col_y]

    neg_binom_model = sm.GLM(y, X, family=NegativeBinomial())

    # Fit the model
    neg_binom_results = neg_binom_model.fit()

    # Print the results
    print(neg_binom_results.summary())


def remove_outliers(df, column):
    """
    Remove outliers using Inter-quartile Range (IQR)
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def sum_table_by_year(df_, col_year="YearMonth"):
    """
    Can only be used after pivoted by year
    """

    # Grouping the data by 'LookUp_BoroughName' and summing up all the monthly crime counts
    df_ = df_.groupby([col_year]).sum()
    # Resetting the index to have 'MajorText' as a column
    df_.reset_index(inplace=True)
    return df_


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def logmod(value):
    return np.log(value)


def sig_process_data(df_, mod='sigmoid'):
    """
    @TODO
    :param df_: A column of a dataset | Normally pd.DataFrame
    :param mod: Apply what kind of process to the data
    :return: Normalized data
    """
    # Convert data to a numpy array for easier manipulation
    data = df_
    normalized_data = None

    if mod == 'sigmoid':
        # Optionally, standardize the data first
        data_standardized = (data - np.mean(data)) / np.std(data)

        # Apply the sigmoid function
        normalized_data = sigmoid(data_standardized)
    elif mod == 'log':
        normalized_data = logmod(data)

    return normalized_data


def plot_trend(df_, col_year='YearMonth', col_data='newDailyNsoDeathsByDeathDate',
               title='Chart Title', is_bar=False):
    if df_ is None:
        raise Exception("Null object find: 'df_'")

    # Extracting YearMonth for the x-axis
    x = df_[col_year]

    # Plotting line graphs for each borough
    plt.figure(figsize=(15, 6))
    if is_bar:
        plt.bar(x, df_[col_data], align="center", color="steelblue", alpha=0.6)
    else:
        plt.plot(x, df_[col_data])

    # Adding labels and title
    plt.xlabel(col_year)
    plt.ylabel('Y')
    plt.title(title)
    plt.xticks(rotation=45)  # Rotating x-axis labels for better readability
    # plt.legend()
    plt.show()


def plot_correlation_matrix(df, title='Correlation Matrix'):
    plt.rcParams["axes.grid"] = False
    f = plt.figure(figsize=(19, 15))

    plt.matshow(df.corr(), fignum=f.number)

    self_define = ["death", "vaccination_2", "hospitalCases", "positiveTest"]

    plt.xticks(range(df.shape[1]), self_define, fontsize=14, rotation=90)
    plt.yticks(range(df.shape[1]), self_define, fontsize=14)

    cb = plt.colorbar()

    cb.ax.tick_params(labelsize=14)
    plt.title(title, fontsize=16)

    plt.show()


def plot_bar_chart(df_, borough: str, col_year='YearMonth'):
    if df_ is None:
        raise Exception("Null object find: 'df_'")

    # Extracting YearMonth for the x-axis
    x = df_[col_year]

    plt.bar(x, df_[borough])
    plt.xlabel(col_year)
    plt.ylabel('Values')
    plt.xticks(rotation=45)  # Rotating x-axis labels for better readability
    plt.show()


def plot_reg(X_test, y_test, y_prediction):
    # Plot outputs with scatter and line
    plt.scatter(X_test, y_test, s=dot_size, label="Original Data", color="black")
    plt.plot(X_test, y_prediction, color="m")
    plt.xlabel("Independent variable (X)")
    plt.ylabel("Dependent variable (Y)")

    plt.show()


def plot_hist(df, x: str,
              title: str, x_axis: str):
    # Histogram view
    plt.figure(figsize=(12, 8))
    plt.hist(df[x], bins=30, color="#FF4500", alpha=0.7)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_scatter(df, col_x: str, col_y: str, point_size=5):
    # Create a scatter plot with specified point size
    plt.scatter(df[col_x], df[col_y], s=point_size)

    # Adding title and labels (optional)
    plt.title('Scatter Plot of x vs y with Custom Point Size')
    plt.xlabel('x')
    plt.ylabel('y')

    # Show the plot
    plt.show()
