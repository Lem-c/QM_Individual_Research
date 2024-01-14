"""
@ Author    : Lem Chen
@ Time      : 23/11/2023
"""

import NumericDataFrame as myDF
import pandas as pd
import util

# data from: https://coronavirus.data.gov.uk/details/download
dose_taken_data_url = "./data/daily_first_dose.csv"
death_data_url = "./data/deathDate.csv"
cases_data_url = "./data/hospital_cases_2023-12-14.csv"
PCR_test_data_url = "./data/testingData.csv"


if __name__ == "__main__":
    vac_dose_data_obj = myDF.NDF(dose_taken_data_url, death_data_url)
    vac_dose_data_obj.insert_df(cases_data_url)
    vac_dose_data_obj.insert_df(PCR_test_data_url)
    # vac_dose_data_obj.merge_reg_df(vac_dose_data_obj.multi_df[2], 'date', 'date')
    # vac_dose_data_obj.sum_multi_data()

    # Join the data by date
    vac_dose_data_obj.simple_join(0, 1, 'date', 'date', is_daily=False)

    # util.plot_hist(vac_dose_data_obj.multi_df[0], "cumDailyNsoDeathsByDeathDate",
    #                "Histogram of Cumulative Reported Death Count", "Cumulative Reported Death Count")
    # util.plot_hist(vac_dose_data_obj.multi_df[1], "cumPeopleVaccinatedFirstDoseByVaccinationDate",
    #                "Histogram of Cumulative First Dose Vaccinations", "Cumulative First Dose Vaccinations")
    # util.plot_hist(vac_dose_data_obj.multi_df[2], "hospitalCases",
    #                "Histogram of Hospital Infected Patient Cases", "Hospital Infected Patient Cases")
    # util.plot_hist(vac_dose_data_obj.multi_df[3], "uniqueCasePositivityBySpecimenDateRollingSum",
    #                "Histogram of Positive PCR Test Statistical Data",
    #                "Unique Case Positivity by Specimen Date Rolling Sum")

    # Data visualization
    vac_dose_data_obj.reg_data_sight('date', 'cumDailyNsoDeathsByDeathDate', 'Daily death',
                                     'date', 'cumPeopleVaccinatedFirstDoseByVaccinationDate', 'Daily first vac dose')
    vac_dose_data_obj.plot_box_reg('cumPeopleVaccinatedFirstDoseByVaccinationDate',
                                   'cumDailyNsoDeathsByDeathDate',
                                   'Distribution of Number of Vaccinations Delivered',
                                   'Distribution of Sum Daily NSO Deaths By Death Date')

    util.plot_scatter(vac_dose_data_obj.reg_df_,
                      'cumDailyNsoDeathsByDeathDate', 'cumPeopleVaccinatedFirstDoseByVaccinationDate',
                      point_size=2)

    # Data normalization
    # log is better than sigmoid
    vac_dose_data_obj.normalize_colum('cumPeopleVaccinatedFirstDoseByVaccinationDate', mod='log')
    vac_dose_data_obj.normalize_colum('cumDailyNsoDeathsByDeathDate', mod='log')

    vac_dose_data_obj.reg_df_ = util.remove_outliers_iqr(vac_dose_data_obj.reg_df_,
                                                         "cumDailyNsoDeathsByDeathDate",
                                                         "cumPeopleVaccinatedFirstDoseByVaccinationDate")

    # Simple linear regression
    util.simple_linear_regression(vac_dose_data_obj.reg_df_,
                                  'cumDailyNsoDeathsByDeathDate', 'cumPeopleVaccinatedFirstDoseByVaccinationDate',
                                  isPolynomial=False, polynomialDegree=2)

    util.stats_linear_regression(vac_dose_data_obj.reg_df_,
                                 'cumDailyNsoDeathsByDeathDate', 'cumPeopleVaccinatedFirstDoseByVaccinationDate',
                                 'death', 'doseTake', 'Residual vs. Fitted Plot')

    util.support_vector_regression(vac_dose_data_obj.reg_df_, 'cumDailyNsoDeathsByDeathDate',
                                   'cumPeopleVaccinatedFirstDoseByVaccinationDate')

    # VIF and Multi-leaner------
    vac_dose_data_obj.merge_reg_df(vac_dose_data_obj.multi_df[2], "date", "date", "hospitalCases")
    vac_dose_data_obj.merge_reg_df(vac_dose_data_obj.multi_df[3], "date", "date",
                                   "uniquePeopleTestedBySpecimenDateRollingSum")

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    # Assuming 'dose_taken_num' is the dependent variable, we drop it for VIF calculation
    joint_data = vac_dose_data_obj.reg_df_[["cumDailyNsoDeathsByDeathDate",
                                            "cumPeopleVaccinatedFirstDoseByVaccinationDate",
                                            "hospitalCases",
                                            "uniquePeopleTestedBySpecimenDateRollingSum"]]
    joint_data.dropna(axis=1)

    X = joint_data.drop('cumPeopleVaccinatedFirstDoseByVaccinationDate', axis=1)
    X = add_constant(X)  # adding a constant for the intercept

    # Calculating VIF for each independent variable
    # vif_data = pd.DataFrame()
    # vif_data['feature'] = X.columns
    # vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    #
    # print(vif_data)

    util.stats_mutil_linear_regression(joint_data, ["cumDailyNsoDeathsByDeathDate",
                                                    "hospitalCases",
                                                    "uniquePeopleTestedBySpecimenDateRollingSum"],
                                       "cumPeopleVaccinatedFirstDoseByVaccinationDate")

    util.plot_correlation_matrix(joint_data)

    print(joint_data.describe())


