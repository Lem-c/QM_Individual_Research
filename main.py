"""
@ Author    : Lem Chen
@ Time      : 23/11/2023
"""

import NumericDataFrame as myDF
import pandas as pd
import util

# data from: https://coronavirus.data.gov.uk/details/download
first_dose_taken_data_url = "./data/daily_first_dose.csv"
second_dose_taken_data_url = "./data/daily_second_dose.csv"
death_data_url = "./data/deathDate.csv"
cases_data_url = "./data/hospital_cases_2023-12-14.csv"
PCR_test_data_url = "./data/testingData.csv"


if __name__ == "__main__":
    vac_dose_data_obj = myDF.NDF(second_dose_taken_data_url, death_data_url)
    vac_dose_data_obj.insert_df(first_dose_taken_data_url)
    vac_dose_data_obj.insert_df(cases_data_url)
    vac_dose_data_obj.insert_df(PCR_test_data_url)
    # vac_dose_data_obj.merge_reg_df(vac_dose_data_obj.multi_df[2], 'date', 'date')
    vac_dose_data_obj.sum_multi_data()

    # Join the data by date
    vac_dose_data_obj.simple_join(0, 1, 'date', 'date', is_daily=False)

    # util.plot_hist(vac_dose_data_obj.multi_df[0], "cumDailyNsoDeathsByDeathDate",
    #                "Histogram of Cumulative Reported Death Count", "Cumulative Reported Death Count")
    # util.plot_hist(vac_dose_data_obj.multi_df[1], "cumPeopleVaccinatedSecondDoseByVaccinationDate",
    #                "Histogram of Cumulative Second Dose Vaccinations", "Cumulative Second Dose Vaccinations")
    # util.plot_hist(vac_dose_data_obj.multi_df[2], "hospitalCases",
    #                "Histogram of Hospital Infected Patient Cases", "Hospital Infected Patient Cases")
    # util.plot_hist(vac_dose_data_obj.multi_df[3], "uniqueCasePositivityBySpecimenDateRollingSum",
    #                "Histogram of Positive PCR Test Statistical Data",
    #                "Unique Case Positivity by Specimen Date Rolling Sum")

    # # Data visualization
    # vac_dose_data_obj.reg_data_sight('date', 'cumDailyNsoDeathsByDeathDate', 'Daily death',
    #                                  'date', 'cumPeopleVaccinatedSecondDoseByVaccinationDate', 'Daily first vac dose')

    # util.plot_scatter(vac_dose_data_obj.reg_df_,
    #                   'cumDailyNsoDeathsByDeathDate', 'cumPeopleVaccinatedSecondDoseByVaccinationDate',
    #                   point_size=2)
    #
    # # # Data normalization
    # # # log is better than sigmoid
    # # vac_dose_data_obj.normalize_colum('cumPeopleVaccinatedSecondDoseByVaccinationDate', mod='log')
    # # vac_dose_data_obj.normalize_colum('cumDailyNsoDeathsByDeathDate', mod='log')
    #
    # for col in vac_dose_data_obj.reg_df_.columns:
    #     if vac_dose_data_obj.reg_df_[col].dtype in ['float64', 'int64']:  # Apply only to numeric columns
    #         vac_dose_data_obj.reg_df_ = util.remove_outliers(vac_dose_data_obj.reg_df_, col)
    #
    # # Simple linear regression
    # util.simple_linear_regression(vac_dose_data_obj.reg_df_,
    #                               'cumDailyNsoDeathsByDeathDate', 'cumPeopleVaccinatedSecondDoseByVaccinationDate',
    #                               isPolynomial=False, polynomialDegree=2)
    #
    # util.stats_linear_regression(vac_dose_data_obj.reg_df_,
    #                              'cumDailyNsoDeathsByDeathDate', 'cumPeopleVaccinatedSecondDoseByVaccinationDate',
    #                              'death', 'doseTake', 'Residual vs. Fitted Plot')
    #
    # util.support_vector_regression(vac_dose_data_obj.reg_df_, 'cumDailyNsoDeathsByDeathDate',
    #                                'cumPeopleVaccinatedSecondDoseByVaccinationDate')
    #
    # ------VIF and Multi-leaner------
    vac_dose_data_obj.merge_reg_df(vac_dose_data_obj.multi_df[3], "date", "date", "hospitalCases")
    vac_dose_data_obj.merge_reg_df(vac_dose_data_obj.multi_df[4], "date", "date",
                                   "uniquePeopleTestedBySpecimenDateRollingSum")

    # vac_dose_data_obj.normalize_colum('hospitalCases', mod='log')
    # vac_dose_data_obj.normalize_colum('uniquePeopleTestedBySpecimenDateRollingSum', mod='log')

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # Assuming 'dose_taken_num' is the dependent variable, we drop it for VIF calculation
    joint_data = vac_dose_data_obj.reg_df_[["cumDailyNsoDeathsByDeathDate",
                                            "cumPeopleVaccinatedSecondDoseByVaccinationDate",
                                            "hospitalCases",
                                            "uniquePeopleTestedBySpecimenDateRollingSum"]]
    joint_data.dropna(axis=1)

    # X = joint_data.drop('cumPeopleVaccinatedSecondDoseByVaccinationDate', axis=1)
    # Calculating VIF for each independent variable
    # vif_data = pd.DataFrame()
    # vif_data['feature'] = X.columns
    # vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    #
    # print(vif_data)

    util.auto_encoding_df(joint_data, "cumDailyNsoDeathsByDeathDate", 10)
    util.auto_encoding_df(joint_data, "hospitalCases", 5)
    util.auto_encoding_df(joint_data, "uniquePeopleTestedBySpecimenDateRollingSum", 10)

    print(joint_data.head())

    joint_data.to_csv("temp.csv")
    util.stats_neg_binom_reg(joint_data, [
                                                    "cumDailyNsoDeathsByDeathDate",
                                                    "hospitalCases",
                                                    "uniquePeopleTestedBySpecimenDateRollingSum"
    ],
                                       "cumPeopleVaccinatedSecondDoseByVaccinationDate")

    # util.plot_correlation_matrix(joint_data)

    # print(joint_data.describe())
