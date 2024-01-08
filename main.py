"""
@ Author    : Lem Chen
@ Time      : 23/11/2023
"""

import NumericDataFrame as myDF
import util

# data from: https://coronavirus.data.gov.uk/details/download
dose_taken_data_url = "./data/daily_first_dose.csv"
death_data_url = "./data/data_2023-Nov-23.csv"
cases_data_url = "./data/hospital_cases_2023-12-14.csv"


if __name__ == "__main__":
    vac_dose_data_obj = myDF.NDF(dose_taken_data_url, death_data_url)
    vac_dose_data_obj.insert_df(cases_data_url)
    vac_dose_data_obj.print_column_names()

    # Join the data by date
    vac_dose_data_obj.simple_join(0, 1, 'date', 'date', is_daily=False)

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

    # Simple linear regression
    util.simple_linear_regression(vac_dose_data_obj.reg_df_,
                                  'cumDailyNsoDeathsByDeathDate', 'cumPeopleVaccinatedFirstDoseByVaccinationDate',
                                  isPolynomial=False, polynomialDegree=2)

    # util.stats_linear_regression(vac_dose_data_obj.reg_df_,
    #                              'cumDailyNsoDeathsByDeathDate', 'cumPeopleVaccinatedFirstDoseByVaccinationDate')
