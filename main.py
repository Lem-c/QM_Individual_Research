"""
@ Author    : Lem Chen
@ Time      : 23/11/2023
"""

import NumericDataFrame as myDF
import util

# data from: https://coronavirus.data.gov.uk/details/download
dose_taken_data_url = "./data/daily_first_dose.csv"
death_data_url = "./data/data_2023-Nov-23.csv"


if __name__ == "__main__":
    crime_df = myDF.NDF(dose_taken_data_url, death_data_url)
    crime_df.print_column_names()

    # crime_df.print_column_names()
    crime_df.simple_join('date', 'date', is_daily=False)

    # Data visualization
    crime_df.reg_data_sight('date', 'cumDailyNsoDeathsByDeathDate', 'Daily death',
                            'date', 'cumPeopleVaccinatedFirstDoseByVaccinationDate', 'Daily first vac dose')
    crime_df.plot_box_reg('cumPeopleVaccinatedFirstDoseByVaccinationDate',
                          'cumDailyNsoDeathsByDeathDate',
                          'Distribution of Number of Vaccinations Delivered',
                          'Distribution of Sum Daily NSO Deaths By Death Date')

    util.plot_scatter(crime_df.reg_df_,
                      'cumDailyNsoDeathsByDeathDate', 'cumPeopleVaccinatedFirstDoseByVaccinationDate',
                      point_size=2)

    # Data normalization
    # log is better than sigmoid
    crime_df.normalize_colum('cumPeopleVaccinatedFirstDoseByVaccinationDate', mod='log')
    crime_df.normalize_colum('cumDailyNsoDeathsByDeathDate', mod='log')

    util.simple_linear_regression(crime_df.reg_df_,
                                  'cumDailyNsoDeathsByDeathDate', 'cumPeopleVaccinatedFirstDoseByVaccinationDate',
                                  isPolynomial=True, polynomialDegree=2)
