import pandas as pd
import util

# pandas print option
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


class AllDataFrameInOne:
    def __init__(self, firstDataFile: str, secondDataFile: str,
                 rightHeader=1, leftHeader=1,
                 rightSheet=1, leftSheet=1):
        """

        :param firstDataFile: right df
        :param secondDataFile: left df
        :param rightHeader: The row represents the right df column name
        :param leftHeader: The row represents the left df column name
        :param rightSheet: ID if multi sheets
        :param leftSheet: ID if multi sheets
        """
        self._right_independent_df_ = util.format_read_csv(firstDataFile,
                                                           sheet_id=rightSheet,
                                                           header_id=rightHeader)
        self._left_independent_df_ = util.format_read_csv(secondDataFile,
                                                          sheet_id=leftSheet,
                                                          header_id=leftHeader)

        # The df used for data visualization
        self.borough_sum_df_ = None
        # The df used for linear regression
        self.reg_df_ = None

    def filter_case(self, tar_col: str, main_text='MinorText'):
        """

        :param tar_col: The column that going to be processed
        :param main_text: The column name for indexing
        """
        return self._left_independent_df_[self._left_independent_df_[main_text] == tar_col]

    def join_minor_tables(self, crime_type: str, crime_year: int, rent_year: str,
                          left_merge_column='LookUp_BoroughName',
                          right_merge_column='Area'):
        """
        @TODO: Comments on the way...
        :param crime_type: The specific type of crime in 'MinorText' column
        :param crime_year: The first year sum from
        :param rent_year: In type of '20XX-XX'
        :param left_merge_column: pd.merge(..., left_on=? | keep after merge for indexing
        :param right_merge_column: pd.merge(..., right_on=?
        :return: A left joint dataset contains borough name, crime data sum in a year and rent year
        """
        filtered_crime_df = self.filter_case(crime_type)
        columns = [str(year_month) for year_month in range(crime_year, crime_year + 12)]
        filtered_crime_df['year_total'] = filtered_crime_df[columns].sum(axis=1)
        filtered_crime_df = filtered_crime_df.iloc[:, :3].join(filtered_crime_df['year_total'])

        joined_data = pd.merge(filtered_crime_df, self._right_independent_df_,
                               left_on=left_merge_column,
                               right_on=right_merge_column, how='left')

        # Handle the NA data and remove them
        joined_data.replace('..', pd.NA, inplace=True)
        joined_data.dropna(inplace=True)
        self.reg_df_ = joined_data[[left_merge_column, 'year_total', rent_year]]

        return self.reg_df_

    def _sum_table_by_groups(self, table_main_text='MajorText', data_column='LookUp_BoroughName'):
        # Grouping the data by 'MajorText' and summing up all the monthly crime counts
        self._left_independent_df_ = self._left_independent_df_.groupby([table_main_text, data_column]).sum()
        # Resetting the index to have 'MajorText' as a column
        self._left_independent_df_.reset_index(inplace=True)

    def join_multi_row(self, selected_crime: list, crime_year: int, rent_year: str,
                       text_column='MinorText',
                       group_column='LookUp_BoroughName',
                       join_column='Area'):
        """
        @TODO: Comments on the way...
        :param selected_crime: It should be a list saves more than one crime type
                                Like the 'selected_crimes' in main
        :param crime_year: Used to sum a year's all cases
        :param rent_year: The data in this year
        :param text_column: Name of the processing target
        :param group_column: The column data going to grouped by
        :param join_column: The column that right table used for join
        :return: Merged dataframe used for multi linear regression
        """

        if text_column == 'MajorText':
            self._sum_table_by_groups(text_column, )

        filtered_data = self._left_independent_df_[self._left_independent_df_[text_column].isin(selected_crime)]

        if filtered_data is None:
            raise Exception("Fail to find target crime type")

        # Columns for the year 2020
        columns = [str(year_month) for year_month in range(crime_year, crime_year + 12)]
        # Group by borough and sum the monthly data for the 'rent_year'
        filtered_data = filtered_data.groupby([group_column, text_column])[columns].sum()
        filtered_data['year_total'] = filtered_data.sum(axis=1)
        # Pivot the crime data count
        filtered_data = filtered_data.reset_index().pivot(index=group_column,
                                                          columns=text_column,
                                                          values='year_total')

        rent_data = self._right_independent_df_[[join_column, rent_year]].dropna()
        rent_data[rent_year] = pd.to_numeric(rent_data[rent_year], errors='coerce')

        merged_data = pd.merge(filtered_data, rent_data,
                               left_index=True,
                               right_on=join_column)

        self.reg_df_ = merged_data
        return self.reg_df_

    def join_all_text(self, columns_str: list, pivot_tar: str, val_col_name: str):
        """
        Add all crimes together for analysis
        Example usage:
            join_all_crime(["MajorText", "MinorText", "LookUp_BoroughName"],
                            "LookUp_BoroughName",
                            "CrimeCount")
        :param columns_str: The list no null contains columns with string type data
        :param pivot_tar: The column that going to be applied pivot
        :param val_col_name: New column name after melting and indexing
        :return: new table saves all borough names as column name
        """

        current_columns = columns_str

        # Melting the dataframe to transform the years columns
        df_melted = self._left_independent_df_.melt(id_vars=current_columns,
                                                    var_name="YearMonth",
                                                    value_name=val_col_name)
        if pivot_tar in columns_str:
            current_columns.remove(pivot_tar)
        current_columns.append("YearMonth")

        # Pivoting the dataframe to make 'LookUp_BoroughName' as column headers
        df_pivoted = df_melted.pivot_table(index=current_columns,
                                           columns=pivot_tar,
                                           values=val_col_name).reset_index()

        current_columns.remove("YearMonth")
        df_pivoted = util.sum_table_by_year(df_pivoted)
        df_final = df_pivoted.drop(columns=current_columns)

        self.borough_sum_df_ = df_final
        return df_final

    def print_column_names(self):
        print(self._right_independent_df_.columns)
        print("____")
        print(self._left_independent_df_.columns)

