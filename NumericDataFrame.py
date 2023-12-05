import pandas as pd

import util
from AllDataFrameInOne import AllDataFrameInOne


class NDF(AllDataFrameInOne):
    def __init__(self, rightDataFile: str, leftDataFile: str,
                 rightHeader=0, leftHeader=0,
                 rightSheet=1, leftSheet=1):
        super().__init__(rightDataFile, leftDataFile, rightHeader, leftHeader, rightSheet, leftSheet)

    def edit_reg_df(self, new_df):
        self.reg_df_ = new_df

    def simple_join(self, left_col: str, right_col: str, time_format=True, is_daily=True):
        """
        Join two table by two columns
        Contains time formatting and weekly separation
        :param left_col: The column used for join in the __left__ df
        :param right_col:The column used for join in the __right__ df
        :param time_format: (Bool) whether the time format is standard
        :param is_daily: (Bool) whether convert the daily time to weekly
        :return: None
        """
        if time_format:
            # Filtering out non-datetime entries from the data
            self._left_independent_df_[left_col] = self._left_independent_df_[left_col].astype(str)
            self._left_independent_df_ = self._left_independent_df_[self._left_independent_df_[
                left_col].str.match(r'\d{4}-\d{2}-\d{2}')]

            self._right_independent_df_[right_col] = self._right_independent_df_[right_col].astype(str)
            self._right_independent_df_ = self._right_independent_df_[self._right_independent_df_[
                right_col].str.match(r'\d{4}-\d{2}-\d{2}')]

            # Convert 'Week Commencing' and 'date' columns to date format without time
            self._left_independent_df_[left_col] = pd.to_datetime(
                self._left_independent_df_[left_col]).dt.date
            self._right_independent_df_[right_col] = pd.to_datetime(
                self._right_independent_df_[right_col]).dt.date

        if is_daily:
            # clip data from day to week | jump read the data and sum them by weeks
            self._left_independent_df_ = self.format_dataset_weekly('date',
                                                                    'newDailyNsoDeathsByDeathDate',
                                                                    pd.to_datetime('23/10/23').date())

        # right join
        self.reg_df_ = pd.merge(self._left_independent_df_, self._right_independent_df_,
                                left_on=left_col,
                                right_on=right_col, how='right')

        self.reg_df_ = self.reg_df_.dropna(how='all', axis=0)
        self.reg_df_ = self.reg_df_.dropna(how='all', axis=1)

    def format_dataset_weekly(self, date_col: str, data_col: str, start_date) -> pd.DataFrame:
        # !Note: The target dataframe's colum should be converted to datatime
        # format: pd.to_datetime('23/10/23').date()
        date_index = self._left_independent_df_.index[self._left_independent_df_[date_col] == start_date]

        if len(date_index) < 1:
            raise Exception("Could not find element in: _left_independent_df_: pd.DataFrame")

        start_index = int(date_index[0])

        # create new df by columns
        out_df = pd.DataFrame(columns=self._left_independent_df_.columns)
        new_data_col = []

        for i in range(len(self._left_independent_df_)):
            current_value = 0
            if (start_index + i*7 + 7) > len(self._left_independent_df_):
                break

            out_df.loc[i] = self._left_independent_df_.loc[start_index + (i*7)]

            for day in range(7):
                current_value += self._left_independent_df_.loc[start_index + (i*7+day)][data_col]

            new_data_col.append(current_value)

        # change column data
        out_df[data_col] = new_data_col

        return out_df

    def normalize_colum(self, tar_col: str, mod: str):
        if mod is not None:
            self.reg_df_[tar_col] = util.sig_process_data(self.reg_df_[tar_col], mod)

    def plot_box_reg(self, col_0: str, col_1: str,
                     title_0: str, title_1: str):
        from matplotlib import pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Distribution of 'Number of vaccinations delivered'
        sns.boxplot(ax=axes[0], x=self.reg_df_[col_0])
        axes[0].set_title(title_0)

        # Distribution of 'newDailyNsoDeathsByDeathDate'
        sns.boxplot(ax=axes[1], x=self.reg_df_[col_1])
        axes[1].set_title(title_1)

        plt.show()

    def org_data_insight(self, left_col_date: str, left_col_value: str, left_title: str,
                         right_col_date: str, right_col_value: str, right_title: str):
        util.plot_trend(self._left_independent_df_, left_col_date, left_col_value, left_title)
        util.plot_trend(self._right_independent_df_, right_col_date, right_col_value, right_title)

    def reg_data_sight(self, left_col_date: str, left_col_value: str, left_title: str,
                       right_col_date: str, right_col_value: str, right_title: str):
        util.plot_trend(self.reg_df_, left_col_date, left_col_value, left_title)
        util.plot_trend(self.reg_df_, right_col_date, right_col_value, right_title)
