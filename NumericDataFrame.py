import pandas as pd

import util
from AllDataFrameInOne import AllDataFrameInOne


class NDF(AllDataFrameInOne):
    def __init__(self, rightDataFile: str, leftDataFile: str,
                 rightHeader=0, leftHeader=0,
                 rightSheet=1, leftSheet=1):
        super().__init__(rightDataFile, leftDataFile, rightHeader, leftHeader, rightSheet, leftSheet)

        # Create and init data frame list
        self.multi_df = []
        self._init_df_list()

    def _init_df_list(self):
        self.multi_df.append(self.left_df)
        self.multi_df.append(self._right_independent_df_)

    def edit_reg_df(self, new_df):
        self.reg_df_ = new_df

    def insert_df(self, df_url: str, head=0, is_time_format=True, time_col="date"):
        new_df = util.format_read_csv(df_url, header_id=head)
        new_df.dropna(axis=1)

        if is_time_format:
            # Filtering out non-datetime entries from the data
            new_df[time_col] = new_df[time_col].astype(str)
            new_df = new_df[new_df[time_col].str.match(r'\d{4}-\d{2}-\d{2}')]
            new_df[time_col] = pd.to_datetime(new_df[time_col]).dt.date

        self.multi_df.append(new_df)

    def merge_reg_df(self, df_, reg_col: str, tar_col: str, new_column_to_keep):
        reg_copy = self.reg_df_

        self.reg_df_ = pd.merge(self.reg_df_, df_,
                                left_on=reg_col,
                                right_on=tar_col, how='left')

        self.reg_df_ = pd.concat([reg_copy, self.reg_df_[new_column_to_keep]], axis=1)

    def simple_join(self, left_index: int, right_index: int,
                    left_col: str, right_col: str,
                    time_format=True, is_daily=True):
        """
        Join two table by two columns
        Contains time formatting and weekly separation
        :param left_index: Index in: multi_df: list
        :param right_index: Index in: multi_df: list
        :param left_col: The column used for join in the __left__ df
        :param right_col:The column used for join in the __right__ df
        :param time_format: (Bool) whether the time format is standard
        :param is_daily: (Bool) whether convert the daily time to weekly
        :return: None
        """

        left_df = self.multi_df[left_index]
        right_df = self.multi_df[right_index]

        if time_format:
            # Filtering out non-datetime entries from the data
            left_df[left_col] = left_df[left_col].astype(str)
            left_df = left_df[left_df[
                left_col].str.match(r'\d{4}-\d{2}-\d{2}')]

            right_df[right_col] = right_df[right_col].astype(str)
            right_df = right_df[right_df[
                right_col].str.match(r'\d{4}-\d{2}-\d{2}')]

            # Convert 'Week Commencing' and 'date' columns to date format without time
            left_df[left_col] = pd.to_datetime(
                left_df[left_col]).dt.date
            right_df[right_col] = pd.to_datetime(
                right_df[right_col]).dt.date

        if is_daily:
            # clip data from day to week | jump read the data and sum them by weeks
            left_df = self.format_dataset_weekly('date',
                                                 'newDailyNsoDeathsByDeathDate',
                                                 pd.to_datetime('23/10/23').date())

        # right join
        self.reg_df_ = pd.merge(left_df, right_df,
                                left_on=left_col,
                                right_on=right_col, how='right')

        self.reg_df_ = self.reg_df_.dropna(how='all', axis=0)
        self.reg_df_ = self.reg_df_.dropna(how='all', axis=1)

    def format_dataset_weekly(self, df_index: int,
                              date_col: str, data_col: str, start_date) -> pd.DataFrame:
        target_df = self.multi_df[df_index]

        # !Note: The target dataframe's colum should be converted to datatime
        # format: pd.to_datetime('23/10/23').date()
        date_index = target_df.index[target_df[date_col] == start_date]

        if len(date_index) < 1:
            raise Exception("Could not find element in: _left_independent_df_: pd.DataFrame")

        start_index = int(date_index[0])

        # create new df by columns
        out_df = pd.DataFrame(columns=target_df.columns)
        new_data_col = []

        for i in range(len(target_df)):
            current_value = 0
            if (start_index + i*7 + 7) > len(target_df):
                break

            out_df.loc[i] = target_df.loc[start_index + (i * 7)]

            for day in range(7):
                current_value += target_df.loc[start_index + (i * 7 + day)][data_col]

            new_data_col.append(current_value)

        # change column data
        out_df[data_col] = new_data_col

        return out_df

    def normalize_colum(self, tar_col: str, mod: str):
        """
        Normalization the data frame used for regression
        :param tar_col: The target column contains the data
        :param mod: <str, 'log'> | <str, 'sigmoid'>
        """

        if mod is not None:
            self.reg_df_[tar_col] = util.sig_process_data(self.reg_df_[tar_col], mod)

    def plot_box_reg(self):
        """
        Plot the box chart to visualize the data
        !Note: The data frame is the merged dataset

        :param title_0: Title of box plot
        :return:
        """
        from matplotlib import pyplot as plt

        # Number of rows and columns for subplots
        n_cols = 2  # You can change this as per your requirement
        n_rows = (len(self.reg_df_.columns) + 1) // n_cols

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        # Plotting box plots for each column
        for i, col in enumerate(self.reg_df_.columns):
            if self.reg_df_[col].dtype in ['float64', 'int64']:
                self.reg_df_.boxplot(column=[col], ax=axes[i])
                axes[i].set_title(col)
            else:
                axes[i].set_visible(False)

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

    def org_data_insight(self, left_col_date: str, left_col_value: str, left_title: str,
                         right_col_date: str, right_col_value: str, right_title: str):
        util.plot_trend(self.left_df, left_col_date, left_col_value, left_title)
        util.plot_trend(self._right_independent_df_, right_col_date, right_col_value, right_title)

    def reg_data_sight(self, left_col_date: str, left_col_value: str, left_title: str,
                       right_col_date: str, right_col_value: str, right_title: str):
        util.plot_trend(self.reg_df_, left_col_date, left_col_value, left_title)
        util.plot_trend(self.reg_df_, right_col_date, right_col_value, right_title)

    def sum_multi_data(self):
        for _ in self.multi_df:
            print(_.describe())

    def print_column_names(self):
        for df in self.multi_df:
            print(df.columns)
            print('------')
