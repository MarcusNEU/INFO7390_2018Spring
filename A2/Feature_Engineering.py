import pandas as pd


class DfAfterFE:
    def __init__(self, df):
        self.df = df
        self.date = self.df.date

    @staticmethod
    def time_to_second(time):
        return time.total_seconds()

    @staticmethod
    def weekday_status(day):
        if day == 'Saturday' or day == 'Sunday':
            return 'Weekend'
        else:
            return 'Weekday'

    def feature_engineering(self):
        self.df['NSM'] = pd.to_datetime(self.date) - pd.to_datetime(
            self.date.str.split('\s+').str[0] + " 00:00:00")
        self.df['NSM'] = self.df['NSM'].apply(self.time_to_second)
        self.df['Dayoftheweek'] = pd.to_datetime(self.date).dt.weekday_name.astype('category')
        self.df['Weekdaystatus'] = self.df['Dayoftheweek'].apply(self.weekday_status).astype('category')
        self.df = pd.get_dummies(self.df, columns=['Dayoftheweek', 'Weekdaystatus'])