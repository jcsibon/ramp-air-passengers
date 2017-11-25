import pandas as pd
import numpy as np
import os


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df

        ## print(X_encoded.head())

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))

        # following http://stackoverflow.com/questions/16453644/regression-with-date-variable-using-scikit-learn
        X_encoded['Date'] = X_encoded['DateOfDeparture']
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)


        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
        
        ## print(X_encoded.head())

        if True:
            path = os.path.dirname(__file__)
            ## print(path)
            external_data = pd.read_csv(os.path.join(path, 'external_data.csv'))
            X_encoded = pd.merge(
                X_encoded, external_data, how='left',
                left_on=['Date', 'Departure', 'Arrival'],
                right_on=['Date', 'Departure', 'Arrival'],
                sort=False)

            ## print(X_encoded['DepEvents'].unique())
            
            def withFrog(Value):
                if type(Value) is str:
                    if Value.find("Frog") == -1:
                        return False
                    else:
                        return True
                else:
                    return False

            def withThun(Value):
                if type(Value) is str:
                    if Value.find("Thunderstorm") == -1:
                        return False
                    else:
                        return True
                else:
                    return False

            def withSnow(Value):
                if type(Value) is str:
                    if Value.find("Snow") == -1:
                        return False
                    else:
                        return True
                else:
                    return False

            def withRain(Value):
                if type(Value) is str:
                    if Value.find("Rain") == -1:
                        return False
                    else:
                        return True
                else:
                    return False

        
            X_encoded['DepFrog'] = X_encoded['DepEvents'].apply(withFrog)
            X_encoded['DepThun'] = X_encoded['DepEvents'].apply(withThun)
            X_encoded['DepSnow'] = X_encoded['DepEvents'].apply(withSnow)
            X_encoded['DepRain'] = X_encoded['DepEvents'].apply(withRain)
            X_encoded['ArrFrog'] = X_encoded['ArrEvents'].apply(withFrog)
            X_encoded['ArrThun'] = X_encoded['ArrEvents'].apply(withThun)
            X_encoded['ArrSnow'] = X_encoded['ArrEvents'].apply(withSnow)
            X_encoded['ArrRain'] = X_encoded['ArrEvents'].apply(withRain)

        X_encoded.to_csv("X_array_2.csv")

        ## Suppression des colonnes non numériques

        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)

        # X_encoded = X_encoded.drop('WeeksToDeparture', axis=1)
        # X_encoded = X_encoded.drop('std_wtd', axis=1)

        if True:
            X_encoded = X_encoded.drop('Date', axis=1)
            X_encoded = X_encoded.drop('DepEvents', axis=1)
            X_encoded = X_encoded.drop('ArrEvents', axis=1)
            X_encoded = X_encoded.drop('DepMaxGustSpeedKmH', axis=1)
            X_encoded = X_encoded.drop('ArrMaxGustSpeedKmH', axis=1)
            X_encoded = X_encoded.drop('DepPrecipitationmm', axis=1)
            X_encoded = X_encoded.drop('ArrPrecipitationmm', axis=1)

        if True:
            m = X_encoded.isnull().any()
            print("========= COLUMNS WITH NULL VALUES =================")
            print(m[m])
            m = np.isfinite(X_encoded.select_dtypes(include=['float64'])).any()
            print("========= COLUMNS WITH INFINITE VALUES =================")
            print(m[m])

        X_array = X_encoded.values

        return X_array
