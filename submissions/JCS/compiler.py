import pandas as pd

weather = pd.read_csv("data/weather.csv")

depWeather = weather.copy()
arrWeather = weather.copy()

depWeather.rename(columns={'AirPort':'Departure','MaxTemperatureC':'DepMaxTemperatureC','MeanTemperatureC':'DepMeanTemperatureC','MinTemperatureC':'DepMinTemperatureC','DewPointC':'DepDewPointC','MeanDewPointC':'DepMeanDewPointC','MinDewpointC':'DepMinDewpointC','MaxHumidity':'DepMaxHumidity','MeanHumidity':'DepMeanHumidity','MinHumidity':'DepMinHumidity','MaxSeaLevelPressurehPa':'DepMaxSeaLevelPressurehPa','MeanSeaLevelPressurehPa':'DepMeanSeaLevelPressurehPa','MinSeaLevelPressurehPa':'DepMinSeaLevelPressurehPa','MaxVisibilityKm':'DepMaxVisibilityKm','MeanVisibilityKm':'DepMeanVisibilityKm','MinVisibilitykM':'DepMinVisibilitykM','MaxWindSpeedKmH':'DepMaxWindSpeedKmH','MeanWindSpeedKmH':'DepMeanWindSpeedKmH','MaxGustSpeedKmH':'DepMaxGustSpeedKmH','Precipitationmm':'DepPrecipitationmm','CloudCover':'DepCloudCover','Events':'DepEvents','WindDirDegrees':'DepWindDirDegrees'}, inplace=True)

arrWeather.rename(columns={'AirPort':'Arrival','MaxTemperatureC':'ArrMaxTemperatureC','MeanTemperatureC':'ArrMeanTemperatureC','MinTemperatureC':'ArrMinTemperatureC','DewPointC':'ArrDewPointC','MeanDewPointC':'ArrMeanDewPointC','MinDewpointC':'ArrMinDewpointC','MaxHumidity':'ArrMaxHumidity','MeanHumidity':'ArrMeanHumidity','MinHumidity':'ArrMinHumidity','MaxSeaLevelPressurehPa':'ArrMaxSeaLevelPressurehPa','MeanSeaLevelPressurehPa':'ArrMeanSeaLevelPressurehPa','MinSeaLevelPressurehPa':'ArrMinSeaLevelPressurehPa','MaxVisibilityKm':'ArrMaxVisibilityKm','MeanVisibilityKm':'ArrMeanVisibilityKm','MinVisibilitykM':'ArrMinVisibilitykM','MaxWindSpeedKmH':'ArrMaxWindSpeedKmH','MeanWindSpeedKmH':'ArrMeanWindSpeedKmH','MaxGustSpeedKmH':'ArrMaxGustSpeedKmH','Precipitationmm':'ArrPrecipitationmm','CloudCover':'ArrCloudCover','Events':'ArrEvents','WindDirDegrees':'ArrWindDirDegrees'}, inplace=True)

data = pd.merge(depWeather, arrWeather, on=['Date'])

holiday = pd.read_csv("data/holiday.csv")

data = pd.merge(data, holiday, on=['Date'])

data.to_csv("external_data.csv")
