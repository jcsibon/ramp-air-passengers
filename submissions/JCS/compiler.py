import pandas as pd

weather = pd.read_csv("data/weather.csv")

depWeather = weather.copy()
arrWeather = weather.copy()

depWeather.rename(columns={'AirPort':'Departure','MaxTemperatureC':'DepMaxTemperatureC','MeanTemperatureC':'DepMeanTemperatureC','MinTemperatureC':'DepMinTemperatureC','DewPointC':'DepDewPointC','MeanDewPointC':'DepMeanDewPointC','MinDewpointC':'DepMinDewpointC','MaxHumidity':'DepMaxHumidity','MeanHumidity':'DepMeanHumidity','MinHumidity':'DepMinHumidity','MaxSeaLevelPressurehPa':'DepMaxSeaLevelPressurehPa','MeanSeaLevelPressurehPa':'DepMeanSeaLevelPressurehPa','MinSeaLevelPressurehPa':'DepMinSeaLevelPressurehPa','MaxVisibilityKm':'DepMaxVisibilityKm','MeanVisibilityKm':'DepMeanVisibilityKm','MinVisibilitykM':'DepMinVisibilitykM','MaxWindSpeedKmH':'DepMaxWindSpeedKmH','MeanWindSpeedKmH':'DepMeanWindSpeedKmH','MaxGustSpeedKmH':'DepMaxGustSpeedKmH','Precipitationmm':'DepPrecipitationmm','CloudCover':'DepCloudCover','Events':'DepEvents','WindDirDegrees':'DepWindDirDegrees'}, inplace=True)

arrWeather.rename(columns={'AirPort':'Arrival','MaxTemperatureC':'ArrMaxTemperatureC','MeanTemperatureC':'ArrMeanTemperatureC','MinTemperatureC':'ArrMinTemperatureC','DewPointC':'ArrDewPointC','MeanDewPointC':'ArrMeanDewPointC','MinDewpointC':'ArrMinDewpointC','MaxHumidity':'ArrMaxHumidity','MeanHumidity':'ArrMeanHumidity','MinHumidity':'ArrMinHumidity','MaxSeaLevelPressurehPa':'ArrMaxSeaLevelPressurehPa','MeanSeaLevelPressurehPa':'ArrMeanSeaLevelPressurehPa','MinSeaLevelPressurehPa':'ArrMinSeaLevelPressurehPa','MaxVisibilityKm':'ArrMaxVisibilityKm','MeanVisibilityKm':'ArrMeanVisibilityKm','MinVisibilitykM':'ArrMinVisibilitykM','MaxWindSpeedKmH':'ArrMaxWindSpeedKmH','MeanWindSpeedKmH':'ArrMeanWindSpeedKmH','MaxGustSpeedKmH':'ArrMaxGustSpeedKmH','Precipitationmm':'ArrPrecipitationmm','CloudCover':'ArrCloudCover','Events':'ArrEvents','WindDirDegrees':'ArrWindDirDegrees'}, inplace=True)

data = pd.merge(depWeather, arrWeather, on=['Date'])

holiday = pd.read_csv("data/holiday.csv")

data = pd.merge(data, holiday, on=['Date'])

cities = pd.read_csv("data/cities.csv")

depCities = cities.copy()
depCities.rename(columns={'AirPort':'Departure','Latitude':'DepLatitude','Longitude':'DepLongitude','State':'DepState','Pop_2010':'DepPop_2010','Pop_2016':'DepPop_2016','Pop_2015':'DepPop_2015','Age_median':'DepAge_median','Companies':'DepCompanies','Graduates':'DepGraduates','Housings':'DepHousings','Income':'DepIncome','Foreigners':'DepForeigners','Poverty':'DepPoverty','Veterans':'DepVeterans'}, inplace=True)
data = pd.merge(data, depCities, on=['Departure'])

arrCities = cities.copy()
arrCities.rename(columns={'AirPort':'Arrival','Latitude':'ArrLatitude','Longitude':'ArrLongitude','State':'ArrState','Pop_2010':'ArrPop_2010','Pop_2016':'ArrPop_2016','Pop_2015':'ArrPop_2015','Age_median':'ArrAge_median','Companies':'ArrCompanies','Graduates':'ArrGraduates','Housings':'ArrHousings','Income':'ArrIncome','Foreigners':'ArrForeigners','Poverty':'ArrPoverty','Veterans':'ArrVeterans'}, inplace=True)
data = pd.merge(data, arrCities, on=['Arrival'])

data.to_csv("external_data.csv")
