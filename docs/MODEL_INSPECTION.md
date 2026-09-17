# Uploaded artifact inspection

The uploaded files were copied into this repository without executing them.

## WeatherGPT bundle

File:

```text
models/weather/WeatherGPT_39_Feature_Model.pickle
```

Static pickle inspection shows:

- XGBoost `XGBRegressor` bundle
- 39 input features
- 19 model keys: 17 direct targets + wind direction sin/cos
- output target list includes `wind_direction_10m`
- core lags: 1, 2, 3 and 24 hours
- training window stored in artifact: 2024-01-01 to 2026-09-09
- embedded XGBoost serialization version appears to be 3.4.1
- log1p targets found: precipitation, rain, vapour_pressure_deficit

39 features:

```text
temperature_2m
relative_humidity_2m
dew_point_2m
apparent_temperature
precipitation
rain
surface_pressure
et0_fao_evapotranspiration
vapour_pressure_deficit
wind_speed_10m
wind_gusts_10m
soil_temperature_0_to_7cm
soil_moisture_0_to_7cm
shortwave_radiation
direct_normal_irradiance
diffuse_radiation
cloud_cover
latitude
longitude
wind_u_vector
wind_v_vector
wet_bulb_temp
sin_hour
cos_hour
sin_day
cos_day
pressure_tendency_3h
temp_lag_1
humidity_lag_1
radiation_lag_1
temp_lag_2
humidity_lag_2
radiation_lag_2
temp_lag_3
humidity_lag_3
radiation_lag_3
temp_lag_24
humidity_lag_24
radiation_lag_24
```

## Disaster bundle

File:

```text
models/disaster/disaster_prediction_pipeline.pkl
```

Static inspection shows:

- XGBoost `XGBClassifier`
- 18 classes
- `RobustScaler`
- `LabelEncoder`
- exact 20-feature order saved in the bundle
- stored scikit-learn version: 1.6.1
- embedded XGBoost serialization version appears to be 3.4.1

Classes:

```text
air_pollution
cold_wave
cyclone
drought
earthquake
extreme_temperature
flood
heat_wave
heavy_rainfall
high_pressure
landslide
low_pressure
nominal
storm_surge
strong_winds
thunderstorm
tsunami
volcano
```

## Waterlogging upload

`final_test_predictions.pkl` is a pandas prediction dataset, not a trained
estimator. Detected columns:

```text
state
district
datetime
waterlogged
prediction
probability
```

It is stored under:

```text
data/waterlogging/final_test_predictions.pkl
```

The actual trained `waterlogging.pkl` is still required for new inference.

## Agriculture

No agriculture model was included in this upload. Put the trained file at:

```text
models/agriculture/agriculture.pkl
```

and update its `metadata.json` with the exact training feature order.
