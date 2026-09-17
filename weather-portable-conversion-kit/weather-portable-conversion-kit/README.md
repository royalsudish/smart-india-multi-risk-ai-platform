# WeatherGPT portable conversion

Use this in Google Colab / Linux because the original pickle currently fails to
deserialize on Windows.

## Colab steps

1. Upload:
   - `WeatherGPT_39_Feature_Model.pickle`
   - `convert_weather_pickle_to_portable.py`

2. Install the same XGBoost family version used to create the model:

```python
!pip install -q xgboost==3.4.1
```

3. Run:

```python
!python convert_weather_pickle_to_portable.py
```

4. Download:

```text
weather_portable.zip
```

5. Extract it into your project at:

```text
models/weather/portable/
```

Final layout should be:

```text
models/weather/portable/
├── manifest.json
└── models/
    ├── temperature_2m.json
    ├── ...
    └── wind_direction_cos.json
```

The Windows Flask app should load these `.json` XGBoost models instead of
unpickling the original Linux/Colab bundle.
