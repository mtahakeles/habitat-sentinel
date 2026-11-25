# Habitat Sentinel

A small wildfire dashboard I built for the [Hack the Habitat](https://hack-the-habitat.devpost.com) hackathon (2026). It has two parts:

1. **Live map** of open wildfire events from [NASA EONET](https://eonet.gsfc.nasa.gov/), fetched in the browser.
2. **Risk gauge**: two sliders (vegetation health and land surface temperature) feed a logistic regression trained on 1,713 MODIS-derived observations from Central Canada ([dataset](https://github.com/ouladsayadyounes/WildFires)).

## The model

| | |
|---|---|
| Features | NDVI, LST |
| Model | logistic regression, `class_weight="balanced"` |
| Test split | 25%, stratified |
| ROC-AUC | 0.71 |
| Recall / precision | 0.70 / 0.36 |

Only 23% of the rows are fires. My first unweighted model hit 0.13 recall, meaning it basically always said "no fire". Weighting the classes fixed that at the cost of many false alarms. For an early-warning tool I think that's the right side to err on, but the precision number is low and the page says so.

The training script writes the coefficients and scaler parameters to `site/model_results.json`, and `site/app.js` computes the same sigmoid in JavaScript. That way the site needs no backend.

**Limits:** two features is very little. Wind, humidity, fuel load and terrain matter a lot and aren't in the data. The data also covers a single region.

## Run it

```bash
# website
cd site && python -m http.server 8000      # open http://localhost:8000

# retrain the model (from the repo root)
pip install -r requirements.txt
python analysis/train_risk_model.py
```

## Layout

```
analysis/train_risk_model.py   trains the model, writes site/model_results.json
data/WildFires_DataSet.csv     training data
site/                          static page (HTML/CSS/JS, Leaflet map)
```

## Credits

NASA EONET API v3 · WildFires dataset by ouladsayadyounes (MODIS/Terra via LP DAAC) · Leaflet and CARTO basemap tiles.
