import os
import json
import joblib
import pandas as pd

from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_catboost.pkl")


try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
    model_error = None
except Exception as error:
    model = None
    model_loaded = False
    model_error = str(error)


default_payload = {
    "utm_source": "other",
    "utm_medium": "cpc",
    "utm_campaign": "other",
    "utm_adcontent": "unknown",
    "utm_keyword": "unknown",
    "device_category": "mobile",
    "device_os": "unknown",
    "device_brand": "unknown",
    "device_screen_resolution": "414x896",
    "device_browser": "Chrome",
    "geo_country": "Russia",
    "geo_city": "Moscow",
    "visit_month": 5,
    "visit_dayofweek": 4,
    "visit_hour": 19,
    "is_weekend": 0,
    "is_night": 0,
    "is_work_time": 0,
    "is_evening": 1,
    "utm_source_missing": 0,
    "utm_medium_missing": 0,
    "utm_campaign_missing": 0,
    "utm_keyword_missing": 0,
    "device_category_missing": 0,
    "device_os_missing": 0,
    "device_brand_missing": 0,
    "device_browser_missing": 0,
    "geo_country_missing": 0,
    "geo_city_missing": 0,
    "is_organic": 0,
    "is_paid": 1,
    "visit_number_clipped": 2,
    "visit_number_is_outlier": 0,
    "hits_count_clipped": 8,
    "hits_count_is_outlier": 0,
    "visit_number_log": 1.0986,
    "hits_count_log": 2.1972,
    "unique_pages_log": 1.6094,
    "unique_events_log": 1.7918
}


categorical_features = [
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_adcontent",
    "utm_keyword",
    "device_category",
    "device_os",
    "device_brand",
    "device_screen_resolution",
    "device_browser",
    "geo_country",
    "geo_city"
]


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        model_loaded=model_loaded,
        model_error=model_error,
        model_path=MODEL_PATH,
        default_payload=json.dumps(
            default_payload,
            ensure_ascii=False,
            indent=4
        )
    )


@app.route("/predict", methods=["POST"])
def predict():

    if not model_loaded:
        return jsonify({
            "status": "error",
            "message": "Модель не загружена",
            "details": model_error
        }), 500

    try:
        raw_text = request.form.get("json_data", "")
        data = json.loads(raw_text)

        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        if hasattr(model, "feature_names_"):
            expected_features = list(model.feature_names_)

            for column in expected_features:
                if column not in df.columns:
                    df[column] = "unknown" if column in categorical_features else 0

            df = df[expected_features]

        for column in df.columns:
            if column in categorical_features:
                df[column] = df[column].fillna("unknown").astype(str)
            else:
                df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

        prob = model.predict_proba(df)[0][1]
        pred_class = model.predict(df)[0]

        return render_template(
            "result.html",
            probability=round(float(prob) * 100, 2),
            predicted_class=int(pred_class)
        )

    except Exception as error:
        return jsonify({
            "status": "error",
            "message": "Ошибка предсказания",
            "details": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )