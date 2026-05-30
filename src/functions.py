import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import  train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)
import warnings

from IPython.display import display
sns.set_theme(style="whitegrid", palette="muted")
warnings.filterwarnings("ignore")


### Первичный анализ

def df_anzeigen(df, name):
    print(f"Структура {name}\n") 
    print(f"Размер  датасета: {df.shape}")
    print("\n************* info *************\n")
    display(df.info())
    print("\n************* head *************\n")
    display(df.head())
    
def df_model_anzeigen(df, name):
    print(f"Структура {name}\n")  
    print(f"Размер итогового датасета: {df.shape}")
    print("\nТипы данных:")
    print(df.dtypes)
    print("\nРаспределение target:")
    display(df["target"].value_counts())
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="target")
    plt.title("Распределение целевой переменной")
    plt.xlabel("Класс target")
    plt.ylabel("Количество наблюдений")
    plt.show()    
    print("\n*************   target   *************")
    counts = df['target'].value_counts()
    percents = df['target'].value_counts(normalize=True) * 100
    print(f"Без конверсии: {counts[0]} ({percents[0]:.2f}%)")
    print(f"Конверсия:     {counts[1]} ({percents[1]:.2f}%)")
    print(f"Дисбаланс:     1:{int(counts[0]/counts[1])}")
    
def df_miss_feat(df):
    miss_ = pd.DataFrame({"count": df.isna().sum(), "percent": (df.isna().mean() * 100).round(2)})
    miss_ = miss_[miss_["count"] > 0].sort_values(by="count", ascending=False)
    print("\nПропуски:")
    display(miss_)
    
def df_dupl_feat(df):
    dupl_ = df.duplicated().sum()
    print(f"\nДубликатов: {dupl_}")
    
def df_isna_dupl_pruefen(df, name):
    print(f"*************   Анализ {name}   *************")
    df_miss_feat(df)
    df_dupl_feat(df)
        
def df_unique_sesseion(df1, name1, df2, name2):
    print("*************   Анализ session_id   *************\n")
    s1 = df1["session_id"].nunique()
    print(f"Уникальных session_id в {name1}: {s1}")
    l1 = len(df1)
    print(f"Строк в {name1}: {l1}\n")
    s2 = df2["session_id"].nunique()
    print(f"Уникальных session_id в {name2}: {s2}")
    
def df_actions(df, q_n):
    all_actions = (df["event_action"].value_counts().reset_index())
    all_actions.columns = ["event_action","frequency"]
    q = all_actions["frequency"].quantile(q_n)
    print("*************   Анализ event_action   *************\n")
    print(f"Уникальных типов событий: {len(all_actions)}")
    top_actions = all_actions[all_actions["frequency"] >= q]
    print(f"\nActions >= {int(q_n*100)}-му квантилю:")
    display(top_actions)

### Подготовка данных

def df_to_datetime(df1, df2):
    df1["visit_date"] = pd.to_datetime(df1["visit_date"])
    df1["visit_datetime"] = pd.to_datetime(
        df1["visit_date"].astype(str) + " " +
        df1["visit_time"].astype(str))
    df2["hit_date"] = pd.to_datetime(df2["hit_date"])
    print("Преобразование дат и времени завершено")
    
def df_datetime_split(df):
    df["visit_month"] = (df["visit_datetime"].dt.month)
    df["visit_dayofweek"] = (df["visit_datetime"].dt.dayofweek)
    df["visit_hour"] = (df["visit_datetime"].dt.hour)
    df["is_weekend"] = (df["visit_dayofweek"].isin([5, 6]).astype(int))
    print("Создание временных признаков завершено")
    
def df_visit_hour(df):
    df["is_night"] = (df["visit_hour"].between(0, 6)).astype(int)
    df["is_work_time"] = (df["visit_hour"].between(9, 18)).astype(int)
    df["is_evening"] = (df["visit_hour"].between(18, 23)).astype(int)
    print("Создание признаков времени посещения завершено")
    
def df_cat_luecken_ausfuellen(df, cat_col):
    for col in cat_col:
        if col in df.columns:
            df[f"{col}_missing"] = (df[col].isna().astype(int))
            df[col] = (df[col].fillna("unknown").astype(str))
    print("Пропуски заполнены")
            
def df_cat_reduce(df, column, top_n):
    top_cat = (df[column].value_counts().head(top_n).index)
    df[column] = np.where(df[column].isin(top_cat), df[column], "other")
    return df            

def df_top_n_pop(df, col_zu_red, top_n):
    for col in col_zu_red:
        if col in df.columns:
            df = df_cat_reduce(df, col, top_n)
    print(f"Колличество категорий сокращено до top-{top_n}")
    
def df_traffic(df):
    organic_mediums = ["organic", "referral", "(none)"]
    df["is_organic"] = (df["utm_medium"].isin(organic_mediums).astype(int))
    df["is_paid"] = ((df["is_organic"] == 0).astype(int))
    print("Трафик разделён по типам")
    
def df_add_target(df1, df2, target_actions):
    df2["target"] = (df2["event_action"].isin(target_actions).astype(int))
    session_target = (df2.groupby("session_id")["target"].max().reset_index())
    session_features = (df2.groupby("session_id").agg(
            hits_count=("hit_number", "count"),
            unique_pages=("hit_page_path", "nunique"),
            unique_events=("event_action", "nunique")
        ).reset_index())
    df = df1.merge(session_target, on="session_id", how="left")
    df = df.merge(session_features, on="session_id", how="left")
    df["target"] = (df["target"].fillna(0).astype(int))
    behavior_columns = ["hits_count", "unique_pages", "unique_events"]
    for column in behavior_columns:
        df[column] = df[column].fillna(0)
    return df

### EDA

def df_num_stat(df):
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_columns = [column for column in numeric_columns if column != "target"]
    print("\nОсновная статистика по числовым признакам:")
    display(df[numeric_columns].describe().T)
    return numeric_columns

def df_visit_stat(df):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    sns.histplot(
        data=df,
        x="visit_number",
        bins=50,
        kde=True,
        ax=axes[0]
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Распределение количества визитов")
    axes[0].set_xlabel("Количество визитов")
    axes[0].set_ylabel("Частота (log scale)")
    sns.boxplot(
        data=df,
        x="visit_number",
        ax=axes[1]
    )
    axes[1].set_title("BoxPlot признака visit_number")
    axes[1].set_xlabel("Количество визитов")
    plt.tight_layout()
    plt.show()
    print("*************   Распределение visits   *************")
    print("Min:", df["visit_number"].min())
    print("Median:", df["visit_number"].median())
    print("Mean:", round(df["visit_number"].mean(), 2))
    print("Max:", df["visit_number"].max())
    print("*************   Квантили   *************")
    display(df["visit_number"].quantile([0.5, 0.75, 0.90, 0.95, 0.99]))
    
def zeit_conv(df):
    time_columns = ["visit_month", "visit_dayofweek", "visit_hour"]
    time_columns = [column for column in time_columns if column in df.columns]
    for column in time_columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(
            data=df,
            x=column,
            hue="target",
            ax=ax
        )
        ax.set_yscale("log")
        ax.set_title(f"Распределение визитов по признаку {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Количество наблюдений (log scale)")
        plt.tight_layout()
        plt.show()
        
def traffic_conv(df):
    organic_mediums = ["organic", "referral", "(none)"]
    df = df.copy()
    df["traffic_type"] = np.where(df["utm_medium"].isin(organic_mediums), "organic", "paid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(
        data=df,
        x="traffic_type",
        hue="target",
        ax=ax
    )
    ax.set_yscale("log")
    ax.set_title("Распределение organic / paid traffic")
    ax.set_xlabel("Тип трафика")
    ax.set_ylabel("Количество наблюдений (log scale)")
    plt.tight_layout()
    plt.show()
    print("*************   Конверсия/Трафик   *************")
    traffic_conv = df.groupby('is_paid')['target'].mean() * 100
    print(f"Organic:     {traffic_conv[0]:.2f}%")
    print(f"Paid:        {traffic_conv[1]:.2f}%")
    
def df_traffic_quelle(df):
    if "utm_medium" in df.columns:
        conversion_by_medium = (df.groupby("utm_medium")["target"].mean().sort_values(ascending=False).head(15))
        fig, ax = plt.subplots(figsize=(10, 5))
        conversion_by_medium.sort_values().plot(
            kind="barh",
            ax=ax
        )
        ax.set_title("Конверсия по utm_medium")
        ax.set_xlabel("Средний target")
        ax.set_ylabel("utm_medium")
        plt.tight_layout()
        plt.show()
        
def df_device_cat(df):
    if "device_category" in df.columns:
        conversion_by_device = (df.groupby("device_category")["target"].mean().sort_values(ascending=False))
        fig, ax = plt.subplots(figsize=(8, 4))
        conversion_by_device.sort_values().plot(
            kind="barh",
            ax=ax
        )
        ax.set_title("Конверсия по типу устройства")
        ax.set_xlabel("Средний target")
        ax.set_ylabel("Тип устройства")
        plt.tight_layout()
        plt.show()
        
def df_corr(df, corr_col):
    corr_matrix = df[corr_col].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )
    ax.set_title("Тепловая карта корреляции")
    plt.tight_layout()
    plt.show()

### Feature Engineering

def df_clip(df, col_to_clip, q):
    for col in col_to_clip:
        upper_limit = (df[col].quantile(q))
        df[f"{col}_clipped"] = (df[col].clip(upper=upper_limit))
        df[f"{col}_is_outlier"] = (df[col] > upper_limit).astype(int)
        print(f"{col}: верхняя граница = {round(upper_limit, 2)}")
    
def df_col_to_log(df, numeric_columns):
    col_to_log = []
    for column in numeric_columns:
        if (
            df[column].nunique() > 2
            and (df[column] >= 0).all()
            and abs(df[column].skew()) > 1
        ):
            col_to_log.append(column)
    return col_to_log 
    
def df_log(df, col_to_log):
    log_columns = [col for col in col_to_log if col in df.columns]
    for column in log_columns:
        df[f"{column}_log"] = (np.log1p(df[column]))
        df.drop(columns=column, inplace=True)
        print(f"{column} -> {column}_log")
    print("\nПризнаки логарифмированы")

### ML

def df_train_source(df):       
    df_train_source, _ = train_test_split(
        df,
        train_size=0.25,
        random_state=42,
        stratify=df["target"]
    )
    df_train_source = df_train_source.reset_index(drop=True)

    print("Размер данных для обучения:", df_train_source.shape)
    target_column = "target"
    drop_columns = [
        "target",
        "session_id",
        "client_id"
    ]
    drop_columns = [
        column for column in drop_columns
        if column in df_train_source.columns
    ]
    y = df_train_source[target_column]
    X = df_train_source.drop(columns=drop_columns)
    return X, y

def df_tt_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    return X_train, X_test, y_train, y_test

def df_features_to_list(X):
    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("Числовых признаков:", len(numeric_features))
    print("Категориальных признаков:", len(categorical_features))
    return numeric_features, categorical_features
  
def ml_preproc_pipe(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        sparse_threshold=0.3
    )
    return preprocessor

def ml_lr(X_train, X_test, y_train, y_test, preprocessor):
    logreg_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]
    )
    logreg_model.fit(X_train, y_train)
    logreg_proba = logreg_model.predict_proba(X_test)[:, 1]
    logreg_pred = logreg_model.predict(X_test)
    logreg_roc_auc = roc_auc_score(y_test, logreg_proba)
    print("\nLOGISTIC REGRESSION")
    print("ROC-AUC:", round(logreg_roc_auc, 4))
    print(classification_report(y_test, logreg_pred))
    return logreg_model, logreg_roc_auc, logreg_pred, logreg_proba
    
def ml_rf(X_train, X_test, y_train, y_test, preprocessor):
    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]
    )
    rf_model.fit(X_train, y_train)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = rf_model.predict(X_test)
    rf_roc_auc = roc_auc_score(y_test, rf_proba)
    print("\nRANDOM FOREST")
    print("ROC-AUC:", round(rf_roc_auc, 4))
    print(classification_report(y_test, rf_pred))
    return rf_model, rf_roc_auc, rf_pred, rf_proba

def ml_cb_vorbereitung(X, y, categorical_features, numeric_features):
    X_catboost = X.copy()
    for column in categorical_features:
        X_catboost[column] = X_catboost[column].fillna("unknown").astype(str)
    for column in numeric_features:
        X_catboost[column] = X_catboost[column].fillna(X_catboost[column].median())
    X_train_cb, X_test_cb, y_train_cb, y_test_cb = train_test_split(
        X_catboost,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    cat_features_indices = [
        X_catboost.columns.get_loc(column)
        for column in categorical_features
    ]
    return X_train_cb, X_test_cb, y_train_cb, y_test_cb, cat_features_indices

def ml_cb(X_train_cb, X_test_cb, y_train_cb, y_test_cb, cat_features_indices):
    catboost_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=100
    )
    catboost_model.fit(
        X_train_cb,
        y_train_cb,
        cat_features=cat_features_indices,
        eval_set=(X_test_cb, y_test_cb),
        early_stopping_rounds=50
    )
    catboost_proba = catboost_model.predict_proba(X_test_cb)[:, 1]
    catboost_pred = catboost_model.predict(X_test_cb)
    catboost_roc_auc = roc_auc_score(y_test_cb, catboost_proba)
    print("\nCATBOOST")
    print("ROC-AUC:", round(catboost_roc_auc, 4))
    print(classification_report(y_test_cb, catboost_pred))
    return catboost_model, catboost_roc_auc, catboost_pred, catboost_proba

def ml_results_vergleich(y_test, y_test_cb, logreg_roc_auc, logreg_proba, rf_roc_auc, rf_proba, catboost_roc_auc, catboost_proba):
    models_results = pd.DataFrame({
        "model": [
            "Logistic Regression",
            "Random Forest",
            "CatBoost"
        ],
        "roc_auc": [
            logreg_roc_auc,
            rf_roc_auc,
            catboost_roc_auc
        ]
    }).sort_values(by="roc_auc", ascending=False)
    display(models_results)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        y_test,
        logreg_proba,
        name="Logistic Regression",
        ax=ax
    )
    RocCurveDisplay.from_predictions(
        y_test,
        rf_proba,
        name="Random Forest",
        ax=ax
    )
    RocCurveDisplay.from_predictions(
        y_test_cb,
        catboost_proba,
        name="CatBoost",
        ax=ax
    )
    ax.set_title("ROC-кривые моделей")
    ax.grid()
    plt.tight_layout()
    plt.show()
    return models_results

def ml_best_model(models_results, y_test, y_test_cb,
                  catboost_model, catboost_proba, catboost_pred,
                  rf_model, rf_proba, rf_pred,
                  logreg_model, logreg_proba, logreg_pred):
    best_model_name = models_results.iloc[0]["model"]
    if best_model_name == "CatBoost":
        best_model = catboost_model
        best_proba = catboost_proba
        best_pred = catboost_pred
        best_y_test = y_test_cb
    elif best_model_name == "Random Forest":
        best_model = rf_model
        best_proba = rf_proba
        best_pred = rf_pred
        best_y_test = y_test
    else:
        best_model = logreg_model
        best_proba = logreg_proba
        best_pred = logreg_pred
        best_y_test = y_test
    print("Лучшая модель:", best_model_name)
    print("ROC-AUC:", round(roc_auc_score(best_y_test, best_proba), 4))
    return best_model_name, best_model, best_pred, best_proba, best_y_test

def ml_conf_matrix(best_model_name, best_y_test, best_pred):
    cm = confusion_matrix(best_y_test, best_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1]
    )
    disp.plot()
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.show()

def ml_features_importance(best_model_name, best_model, X_train_cb):
    if best_model_name == "CatBoost":
        feature_names = X_train_cb.columns
        feature_importances = best_model.get_feature_importance()
        importance_df = pd.DataFrame({"feature": feature_names,"importance": feature_importances}).sort_values(by="importance", ascending=False)
    else:
        feature_names = (best_model.named_steps["preprocessor"].get_feature_names_out())
        model_step = best_model.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            feature_importances = model_step.feature_importances_
        elif hasattr(model_step, "coef_"):
            feature_importances = np.abs(model_step.coef_[0])
        else:
            feature_importances = None
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importances
        }).sort_values(by="importance", ascending=False)
    print("\nТОП-20 важных признаков:")
    display(importance_df.head(20))
    top_features = (importance_df.head(20).sort_values(by="importance"))
    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"],top_features["importance"])
    plt.title(f"Топ-20 важных признаков: {best_model_name}")
    plt.xlabel("Важность признака")
    plt.ylabel("Признак")
    plt.tight_layout()
    plt.show()

### Сохранение модели

def model_speichern(best_model, best_model_name, best_y_test, best_proba):
    model_filename = f"model_{best_model_name}.pkl"
    joblib.dump(best_model, model_filename)
    print(f"Финальная модель: {best_model_name}")
    print(f"Файл модели: {model_filename}")
    print("Финальный ROC-AUC на test:", round(roc_auc_score(best_y_test, best_proba), 4))
