"""
======================================================================
                    SOLARSENSE - FLASK APPLICATION
======================================================================

SolarSense is a solar power monitoring and machine-learning platform.

Current responsibilities of this Flask application:

    1. Render frontend pages
    2. Provide frontend API endpoints
    3. Read the SolarSense dataset
    4. Provide dataset information to the frontend
    5. Provide EDA images to the frontend
    6. Provide preprocessing information
    7. Provide placeholders for ML prediction
    8. Provide placeholders for inverter health
    9. Connect the frontend with existing Python modules later

IMPORTANT:

The actual ML / EDA / preprocessing logic remains inside:

    src/
        Cleaning.py
        Dataset_load.py
        Dataset_M1_Understanding.py
        EDA_Analysis.py
        Final_Preprocessing.py

Do NOT put the actual ML logic inside this file.

======================================================================
"""

# ======================================================================
# IMPORTS
# ======================================================================

import os
from datetime import datetime

import pandas as pd

from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    send_from_directory
)


# ======================================================================
# 1. PROJECT BASE DIRECTORY
# ======================================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ======================================================================
# 2. FLASK APPLICATION
# ======================================================================

app = Flask(
    __name__,

    # Your HTML files are inside:
    #
    # ML Solor Sense/
    # └── SolarSense/
    #     └── templates/

    template_folder=os.path.join(
        BASE_DIR,
        "SolarSense",
        "templates"
    ),

    # Your CSS / JS files are inside:
    #
    # ML Solor Sense/
    # └── SolarSense/
    #     └── static/

    static_folder=os.path.join(
        BASE_DIR,
        "SolarSense",
        "static"
    )
)


# ======================================================================
# 3. PROJECT DIRECTORIES
# ======================================================================

# Main dataset folder

DATASET_DIR = os.path.join(
    BASE_DIR,
    "Datasets"
)


# EDA output folder

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "Outputs"
)


EDA_DIR = os.path.join(
    OUTPUT_DIR,
    "EDA"
)


# Frontend upload folder

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "SolarSense",
    "uploads"
)


# Existing Python source folder

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)


# Make sure upload directory exists

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ======================================================================
# 4. DATASET FILE PATHS
# ======================================================================

# Original generation dataset

RAW_GENERATION_DATASET = os.path.join(
    DATASET_DIR,
    "Plant_1_Generation_Data.csv"
)


# Original weather dataset

RAW_WEATHER_DATASET = os.path.join(
    DATASET_DIR,
    "Plant_1_Weather_Sensor_Data.csv"
)


# Cleaned dataset used by SolarSense

CLEANED_DATASET = os.path.join(
    DATASET_DIR,
    "SolarSense_Cleaned_Data.csv"
)


# Day/night balanced dataset

DAY_NIGHT_DATASET = os.path.join(
    DATASET_DIR,
    "SolarSense_Day_Night_Balanced_Raw_Data.csv"
)


# Final preprocessed dataset

FINAL_PREPROCESSED_DATASET = os.path.join(
    DATASET_DIR,
    "SolarSense_Final_Preprocessed_Data.csv"
)


# ======================================================================
# 5. PROJECT COLUMN DEFINITIONS
# ======================================================================

NUMERICAL_COLUMNS = [

    "DC_POWER",

    "DAILY_YIELD",

    "TOTAL_YIELD",

    "AMBIENT_TEMPERATURE",

    "MODULE_TEMPERATURE",

    "IRRADIATION",

    "AC_POWER"

]


CATEGORICAL_COLUMNS = [

    "INVERTER_ID",

    "WEATHER_SENSOR_ID",

    "PERIOD"

]


DATETIME_COLUMN = "DATE_TIME"


TARGET_COLUMN = "AC_POWER"


# ======================================================================
# 6. HELPER FUNCTION
# ======================================================================

def load_cleaned_dataset():

    """
    Load the SolarSense cleaned dataset.

    Returns:
        pandas.DataFrame
        None if the dataset does not exist.
    """

    if not os.path.exists(
        CLEANED_DATASET
    ):

        return None

    try:

        df = pd.read_csv(
            CLEANED_DATASET
        )

        return df

    except Exception as error:

        print(
            "Error loading cleaned dataset:",
            error
        )

        return None


# ======================================================================
# 7. PAGE ROUTES
# ======================================================================


@app.route("/")
def home():

    return render_template(
        "home.html",
        active_page="home"
    )


# ----------------------------------------------------------------------

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        active_page="dashboard"
    )


# ----------------------------------------------------------------------

@app.route("/dataset")
def dataset():

    return render_template(
        "dataset.html",
        active_page="dataset"
    )


# ----------------------------------------------------------------------

@app.route("/data-understanding")
def data_understanding():

    return render_template(
        "data_understanding.html",
        active_page="data-understanding"
    )


# ----------------------------------------------------------------------

@app.route("/visualization")
def visualization():

    return render_template(
        "visualization.html",
        active_page="visualization"
    )


# ----------------------------------------------------------------------

@app.route("/preprocessing")
def preprocessing():

    return render_template(
        "preprocessing.html",
        active_page="preprocessing"
    )


# ----------------------------------------------------------------------

@app.route("/models")
def models():

    return render_template(
        "models.html",
        active_page="models"
    )


# ----------------------------------------------------------------------

@app.route("/prediction")
def prediction():

    return render_template(
        "prediction.html",
        active_page="prediction"
    )


# ----------------------------------------------------------------------

@app.route("/inverter-health")
def inverter_health():

    return render_template(
        "inverter_health.html",
        active_page="inverter-health"
    )


# ----------------------------------------------------------------------

@app.route("/reports")
def reports():

    return render_template(
        "reports.html",
        active_page="reports"
    )


# ----------------------------------------------------------------------

@app.route("/about")
def about():

    return render_template(
        "about.html",
        active_page="about"
    )


# ----------------------------------------------------------------------

@app.route("/contact")
def contact():

    return render_template(
        "contact.html",
        active_page="contact"
    )


# ======================================================================
# 8. APPLICATION HEALTH API
# ======================================================================

@app.route("/api/health")
def api_health():

    """
    Basic API health check.
    """

    return jsonify({

        "status": "online",

        "application": "SolarSense",

        "timestamp":
            datetime.now().isoformat()

    })


# ======================================================================
# 9. DATASET INFORMATION API
# ======================================================================

@app.route("/api/dataset")
def api_dataset():

    """
    Return information about the cleaned SolarSense dataset.
    """

    df = load_cleaned_dataset()


    # --------------------------------------------------------------
    # Dataset not found
    # --------------------------------------------------------------

    if df is None:

        return jsonify({

            "status": "not_available",

            "message":
                "SolarSense_Cleaned_Data.csv was not found.",

            "dataset_path":
                CLEANED_DATASET

        })


    # --------------------------------------------------------------
    # Date range
    # --------------------------------------------------------------

    date_range = {

        "start": None,

        "end": None

    }


    if DATETIME_COLUMN in df.columns:

        try:

            dates = pd.to_datetime(
                df[DATETIME_COLUMN],
                errors="coerce"
            )


            dates = dates.dropna()


            if len(dates) > 0:

                date_range["start"] = (

                    dates.min()
                    .strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                )


                date_range["end"] = (

                    dates.max()
                    .strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                )

        except Exception:

            pass


    # --------------------------------------------------------------
    # Inverter count
    # --------------------------------------------------------------

    inverter_count = 0


    if "INVERTER_ID" in df.columns:

        inverter_count = int(

            df["INVERTER_ID"]
            .nunique()

        )


    # --------------------------------------------------------------
    # Weather sensor count
    # --------------------------------------------------------------

    weather_sensor_count = 0


    if "WEATHER_SENSOR_ID" in df.columns:

        weather_sensor_count = int(

            df["WEATHER_SENSOR_ID"]
            .nunique()

        )


    # --------------------------------------------------------------
    # Missing values
    # --------------------------------------------------------------

    missing_values = int(

        df.isnull()
        .sum()
        .sum()

    )


    # --------------------------------------------------------------
    # Duplicate records
    # --------------------------------------------------------------

    duplicate_records = int(

        df.duplicated()
        .sum()

    )


    # --------------------------------------------------------------
    # Dataset size
    # --------------------------------------------------------------

    size_mb = 0

    try:

        size_bytes = os.path.getsize(
            CLEANED_DATASET
        )

        size_mb = round(
            size_bytes / (1024 * 1024),
            3
        )

    except Exception:

        size_mb = 0


    # --------------------------------------------------------------
    # Return information
    # --------------------------------------------------------------

    return jsonify({

        "status": "ok",

        "filename":
            os.path.basename(
                CLEANED_DATASET
            ),

        "rows":
            int(df.shape[0]),

        "columns":
            int(df.shape[1]),

        "size_mb":
            size_mb,

        "column_names":
            df.columns.tolist(),

        "inverters":
            inverter_count,

        "weather_sensors":
            weather_sensor_count,

        "missing_values":
            missing_values,

        "duplicate_records":
            duplicate_records,

        "date_range":
            date_range

    })


# ======================================================================
# 10. DATASET PREVIEW API
# ======================================================================

@app.route("/api/dataset/preview")
def api_dataset_preview():

    """
    Return a preview of the cleaned dataset.
    """

    df = load_cleaned_dataset()


    if df is None:

        return jsonify({

            "status": "error",

            "message":
                "Cleaned dataset not found."

        }), 404


    # Number of rows requested

    rows = request.args.get(

        "rows",

        default=10,

        type=int

    )


    # Prevent extremely large requests

    rows = max(
        1,
        min(rows, 100)
    )


    preview = df.head(
        rows
    ).copy()


    # Convert NaN to None
    # so JSON can handle it.

    preview = preview.where(
        pd.notnull(preview),
        None
    )


    return jsonify({

        "status": "ok",

        "columns":
            preview.columns.tolist(),

        "data":
            preview.to_dict(
                orient="records"
            )

    })


# ======================================================================
# 11. DATASET COLUMN INFORMATION API
# ======================================================================

@app.route("/api/dataset/columns")
def api_dataset_columns():

    """
    Return:

        Column name
        Data type
        Category
        Missing values
        Unique values
    """

    df = load_cleaned_dataset()


    if df is None:

        return jsonify({

            "status": "error",

            "message":
                "Cleaned dataset not found."

        }), 404


    column_information = []


    for column in df.columns:


        # ----------------------------------------------------------
        # Identify category
        # ----------------------------------------------------------

        if column in NUMERICAL_COLUMNS:

            category = "Numerical"


        elif column in CATEGORICAL_COLUMNS:

            category = "Categorical"


        elif column == DATETIME_COLUMN:

            category = "Date-Time"


        else:

            category = "Other"


        # ----------------------------------------------------------
        # Column information
        # ----------------------------------------------------------

        column_information.append({

            "column":
                column,

            "data_type":
                str(
                    df[column].dtype
                ),

            "category":
                category,

            "missing_values":
                int(
                    df[column]
                    .isnull()
                    .sum()
                ),

            "unique_values":
                int(
                    df[column]
                    .nunique()
                )

        })


    return jsonify({

        "status": "ok",

        "columns":
            column_information

    })


# ======================================================================
# 12. DATASET STATISTICS API
# ======================================================================

@app.route("/api/dataset/statistics")
def api_dataset_statistics():

    """
    Return descriptive statistics for numerical columns.
    """

    df = load_cleaned_dataset()


    if df is None:

        return jsonify({

            "status": "error",

            "message":
                "Cleaned dataset not found."

        }), 404


    available_columns = [

        column

        for column in NUMERICAL_COLUMNS

        if column in df.columns

    ]


    if not available_columns:

        return jsonify({

            "status": "ok",

            "statistics": {}

        })


    statistics = (

        df[available_columns]
        .describe()
        .round(4)

    )


    return jsonify({

        "status": "ok",

        "statistics":
            statistics.to_dict()

    })


# ======================================================================
# 13. EDA API
# ======================================================================

@app.route("/api/eda")
def api_eda():

    """
    Return all available EDA images.

    Images are generated by your existing:

        src/EDA_Analysis.py

    and stored in:

        Outputs/EDA/
    """

    if not os.path.exists(
        EDA_DIR
    ):

        return jsonify({

            "status":
                "not_available",

            "message":
                "Outputs/EDA directory was not found.",

            "images": []

        })


    allowed_extensions = (

        ".png",

        ".jpg",

        ".jpeg",

        ".webp"

    )


    images = []


    for filename in sorted(
        os.listdir(EDA_DIR)
    ):


        if filename.lower().endswith(
            allowed_extensions
        ):

            images.append({

                "name":
                    filename,

                "url":
                    f"/api/eda/image/{filename}"

            })


    return jsonify({

        "status": "ok",

        "count":
            len(images),

        "images":
            images

    })


# ======================================================================
# 14. EDA IMAGE API
# ======================================================================

@app.route(
    "/api/eda/image/<path:filename>"
)
def api_eda_image(filename):

    """
    Serve an EDA image from:

        Outputs/EDA/
    """

    return send_from_directory(

        EDA_DIR,

        filename

    )


# ======================================================================
# 15. PREPROCESSING API
# ======================================================================

@app.route("/api/preprocessing")
def api_preprocessing():

    """
    Return information about the preprocessing stage.

    Actual preprocessing is performed by:

        src/Final_Preprocessing.py
    """

    cleaned_exists = os.path.exists(
        CLEANED_DATASET
    )


    preprocessed_exists = os.path.exists(
        FINAL_PREPROCESSED_DATASET
    )


    cleaned_rows = 0

    preprocessed_rows = 0


    # --------------------------------------------------------------
    # Cleaned dataset
    # --------------------------------------------------------------

    if cleaned_exists:

        try:

            cleaned_df = pd.read_csv(
                CLEANED_DATASET
            )

            cleaned_rows = int(
                len(cleaned_df)
            )

        except Exception as error:

            print(
                "Error reading cleaned dataset:",
                error
            )


    # --------------------------------------------------------------
    # Final preprocessed dataset
    # --------------------------------------------------------------

    if preprocessed_exists:

        try:

            preprocessed_df = pd.read_csv(
                FINAL_PREPROCESSED_DATASET
            )

            preprocessed_rows = int(
                len(preprocessed_df)
            )

        except Exception as error:

            print(
                "Error reading preprocessed dataset:",
                error
            )


    return jsonify({

        "status": "ok",

        "cleaned_dataset_available":
            cleaned_exists,

        "final_preprocessed_dataset_available":
            preprocessed_exists,

        "records_before":
            cleaned_rows,

        "records_after":
            preprocessed_rows,


        "numerical_preprocessing": {

            "method":
                "StandardScaler",

            "columns":

                [

                    column

                    for column
                    in NUMERICAL_COLUMNS

                ]

        },


        "categorical_preprocessing": {

            "INVERTER_ID":
                "One-Hot Encoding",

            "PERIOD":
                "Binary Encoding"

        },


        "datetime_preprocessing": [

            "DATE_TIME",

            "HOUR",

            "MINUTE",

            "DAY",

            "MONTH",

            "DAY_OF_WEEK",

            "TIME_OF_DAY",

            "TIME_SIN",

            "TIME_COS"

        ],


        "target":
            TARGET_COLUMN

    })


# ======================================================================
# 16. FINAL PREPROCESSED DATASET PREVIEW API
# ======================================================================

@app.route(
    "/api/preprocessing/preview"
)
def api_preprocessing_preview():

    """
    Return preview of final preprocessed dataset.
    """

    if not os.path.exists(
        FINAL_PREPROCESSED_DATASET
    ):

        return jsonify({

            "status":
                "not_available",

            "message":
                "Final preprocessed dataset not found.",

            "path":
                FINAL_PREPROCESSED_DATASET

        }), 404


    try:

        df = pd.read_csv(
            FINAL_PREPROCESSED_DATASET
        )


        rows = request.args.get(

            "rows",

            default=10,

            type=int

        )


        rows = max(
            1,
            min(rows, 100)
        )


        preview = df.head(
            rows
        ).copy()


        preview = preview.where(
            pd.notnull(preview),
            None
        )


        return jsonify({

            "status":
                "ok",

            "columns":
                preview.columns.tolist(),

            "data":
                preview.to_dict(
                    orient="records"
                )

        })


    except Exception as error:

        return jsonify({

            "status":
                "error",

            "message":
                str(error)

        }), 500


# ======================================================================
# 17. MACHINE LEARNING MODELS API
# ======================================================================

@app.route("/api/models")
def api_models():

    """
    Model information endpoint.

    The actual model training will later be connected
    to your ML Python file.

    Target:

        AC_POWER

    This endpoint intentionally does NOT generate
    fake model metrics.
    """

    return jsonify({

        "status":
            "pending",

        "target":
            "AC_POWER",

        "purpose":
            "Predict expected AC power",

        "models": [

            {

                "name":
                    "Linear Regression",

                "status":
                    "Not connected"

            },

            {

                "name":
                    "Random Forest",

                "status":
                    "Not connected"

            },

            {

                "name":
                    "Gradient Boosting",

                "status":
                    "Not connected"

            },

            {

                "name":
                    "XGBoost",

                "status":
                    "Not connected"

            }

        ],

        "best_model":
            None

    })


# ======================================================================
# 18. AC POWER PREDICTION API
# ======================================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    """
    AC Power prediction endpoint.

    This currently validates the frontend input.

    Later this will call the actual trained ML model.

    Main target:

        AC_POWER
    """

    payload = request.get_json(
        silent=True
    ) or {}


    required_fields = [

        "inverter_id",

        "irradiation",

        "ambient_temperature",

        "module_temperature",

        "date",

        "time"

    ]


    missing_fields = [

        field

        for field in required_fields

        if field not in payload

        or payload[field] in (
            "",
            None
        )

    ]


    if missing_fields:

        return jsonify({

            "status":
                "error",

            "message":
                "Required prediction inputs are missing.",

            "missing_fields":
                missing_fields

        }), 400


    # --------------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------------
    #
    # DO NOT generate a random prediction.
    #
    # Later we will connect:
    #
    #     src/Prediction.py
    #
    # to this endpoint.
    #
    # Example future logic:
    #
    # from src.Prediction import predict_ac_power
    #
    # result = predict_ac_power(payload)
    #
    # --------------------------------------------------------------


    return jsonify({

        "status":
            "pending",

        "message":
            "AC_POWER prediction model is not connected yet.",

        "target":
            "AC_POWER",

        "input":
            payload

    })


# ======================================================================
# 19. INVERTER HEALTH API
# ======================================================================

@app.route(
    "/api/inverter-health"
)
def api_inverter_health():

    """
    Inverter health will be determined using:

        Actual AC Power
                 VS
        Expected AC Power

    Expected AC Power will come from the ML model.

    Health states:

        NORMAL
        WARNING
        ABNORMAL

    The actual calculation will be connected later
    to the inverter-health Python module.
    """

    return jsonify({

        "status":
            "pending",

        "message":
            "Inverter health model is not connected yet.",

        "health_logic": {

            "actual_power":
                "AC_POWER",

            "expected_power":
                "ML predicted AC_POWER",

            "comparison":
                "Actual AC Power vs Expected AC Power",

            "statuses": [

                "NORMAL",

                "WARNING",

                "ABNORMAL"

            ]

        },

        "inverters": []

    })


# ======================================================================
# 20. MODEL EVALUATION API
# ======================================================================

@app.route(
    "/api/model-evaluation"
)
def api_model_evaluation():

    """
    Placeholder for actual ML model evaluation.

    Metrics:

        R²
        MAE
        MSE
        RMSE
    """

    return jsonify({

        "status":
            "pending",

        "target":
            "AC_POWER",

        "metrics": [

            "R2",

            "MAE",

            "MSE",

            "RMSE"

        ],

        "models": []

    })


# ======================================================================
# 21. REPORTS API
# ======================================================================

@app.route(
    "/api/reports"
)
def api_reports():

    """
    Return available SolarSense report sections.
    """

    return jsonify({

        "status":
            "ok",

        "reports_available": [

            "dataset_summary",

            "data_understanding",

            "eda_summary",

            "preprocessing_summary",

            "model_performance",

            "prediction_summary",

            "inverter_health",

            "abnormal_performance"

        ]

    })


# ======================================================================
# 22. ERROR HANDLER - 404
# ======================================================================

@app.errorhandler(404)
def page_not_found(error):

    """
    Handle missing pages.
    """

    try:

        return render_template(
            "error.html",
            error_code=404,
            error_message="Page not found."
        ), 404

    except Exception:

        return jsonify({

            "status":
                "error",

            "message":
                "Page not found."

        }), 404


# ======================================================================
# 23. ERROR HANDLER - 500
# ======================================================================

@app.errorhandler(500)
def internal_server_error(error):

    """
    Handle internal server errors.
    """

    return jsonify({

        "status":
            "error",

        "message":
            "Internal server error."

    }), 500


# ======================================================================
# 24. START FLASK APPLICATION
# ======================================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("              SOLARSENSE FLASK APPLICATION")
    print("=" * 70)

    print()

    print("Project Directory:")
    print(BASE_DIR)

    print()

    print("Frontend Templates:")
    print(
        os.path.join(
            BASE_DIR,
            "SolarSense",
            "templates"
        )
    )

    print()

    print("Frontend Static Files:")
    print(
        os.path.join(
            BASE_DIR,
            "SolarSense",
            "static"
        )
    )

    print()

    print("Dataset Directory:")
    print(DATASET_DIR)

    print()

    print("EDA Output Directory:")
    print(EDA_DIR)

    print()

    print("Source Directory:")
    print(SRC_DIR)

    print()

    print("-" * 70)

    print(
        "Cleaned Dataset Exists:",
        os.path.exists(
            CLEANED_DATASET
        )
    )

    print(
        "Final Preprocessed Dataset Exists:",
        os.path.exists(
            FINAL_PREPROCESSED_DATASET
        )
    )

    print(
        "EDA Directory Exists:",
        os.path.exists(
            EDA_DIR
        )
    )

    print()

    print("=" * 70)
    print("Starting SolarSense Flask Server")
    print("URL: http://127.0.0.1:5000")
    print("=" * 70)
    print()


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )