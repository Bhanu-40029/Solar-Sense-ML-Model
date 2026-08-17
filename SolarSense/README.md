# SolarSense — Frontend

An AI-powered solar plant monitoring and inverter analytics platform.
This repository contains **only the frontend**: Flask page routing, Jinja2
templates, CSS and JavaScript. Your existing Python ML / data-processing /
EDA code is **not** included and is not reimplemented here — this project
gives you clean integration points to plug it in later.

## Stack

- Flask (routing + Jinja2 templates)
- HTML / CSS / vanilla JavaScript (no React, no Node, no Bootstrap/Tailwind)

## Folder structure

```
SolarSense/
├── app.py                     Flask app: page routes + mock API routes
├── static/
│   ├── css/style.css          Full design system (dark control-room theme)
│   ├── js/main.js             Frontend-only interactions (search, tabs, forms...)
│   └── images/                Static image assets
├── templates/
│   ├── base.html              Shared layout: sidebar, header, footer
│   ├── home.html               Landing page
│   ├── dashboard.html          KPIs, charts, activity feed
│   ├── dataset.html            Dataset upload + preview table
│   ├── data_understanding.html Column & statistical summary
│   ├── visualization.html      EDA image gallery (tabs, placeholders)
│   ├── preprocessing.html      Preprocessing pipeline visualization
│   ├── models.html             Model cards + evaluation/comparison
│   ├── prediction.html         AC power prediction form + result panel
│   ├── inverter_health.html    Inverter fleet grid + detail panel
│   ├── reports.html            Report summaries + export buttons
│   └── about.html              Project explanation
├── uploads/                    Where uploaded datasets can be stored
└── README.md
```

## Running it

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install Flask:
   ```bash
   pip install flask
   ```

3. Run the app:
   ```bash
   python app.py
   ```

4. Open **http://127.0.0.1:5000** in your browser.

## Connecting your existing Python code

`app.py` contains placeholder API routes that currently return mock JSON:

| Route                 | Purpose                              |
|------------------------|---------------------------------------|
| `/api/dataset`         | Dataset metadata                      |
| `/api/eda`              | EDA image manifest                    |
| `/api/preprocessing`    | Preprocessing pipeline stats          |
| `/api/models`           | Model list + metrics                  |
| `/api/predict`          | AC power prediction (POST)            |
| `/api/inverter-health`  | Per-inverter health scoring           |
| `/api/reports`          | Report generation / export            |

Each route has a `# TODO: connect ...` comment marking where to call into
your existing modules. Replace the mock `jsonify(...)` payload with the real
output of your Python code — the response **shape** (field names) already
matches what the frontend JavaScript expects, so no template changes should
be required for a basic integration.

For EDA images: your Python EDA module should write PNGs to `outputs/EDA/`
(e.g. `outputs/EDA/ac_power_distribution.png`). The `visualization.html`
page already references files at that path and falls back to an
"Analysis not available yet" placeholder if a file is missing. You'll need
to add a Flask route (or static file mapping) to serve `outputs/EDA/` if it
lives outside of `static/`.

## Notes

- No ML, preprocessing, or EDA logic is implemented in this repo — every
  number you see (KPIs, model metrics, inverter stats, dataset preview
  rows) is mock/sample data for layout purposes only.
- All JavaScript in `static/js/main.js` is frontend-only: search, filtering,
  pagination, tabs, and form submission all currently operate on mock data
  or hard-coded DOM content, not real API calls (except the `/api/predict`
  wiring point noted above).
