LANDGUARD AI – FULL-STACK UPGRADE
================================

WHAT IS INCLUDED
- Existing LANDGUARD map and live Open-Meteo weather panel.
- AI Prediction Center UI.
- FastAPI backend with /api/status and /api/predict.
- Random Forest training pipeline (train_model.py).
- CSV schema template.

IMPORTANT HONESTY NOTE
The ZIP you supplied contained only a static HTML site and README; it did not contain a trained model or labelled research dataset. Therefore this package does NOT pretend that a real AI model is already trained. The AI panel will clearly report that no trained model is installed until you add verified data and train it.

MODEL TARGET
This first ML module estimates landslide susceptibility from supplied environmental features. It is NOT a short-term landslide event forecast and its score is NOT the probability that a landslide will happen. A true time-specific forecast requires time-stamped rainfall/soil conditions, event dates, non-event windows, and rigorous temporal/spatial validation.

FEATURES EXPECTED
rainfall_24h_mm, rainfall_72h_mm, slope_deg, elevation_m,
soil_moisture_pct, distance_to_road_m, ndvi, lithology_code, label

label = 1 for a documented landslide/susceptible sample; 0 for a verified stable/non-landslide sample.
Do not invent rows or label every unreported location as stable. Lithology codes must use a consistent documented mapping. Features must be extracted for both positive and negative samples from the same study area and time methodology.

RUN LOCALLY
1. Install Python 3.10+.
2. Open terminal in this folder.
3. Run:  python -m venv .venv
   Windows: .venv\Scripts\activate
   Mac/Linux: source .venv/bin/activate
4. Run:  pip install -r requirements.txt
5. Put your quality-checked dataset at data/training_data.csv (same columns as data/training_data_template.csv).
6. Run:  python train_model.py
7. Run:  uvicorn app:app --reload
8. Open: http://127.0.0.1:8000

DEPLOYMENT
This is now a full-stack app, not a static-only Netlify folder. Netlify can host the frontend, but it cannot run this Python API by itself. Deploy the FastAPI service to a Python host (for example Render) and configure the frontend API base URL if hosting separately. Easiest first test is to run the full app locally as above.

VALIDATION REQUIRED BEFORE PRESENTING AS REAL-WORLD AI
- Verify provenance and licences of every dataset.
- Use spatially separated train/test regions (not random nearby points).
- Check class balance, false negatives, calibration and performance by district.
- Have a landslide/geology expert review labels and features.
- Compare with official GSI/IMD/district-administration information.
- Never use this prototype for emergency decisions.

CURRENT SITE LIMITATIONS RETAINED
Map pins remain approximate district reference points, not official district polygons. Live weather remains an Open-Meteo API integration. No official alerting system is included.
