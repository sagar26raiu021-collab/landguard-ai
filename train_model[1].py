"""Train LANDGUARD Random Forest on a verified, labelled CSV.
Required columns: rainfall_24h_mm,rainfall_72h_mm,slope_deg,elevation_m,
soil_moisture_pct,distance_to_road_m,ndvi,lithology_code,label
label: 1 = mapped landslide/susceptible sample; 0 = verified non-landslide/stable sample.
Use spatially independent train/test splits for scientific evaluation before public use.
"""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"training_data.csv"
OUT=ROOT/"model"
FEATURES=["rainfall_24h_mm","rainfall_72h_mm","slope_deg","elevation_m",
          "soil_moisture_pct","distance_to_road_m","ndvi","lithology_code"]
REQUIRED=FEATURES+["label"]
df=pd.read_csv(DATA)
missing=[c for c in REQUIRED if c not in df.columns]
if missing: raise SystemExit("Missing required columns: "+", ".join(missing))
df=df[REQUIRED].dropna()
if len(df)<100: raise SystemExit("Need at least 100 labelled rows for this prototype training run; more diverse, spatially independent data is strongly recommended.")
if set(df.label.unique())!={0,1}: raise SystemExit("label must contain both classes: 0 and 1.")
X=df[FEATURES]; y=df["label"].astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=400,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1)
model.fit(X_train,y_train)
pred=model.predict(X_test)
metrics={"rows":int(len(df)),"balanced_accuracy":float(balanced_accuracy_score(y_test,pred)),
         "confusion_matrix":confusion_matrix(y_test,pred).tolist(),
         "classification_report":classification_report(y_test,pred,output_dict=True,zero_division=0),
         "features":FEATURES,
         "validation_note":"Random stratified holdout only. Not a spatially independent validation; do not present as operationally validated."}
OUT.mkdir(exist_ok=True)
joblib.dump(model,OUT/"landguard_random_forest.joblib")
(OUT/"metrics.json").write_text(json.dumps(metrics,indent=2))
print("Model saved. Holdout balanced accuracy:",round(metrics["balanced_accuracy"],3))
print("WARNING: use spatial/temporal validation and expert review before interpreting outputs.")
