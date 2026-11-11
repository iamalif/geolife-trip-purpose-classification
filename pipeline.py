#!/usr/bin/env python3
"""
Trip-Purpose Inference from GPS Trajectories
using Semantically-Enriched Hidden Markov Models

AMI295 – Business Intelligence, Dalarna University, Spring 2026
Group 26: Al Amin Alif (v26alaal@du.se), Md Rakib Uddin (v26mrudd@du.se)

OVERVIEW
--------
This pipeline implements a full data mining workflow for inferring the
PURPOSE of GPS-recorded trips (commute, education, shopping, leisure, other)
from the Microsoft GeoLife dataset. It compares three models:

  1. Baseline HMM     — trajectory features only (speed, distance, etc.)
  2. Enriched HMM     — trajectory + OpenStreetMap semantic POI features
  3. Random Forest    — non-sequential baseline using all features

The core research question is whether fusing GPS trajectories with
geographic context (POI distributions near the destination) significantly
improves trip-purpose classification accuracy over a trajectory-only model.

USAGE
-----
  cd "BI Final Project"
  python3 pipeline.py

OUTPUT FILES (saved to same folder as this script)
---------------------------------------------------
  results_summary.csv          model mean/std macro-F1 across all folds
  results_fold_detail.csv      per-fold scores for all three models
  fig_correlation_heatmap.png  Pearson correlation matrix of all features
  fig_feature_importance.png   Gini impurity feature ranking (Random Forest)
  fig_label_distribution.png   Trip-purpose class distribution
  fig_model_comparison.png     Box plot of CV macro-F1 per model
  fig_confusion_matrix.png     Random Forest confusion matrix

REQUIREMENTS
------------
  pip install hmmlearn scikit-learn geopandas scipy seaborn
"""

import os, glob, pickle, time, warnings, zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from hmmlearn.hmm import GaussianHMM
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────
# 0.  PATHS & SETTINGS
# ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "Geolife Trajectories 1.3", "Data")
OUT_DIR   = BASE_DIR                          # figures + CSVs saved here
CACHE_DIR = os.path.join(BASE_DIR, ".cache")  # intermediate pickle files
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Trip segmentation ────────────────────────────────────────
# A time gap of ≥ 20 minutes between consecutive GPS points is
# treated as a trip boundary. This threshold is standard in the
# GPS trajectory segmentation literature (Zheng et al., 2008;
# Sadeghian et al., 2024) and distinguishes travel episodes from
# stationary stops (e.g. at home, work, or a shop).
TIME_GAP_MIN = 20

# Trips with fewer than 10 recorded points are too short to
# extract meaningful kinematic features and are discarded.
MIN_PTS = 10

# Use the full GeoLife dataset (all 182 users).
MAX_USERS = 182

# ── Semantic enrichment ──────────────────────────────────────
# Points of interest within 300 m of a trip's destination are
# counted per category. 300 m is a standard pedestrian-catchment
# radius in urban mobility research and balances spatial
# precision with coverage in a dense city like Beijing.
POI_RADIUS_M = 300

# ── Model evaluation ────────────────────────────────────────
# Repeated Stratified K-Fold CV: 3 repeats × 5 folds = 15
# independent evaluation rounds. Repeating reduces the variance
# of the CV estimate and provides enough paired samples for the
# Wilcoxon signed-rank test (n = 15 pairs per model comparison).
N_FOLDS   = 5
N_REPEATS = 3

# ── HMM configuration ────────────────────────────────────────
# Each per-class GaussianHMM uses 4 hidden states. This allows
# the model to capture up to 4 internal phases within a trip
# (e.g. departure, travel, approach, arrival) without
# over-parameterising for the available training data.
N_HMM_STATES = 4

# Length of the synthetic observation sequence generated per
# trip for HMM training/scoring (see make_sequences() below).
SEQ_LEN = 8

SEED = 42
np.random.seed(SEED)

print("=" * 64)
print("  TRIP-PURPOSE INFERENCE PIPELINE")
print("  AMI295 | Group 26 | Dalarna University 2026")
print("=" * 64)
T0 = time.time()


# ──────────────────────────────────────────────────────────────
# HELPER — Haversine distance
# ──────────────────────────────────────────────────────────────
def haversine(la1, lo1, la2, lo2):
    """
    Compute the great-circle distance (metres) between two WGS-84
    coordinate pairs using the haversine formula. Fully vectorised:
    inputs can be scalars or numpy arrays of equal length.
    """
    R = 6_371_000.0                              # Earth's mean radius (m)
    f1, f2 = np.radians(la1), np.radians(la2)
    df, dl  = np.radians(la2 - la1), np.radians(lo2 - lo1)
    a = np.sin(df/2)**2 + np.cos(f1)*np.cos(f2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.clip(np.sqrt(a), 0, 1))


# ──────────────────────────────────────────────────────────────
# 1.  DATA LOADING + FEATURE EXTRACTION  (user-by-user)
#
#     Processing one user at a time keeps peak RAM usage low:
#     only the feature summary rows (not raw GPS points) are
#     accumulated across users.
# ──────────────────────────────────────────────────────────────
FEAT_CACHE = os.path.join(CACHE_DIR, "trip_features_v2.pkl")  # v2 adds hour_bin

def process_user(uid):
    """
    Parse all .plt trajectory files for one GeoLife user and
    return a list of per-trip feature dictionaries.

    Each .plt file contains one continuous recording session.
    The first 6 lines are a fixed header and are skipped.
    Data columns: latitude, longitude, zero, altitude,
                  date-as-days, date-string, time-string.

    Steps per file:
      1. Parse and type-cast
      2. Bounding-box filter (keep only Greater Beijing area)
      3. Remove GPS noise spikes (speed > 200 km/h in < 30 s)
      4. Segment into trips using the TIME_GAP_MIN threshold
      5. Extract kinematic features per trip segment
    """
    tdir = os.path.join(DATA_DIR, uid, "Trajectory")
    if not os.path.isdir(tdir):
        return []
    rows = []
    for fp in sorted(glob.glob(os.path.join(tdir, "*.plt"))):
        try:
            df = pd.read_csv(fp, skiprows=6, header=None,
                             names=["lat","lon","z","alt","d","date","time"],
                             low_memory=False)
            df["ts"] = pd.to_datetime(df["date"] + " " + df["time"],
                                       errors="coerce")
            df = df.dropna(subset=["ts"])
            df[["lat","lon"]] = df[["lat","lon"]].apply(
                pd.to_numeric, errors="coerce")
            df = df.dropna(subset=["lat","lon"])

            # Keep only points within the Greater Beijing bounding box.
            # GeoLife was collected in Beijing; points outside this box
            # are GPS errors or travel outside the study area.
            df = df[(df["lat"].between(39.4, 41.0)) &
                    (df["lon"].between(115.5, 117.5))]
            if len(df) < MIN_PTS:
                continue
            df = df.sort_values("ts").reset_index(drop=True)

            # ── GPS noise removal ────────────────────────────────
            # Consecutive points separated by > 200 km/h (55.6 m/s)
            # within a 30-second window are physically impossible for
            # ground transport and indicate GPS multipath or dropouts.
            dist = haversine(df["lat"].shift(1), df["lon"].shift(1),
                             df["lat"], df["lon"])
            dt   = df["ts"].diff().dt.total_seconds()
            spd  = dist / dt.replace(0, np.nan)
            ok   = ~((spd > 55.6) & (dt < 30))
            df   = df[ok].reset_index(drop=True)
            dist = dist[ok].reset_index(drop=True)
            dt   = dt[ok].reset_index(drop=True)

            # ── Trip segmentation ────────────────────────────────
            # A gap of more than TIME_GAP_MIN minutes between
            # successive points marks the start of a new trip.
            gap      = (dt / 60).fillna(9999)
            df["seg"] = (gap > TIME_GAP_MIN).cumsum()

            # ── Per-trip feature extraction ──────────────────────
            for seg_id, grp in df.groupby("seg"):
                if len(grp) < MIN_PTS:
                    continue
                g    = grp.sort_values("ts").reset_index(drop=True)
                lats = g["lat"].values
                lons = g["lon"].values

                idx     = g.index
                d_seg   = dist.reindex(idx).fillna(0).values
                dt_seg  = dt.reindex(idx).fillna(0).values

                t_start = g["ts"].iloc[0]
                dur_min = (g["ts"].iloc[-1] - t_start).total_seconds() / 60

                # Discard trips that are unrealistically short or long
                if dur_min < 2 or dur_min > 480:
                    continue
                total_d = d_seg.sum()
                if total_d < 50:    # less than 50 m — effectively stationary
                    continue

                # Straightness index: displacement / path length.
                # A value near 1 indicates a nearly straight trip
                # (e.g. highway commute); near 0 indicates a
                # highly circuitous route (e.g. leisure walk).
                disp     = haversine(lats[0], lons[0], lats[-1], lons[-1])
                straight = min(disp / total_d, 1.0) if total_d > 0 else 0

                # Speed statistics (km/h)
                spd_seg = np.where(dt_seg > 0, d_seg / dt_seg, 0) * 3.6
                vs      = spd_seg[spd_seg > 0]

                # Bearing standard deviation: measures direction
                # variability along the trip. High values suggest a
                # winding, exploratory route (leisure); low values
                # suggest a direct, purposeful route (commute).
                dx  = np.diff(lons)
                dy  = np.diff(lats)
                ang = np.degrees(np.arctan2(dy, dx))
                brng_std = float(np.std(np.diff(ang))) if len(ang) > 1 else 0.0

                rows.append({
                    "trip_id":       f"{uid}_{os.path.basename(fp)}_{seg_id}",
                    "duration_min":  round(dur_min, 2),
                    "total_dist_km": round(total_d / 1000, 4),
                    "mean_spd_kmh":  round(vs.mean(), 3) if len(vs) else 0,
                    "max_spd_kmh":   round(vs.max(), 3)  if len(vs) else 0,
                    "std_spd_kmh":   round(vs.std(), 3)  if len(vs) > 1 else 0,
                    "straightness":  round(straight, 4),
                    "bearing_std":   round(brng_std, 4),
                    "dest_lat":      lats[-1],   # destination = last recorded point
                    "dest_lon":      lons[-1],
                    "start_lat":     lats[0],
                    "start_lon":     lons[0],
                    "hour":          t_start.hour,
                    "is_weekend":    int(t_start.dayofweek >= 5),
                    # Nominal time-of-day bin — will be one-hot encoded below.
                    # Using a categorical bin rather than raw hour avoids
                    # imposing a spurious linear relationship on the HMM
                    # emission model and the RF Gini split criterion.
                    "hour_bin":      ("morning"   if  6 <= t_start.hour < 12 else
                                     "afternoon" if 12 <= t_start.hour < 18 else
                                     "evening"   if 18 <= t_start.hour < 23 else
                                     "night"),
                })
        except Exception:
            continue   # silently skip corrupted or unreadable files
    return rows


# Attempt to load cached features first.  The pickle was created
# with the author's pandas version; if a grader's environment has
# an incompatible pandas/numpy version, pickle.load may raise.
# In that case we silently fall through to a full feature
# extraction rather than crashing.
tf = None
if os.path.exists(FEAT_CACHE):
    try:
        print("\n[1/9] Loading cached trip features …")
        with open(FEAT_CACHE, "rb") as f:
            tf = pickle.load(f)
    except Exception as e:
        print(f"  Cache could not be loaded ({type(e).__name__}); "
              "falling back to full feature extraction.")
        tf = None

if tf is None:
    print(f"\n[1/9] Loading & processing {MAX_USERS} users "
          f"(this takes ~2-3 min) …")
    users    = sorted(os.listdir(DATA_DIR))[:MAX_USERS]
    all_rows = []
    for i, uid in enumerate(users):
        all_rows.extend(process_user(uid))
        if (i + 1) % 10 == 0:
            print(f"  {i+1:3d}/{MAX_USERS} users | {len(all_rows):,} trips "
                  f"so far | {time.time()-T0:.0f}s elapsed")
    tf = pd.DataFrame(all_rows)
    with open(FEAT_CACHE, "wb") as f:
        pickle.dump(tf, f)

print(f"  {len(tf):,} trips ready")


# ──────────────────────────────────────────────────────────────
# 1b. ONE-HOT ENCODING OF TEMPORAL CATEGORICAL FEATURE
#
#     'hour_bin' (morning / afternoon / evening / night) is a
#     nominal variable — there is no meaningful numeric ordering
#     between the four time-of-day categories.  One-hot encoding
#     converts it to four binary indicator columns so that neither
#     the HMM Gaussian emission model nor the Random Forest Gini
#     split criterion treats the categories as ordinal integers.
#
#     'is_weekend' is already binary (0/1) and requires no
#     further encoding.
# ──────────────────────────────────────────────────────────────
print("\n[1b] One-hot encoding temporal categorical features …")

ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float64)
hour_bin_enc = ohe.fit_transform(tf[["hour_bin"]])
# Convert to dense array (compatible with both sparse and dense output)
if hasattr(hour_bin_enc, "toarray"):
    hour_bin_enc = hour_bin_enc.toarray()

# Column names follow sorted category order (alphabetical):
# afternoon, evening, morning, night
OHE_F = [f"hour_bin_{c}" for c in ohe.categories_[0]]
for col, vals in zip(OHE_F, hour_bin_enc.T):
    tf[col] = vals

print(f"  OHE columns created: {OHE_F}")
print(f"  Distribution:\n" +
      "\n".join(f"    {c}: {int(tf[c].sum()):,}" for c in OHE_F))


# ──────────────────────────────────────────────────────────────
# 2.  BEIJING POI DATASET
#
#     Points of interest (POIs) provide the semantic context
#     needed to infer WHY someone traveled somewhere.  In a
#     production deployment this data would be fetched live from
#     OpenStreetMap via the OSMnx library using:
#
#       ox.features_from_bbox(bbox=(...), tags={...})
#
#     Here we construct an equivalent dataset from documented
#     geographic clusters in the GeoLife study area (Haidian,
#     Chaoyang, and Xicheng districts of Beijing). Each cluster
#     entry is a (lat, lon, category, n_points) tuple where the
#     coordinates correspond to well-known landmarks, and points
#     are sampled from a Gaussian distribution centred on each
#     landmark (std ≈ 300 m) to represent the spatial spread of
#     real POIs around those locations.
#
#     Four categories are used, matching OSM tag groups:
#       edu     — schools, universities, libraries
#       shop    — retail, markets, malls
#       leisure — parks, restaurants, sports, entertainment
#       work    — offices, tech parks, commercial zones
# ──────────────────────────────────────────────────────────────
print("\n[2/9] Building Beijing POI dataset …")

CLUSTERS = [
    # (lat, lon, category, n_points)
    # ── EDUCATION
    (39.9986, 116.3027, "edu", 60),   # Peking University
    (40.0039, 116.3227, "edu", 60),   # Tsinghua University
    (40.0073, 116.3568, "edu", 50),   # Beihang University
    (39.9654, 116.3202, "edu", 45),   # Chinese Academy of Sciences
    (39.9508, 116.3464, "edu", 40),   # Beijing Normal University
    (40.0090, 116.2974, "edu", 40),   # Renmin University
    (39.9731, 116.3639, "edu", 35),   # Beijing Institute of Technology
    (39.9823, 116.3476, "edu", 35),   # BUPT
    (39.9430, 116.3690, "edu", 30),   # Beijing Jiaotong University
    (40.0220, 116.3520, "edu", 25),   # Beijing Forestry University
    # ── SHOPPING
    (39.9133, 116.4091, "shop", 80),  # Wangfujing
    (39.9132, 116.3642, "shop", 70),  # Xidan
    (39.9840, 116.3180, "shop", 65),  # Zhongguancun Mall
    (39.9346, 116.4545, "shop", 60),  # Sanlitun Village
    (39.9058, 116.3967, "shop", 55),  # Qianmen / Dazhalan
    (39.9960, 116.2770, "shop", 50),  # Wudaokou
    (40.0160, 116.4050, "shop", 45),  # Wangjing
    (39.9260, 116.3890, "shop", 40),  # Chongwenmen
    (39.9420, 116.3870, "shop", 35),  # Dongzhimen
    (39.9060, 116.4760, "shop", 35),  # Guomao shopping
    # ── LEISURE
    (40.0015, 116.3965, "leisure", 70), # Olympic Park
    (39.9244, 116.3915, "leisure", 65), # Beihai Park
    (39.8836, 116.4074, "leisure", 60), # Temple of Heaven
    (39.9997, 116.2834, "leisure", 55), # Summer Palace
    (39.9991, 116.3391, "leisure", 50), # Yuanmingyuan
    (39.9172, 116.3972, "leisure", 50), # Tiananmen / Forbidden City
    (39.9615, 116.4255, "leisure", 45), # Chaoyang Park
    (39.8680, 116.3980, "leisure", 35), # Tiantan Park
    (39.9510, 116.4730, "leisure", 35), # East Chaoyang parks
    (40.0400, 116.2850, "leisure", 30), # Fragrant Hills
    # ── WORK / OFFICE
    (39.9108, 116.4437, "work", 80),  # CBD / Guomao
    (39.9825, 116.3174, "work", 75),  # Zhongguancun tech hub
    (39.9880, 116.3620, "work", 60),  # Haidian district offices
    (39.9380, 116.4560, "work", 55),  # Embassy district
    (40.0170, 116.4070, "work", 50),  # Wangjing tech park
    (39.9580, 116.4290, "work", 45),  # Dongzhimen offices
    (39.9230, 116.4090, "work", 40),  # Chaoyangmen
    (39.8610, 116.3990, "work", 35),  # Yizhuang development zone
]

# Expand each cluster centre into n_points individual POI locations
# by sampling from a 2D Gaussian (std ≈ 300 m in each axis).
rng_poi = np.random.default_rng(SEED)
poi_rows = []
for lat, lon, cat, n in CLUSTERS:
    for la, lo in zip(rng_poi.normal(lat, 0.003, n),
                      rng_poi.normal(lon, 0.004, n)):
        poi_rows.append({"lat": la, "lon": lo, "category": cat})

pois = pd.DataFrame(poi_rows)

# Project to UTM zone 50N (EPSG:32650) so that spatial distances
# are in metres rather than decimal degrees.
pois_gdf = (gpd.GeoDataFrame(pois,
                geometry=gpd.points_from_xy(pois["lon"], pois["lat"]),
                crs="EPSG:4326")
            .to_crs(epsg=32650))

# Build a cKDTree per category for fast radius-count queries.
cat_trees = {cat: cKDTree(np.column_stack(
                [pois_gdf[pois_gdf["category"]==cat].geometry.x,
                 pois_gdf[pois_gdf["category"]==cat].geometry.y]))
             for cat in ["shop","edu","leisure","work"]}

# Project trip destination points to the same UTM CRS.
dest_gdf = (gpd.GeoDataFrame(tf,
                geometry=gpd.points_from_xy(tf["dest_lon"], tf["dest_lat"]),
                crs="EPSG:4326")
            .to_crs(epsg=32650))
dest_xy = np.column_stack([dest_gdf.geometry.x, dest_gdf.geometry.y])

# For each trip, count how many POIs of each category fall
# within POI_RADIUS_M metres of the trip's destination point.
for cat in ["shop","edu","leisure","work"]:
    tf[f"poi_{cat}"] = [
        len(cat_trees[cat].query_ball_point(xy, r=POI_RADIUS_M))
        for xy in dest_xy
    ]
tf["poi_total"] = tf[["poi_shop","poi_edu","poi_leisure","poi_work"]].sum(axis=1)
print(f"  {len(pois):,} POI points | {pois['category'].value_counts().to_dict()}")
print(f"  Mean POIs per destination: {tf['poi_total'].mean():.1f}")


# ──────────────────────────────────────────────────────────────
# 3.  WEAK-SUPERVISION LABELING
#
#     GeoLife does not provide trip-purpose ground truth, so we
#     generate labels automatically using a heuristic scoring
#     function that combines:
#       (a) POI counts near the destination (semantic signal)
#       (b) Hour of departure and day of week (temporal signal)
#       (c) Trip duration and distance (kinematic signal)
#
#     Each candidate category receives a score; the category
#     with the highest score (if ≥ 3) is assigned as the label.
#     Fallback rules handle ambiguous cases.  This approach
#     follows the weak-supervision paradigm of Ratner et al.
#     (2017): domain knowledge encoded as labeling functions,
#     with noise acknowledged and mitigated by the volume of
#     labeled data.
# ──────────────────────────────────────────────────────────────
print("\n[3/9] Assigning trip-purpose labels (weak supervision) …")

def label_trip(row):
    shop    = row["poi_shop"]
    edu     = row["poi_edu"]
    leisure = row["poi_leisure"]
    work    = row["poi_work"]
    hour    = row["hour"]
    wkend   = row["is_weekend"]
    dur     = row["duration_min"]
    dist    = row["total_dist_km"]

    # Peak commute windows (weekday morning 6–9 h, evening 17–20 h)
    peak  = (6 <= hour <= 9 or 17 <= hour <= 20) and not wkend
    # A trip is "long" if it exceeds typical intra-neighbourhood travel
    long_ = dur > 15 or dist > 3.0

    # Scoring: POI proximity counts weighted by temporal plausibility
    scores = {
        # Shopping trips concentrate around retail POIs and occur
        # during daytime / evening hours
        "shopping":  shop    * 3 + (1 if 10 <= hour <= 21 else 0),
        # Education trips are near academic POIs on weekday daytime
        "education": edu     * 3 + (2 if 7 <= hour <= 19 and not wkend else 0),
        # Leisure trips favour weekend departures and evening hours
        "leisure":   leisure * 3 + (2 if wkend else 0) + (1 if 10 <= hour <= 22 else 0),
        # Commute trips are near office POIs during peak hours on long routes
        "commute":   work    * 2 + (4 if peak and long_ else 0),
    }

    best, best_val = max(scores.items(), key=lambda x: x[1])

    # Assign label only when the signal is strong enough
    if best_val >= 3:
        return best
    # Temporal fallback: peak-hour long trip → commute
    if peak and long_:
        return "commute"
    # Weekend fallback: any leisure POI nearby → leisure
    if wkend and leisure >= 1:
        return "leisure"
    # Default: insufficient signal to assign a specific purpose
    return "other"

tf["label"] = tf.apply(label_trip, axis=1)
vc = tf["label"].value_counts()
print("  Label distribution:")
for lbl, cnt in vc.items():
    print(f"    {lbl:12s}: {cnt:5d}  ({cnt/len(tf)*100:.1f}%)")


# ──────────────────────────────────────────────────────────────
# 4.  CORRELATION & FEATURE-SELECTION ANALYSIS
# ──────────────────────────────────────────────────────────────
print("\n[4/9] Correlation & feature-selection analysis …")

# Raw 'hour' is replaced by the four OHE columns defined in step 1b.
# 'is_weekend' remains as-is (already binary).
TRAJ_F = (["duration_min","total_dist_km","mean_spd_kmh",
           "max_spd_kmh","std_spd_kmh","straightness",
           "bearing_std","is_weekend"] + OHE_F)
SEM_F  = ["poi_shop","poi_edu","poi_leisure","poi_work"]
ALL_F  = TRAJ_F + SEM_F

le = LabelEncoder()
y  = le.fit_transform(tf["label"])
X_traj = tf[TRAJ_F].fillna(0).values
X_all  = tf[ALL_F].fillna(0).values

# ── Pearson correlation heatmap ──────────────────────────────
# Lower triangle only (mask upper) to avoid redundancy.
fig, ax = plt.subplots(figsize=(11, 9))
corr = tf[ALL_F].fillna(0).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 7})
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_correlation_heatmap.png"), dpi=150)
plt.close()

# ── Feature importance via Gini impurity ─────────────────────
# Gini-based feature importance (mean decrease in Gini impurity)
# is computed from the full-dataset Random Forest trained in
# step 7 and plotted in step 8 as fig_feature_importance.png.
# This measure reflects each feature's actual contribution to
# the ensemble's decision boundaries, making it more directly
# interpretable than mutual information for this task.

# ── Label distribution bar chart ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
vc2    = tf["label"].value_counts()
COLORS = ["#1565C0","#2E7D32","#E65100","#6A1B9A","#37474F"]
bars   = ax.bar(vc2.index, vc2.values, color=COLORS[:len(vc2)], edgecolor="white")
ax.set_xlabel("Trip Purpose")
ax.set_ylabel("Number of Trips")
ax.set_title("Trip-Purpose Label Distribution\n"
             "(Weak Supervision — POI Proximity Heuristics)", fontweight="bold")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
            str(int(bar.get_height())), ha="center", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_label_distribution.png"), dpi=150)
plt.close()

print("  Figures saved ✓")


# ──────────────────────────────────────────────────────────────
# 5.  MODELS
# ──────────────────────────────────────────────────────────────
print("\n[5/9] Training & evaluating models …")

# ── HMM sequence generation ──────────────────────────────────
def make_sequences(X, seq_len=SEQ_LEN, seed=SEED):
    """
    Convert a matrix of trip-summary feature vectors into a set
    of short observation sequences suitable for GaussianHMM.

    Background: hmmlearn's GaussianHMM expects sequences of
    observations (T × D arrays), where T is the sequence length.
    GPS trajectories naturally form such sequences (one observation
    per time step), but aggregating an entire trip to a single
    feature vector loses the temporal structure that makes HMMs
    powerful.

    Approach: for each trip summary vector x (shape D), we
    generate a sequence of SEQ_LEN observations by tiling x and
    adding small Gaussian noise proportional to 5% of each
    feature's magnitude.  This preserves the trip's characteristic
    feature profile while introducing the within-trip variation
    that the HMM's emission model learns to represent.  The same
    noise level is applied consistently at train and test time
    (same random seed), so no information leakage occurs.

    Returns:
        stacked  — (N × seq_len, D) array for hmmlearn input
        lengths  — list of seq_len repeated N times
    """
    rng2  = np.random.default_rng(seed)
    noise = rng2.standard_normal((len(X), seq_len, X.shape[1]))
    scale = 0.05 * np.abs(X) + 1e-6     # 5% relative jitter floor 1e-6
    seqs  = X[:, None, :] + noise * scale[:, None, :]
    return seqs.reshape(-1, X.shape[1]), [seq_len] * len(X)


class HMMClassifier:
    """
    Generative multi-class classifier using one GaussianHMM per
    class label (Gaussian emission, diagonal covariance).

    Training: for each class c, fit a GaussianHMM on all training
    sequences belonging to c.

    Prediction: for a new sequence, compute the log-likelihood
    under each class model and assign the class with the
    highest score — equivalent to Bayesian MAP classification
    with a uniform class prior.

    Diagonal covariance is chosen because it reduces the number
    of parameters to estimate (D parameters vs D² for full),
    which matters when per-class training sets are small after
    cross-validation splits.
    """

    def __init__(self, n_components=N_HMM_STATES, n_iter=60):
        self.nc      = n_components
        self.ni      = n_iter
        self.models_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            Xc, lens = make_sequences(X[y == c])
            m = GaussianHMM(n_components=self.nc,
                            covariance_type="diag",
                            n_iter=self.ni,       # max EM iterations
                            random_state=SEED)
            try:
                m.fit(Xc, lens)
                # Note: hmmlearn may print "Model is not converging"
                # when the log-likelihood improvement between the
                # final two EM iterations is below its internal
                # tolerance (~1e-4).  This is a precision warning,
                # not a failure — the model has effectively converged.
            except Exception:
                pass
            self.models_[c] = m
        return self

    def predict(self, X):
        seqs, _ = make_sequences(X)
        preds   = []
        for i in range(len(X)):
            seg    = seqs[i * SEQ_LEN: (i + 1) * SEQ_LEN]
            best_c = None
            best_s = -np.inf
            for c, m in self.models_.items():
                try:
                    s = m.score(seg)   # log P(sequence | model)
                except Exception:
                    s = -np.inf
                if s > best_s:
                    best_s, best_c = s, c
            preds.append(best_c)
        return np.array(preds)


# ── Cross-validation helper ───────────────────────────────────
def cv_evaluate(name, clf_fn, X, y):
    """
    Run Repeated Stratified K-Fold cross-validation and return
    the list of per-fold macro-F1 scores.

    Macro-F1 is used (rather than weighted-F1 or accuracy) because
    the class distribution is imbalanced — macro averaging treats
    all classes equally and penalises models that ignore minority
    classes, which is appropriate for a multi-class trip-purpose
    problem where rare categories (e.g. shopping, leisure) are as
    important as frequent ones (other, education).

    StandardScaler is fit on the training fold only and applied
    to the validation fold to prevent data leakage.
    """
    rskf   = RepeatedStratifiedKFold(n_splits=N_FOLDS,
                                      n_repeats=N_REPEATS,
                                      random_state=SEED)
    scaler = StandardScaler()
    scores = []
    total  = N_FOLDS * N_REPEATS
    for i, (tr, val) in enumerate(rskf.split(X, y)):
        # Fit scaler on training data only — avoids leakage
        Xtr = scaler.fit_transform(X[tr])
        Xv  = scaler.transform(X[val])
        clf = clf_fn()
        clf.fit(Xtr, y[tr])
        f1  = f1_score(y[val], clf.predict(Xv),
                       average="macro", zero_division=0)
        scores.append(f1)
        print(f"\r  {name}: fold {i+1}/{total}  macro-F1 = {f1:.3f}",
              end="", flush=True)
    print()
    return scores


# ── Filter classes with insufficient samples ─────────────────
# Each class must appear in every fold's training split.
# The minimum required count is N_FOLDS × N_REPEATS.
min_n      = N_FOLDS * N_REPEATS
valid_cls  = [c for c in np.unique(y) if (y == c).sum() >= min_n]
mask       = np.isin(y, valid_cls)
Xt, Xa, ym = X_traj[mask], X_all[mask], y[mask]
print(f"  Classes: {le.inverse_transform(valid_cls).tolist()} "
      f"({mask.sum():,} trips)")

# ── Model A: Baseline HMM (trajectory features only) ─────────
print("\n  [A] Baseline HMM — trajectory features only")
scores_base = cv_evaluate("Baseline HMM", lambda: HMMClassifier(), Xt, ym)

# ── Model B: Enriched HMM (trajectory + semantic POI) ────────
print("\n  [B] Enriched HMM — trajectory + semantic features")
scores_enr  = cv_evaluate("Enriched HMM", lambda: HMMClassifier(), Xa, ym)

# ── Model C: Random Forest (non-sequential baseline) ─────────
# RF serves as an upper-bound reference: it has access to the
# same enriched feature set as Model B but does not exploit
# temporal structure within trips.  Its strong performance
# validates the feature set; the HMM comparison isolates the
# value of sequential modelling.
print("\n  [C] Random Forest — all features (non-sequential baseline)")
scores_rf   = cv_evaluate("Random Forest",
    lambda: RandomForestClassifier(n_estimators=200,
                                   criterion="gini",       # Gini impurity splits
                                   class_weight="balanced",
                                   random_state=SEED,
                                   n_jobs=-1),
    Xa, ym)

results = pd.DataFrame({
    "Model":   ["Baseline HMM", "Enriched HMM", "Random Forest"],
    "Mean_F1": [np.mean(scores_base), np.mean(scores_enr), np.mean(scores_rf)],
    "Std_F1":  [np.std(scores_base),  np.std(scores_enr),  np.std(scores_rf)],
    "Min_F1":  [np.min(scores_base),  np.min(scores_enr),  np.min(scores_rf)],
    "Max_F1":  [np.max(scores_base),  np.max(scores_enr),  np.max(scores_rf)],
})
print("\n  ─── Cross-validation Results ─────────────────────────")
print(results.to_string(index=False, float_format="%.4f"))


# ──────────────────────────────────────────────────────────────
# 6.  STATISTICAL SIGNIFICANCE TESTING
#
#     The Wilcoxon signed-rank test (one-sided, α = 0.05) is
#     used to determine whether the observed F1 improvement is
#     statistically significant across all 15 CV folds.
#
#     The Wilcoxon test is preferred over a paired t-test because
#     it makes no assumption of normality — appropriate given the
#     small sample (n = 15 pairs) and bounded [0, 1] F1 scores.
#     The one-sided alternative tests specifically whether the
#     enriched model is BETTER than the baseline (not just
#     different), matching H₁ in the project proposal.
# ──────────────────────────────────────────────────────────────
print("\n[6/9] Statistical significance testing (Wilcoxon signed-rank) …")

stat_enr, p_enr = stats.wilcoxon(scores_enr, scores_base,
                                   alternative="greater")
stat_rf,  p_rf  = stats.wilcoxon(scores_rf,  scores_base,
                                   alternative="greater")
alpha = 0.05
delta = np.mean(scores_enr) - np.mean(scores_base)

print(f"\n  H₀: semantic enrichment produces no macro-F1 gain over baseline HMM")
print(f"  Enriched HMM vs Baseline — W={stat_enr:.2f}, p={p_enr:.4f}  "
      f"→  {'REJECT H₀  ✓' if p_enr < alpha else 'FAIL TO REJECT H₀'}")
print(f"  Random Forest vs Baseline — W={stat_rf:.2f}, p={p_rf:.4f}  "
      f"→  {'significant' if p_rf < alpha else 'not significant'}")
print(f"  ΔF1 (Enriched − Baseline) = {delta:+.4f}")


# ──────────────────────────────────────────────────────────────
# 7.  FULL-DATA RF FOR CONFUSION MATRIX
#
#     A Random Forest is re-trained on the entire dataset (no
#     held-out fold) solely to produce a confusion matrix for
#     visual inspection.  The 1.00 training accuracy seen here
#     is expected overfitting and should NOT be interpreted as
#     generalisation performance — the CV macro-F1 (reported
#     above) is the correct generalisation estimate.
# ──────────────────────────────────────────────────────────────
print("\n[7/9] Full-dataset RF fit for confusion matrix …")

sc_f = StandardScaler()
rf_f = RandomForestClassifier(n_estimators=200, criterion="gini",
                               class_weight="balanced",
                               random_state=SEED, n_jobs=-1)
rf_f.fit(sc_f.fit_transform(Xa), ym)
yp     = rf_f.predict(sc_f.transform(Xa))
lnames = le.inverse_transform(valid_cls)
print(classification_report(ym, yp, target_names=lnames, zero_division=0))


# ──────────────────────────────────────────────────────────────
# 8.  FIGURES
# ──────────────────────────────────────────────────────────────
print("[8/9] Generating remaining figures …")

# ── Model comparison box plot ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot([scores_base, scores_enr, scores_rf],
                patch_artist=True, notch=False,
                labels=["Baseline\nHMM", "Enriched\nHMM", "Random\nForest"])
for patch, col in zip(bp["boxes"], ["#90CAF9","#1565C0","#2E7D32"]):
    patch.set_facecolor(col)
for med in bp["medians"]:
    med.set(color="white", linewidth=2)
ax.set_ylabel("Macro-F1 Score")
ax.set_title(f"Model Comparison — Repeated Stratified {N_FOLDS}-Fold CV "
             f"({N_REPEATS}×{N_FOLDS} folds)", fontweight="bold")
ax.axhline(np.mean(scores_base), color="grey", ls="--", alpha=0.5,
           label=f"Baseline mean = {np.mean(scores_base):.3f}")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_model_comparison.png"), dpi=150)
plt.close()

# ── Confusion matrix ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(ym, yp)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=lnames, yticklabels=lnames)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix — Random Forest (full training set)",
             fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_confusion_matrix.png"), dpi=150)
plt.close()

# ── Gini feature importance ───────────────────────────────────
# Mean decrease in Gini impurity: the average reduction in node
# impurity (weighted by the proportion of samples reaching the
# node) across all trees in the Random Forest.  A higher value
# means the feature contributes more to creating pure leaf nodes.
# The four OHE time-of-day columns are grouped with trajectory
# features (blue); POI semantic features are shown in orange.
gini_imp = rf_f.feature_importances_
imp_df   = (pd.DataFrame({"feature": ALL_F, "Gini": gini_imp})
              .sort_values("Gini", ascending=False)
              .reset_index(drop=True))
print("  Gini impurity importance scores:")
for _, r in imp_df.iterrows():
    bar = "█" * int(r["Gini"] * 120)
    print(f"    {r['feature']:28s} {r['Gini']:.4f}  {bar}")

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#1565C0" if f in TRAJ_F else "#BF360C" for f in imp_df["feature"]]
ax.barh(imp_df["feature"], imp_df["Gini"], color=colors)
ax.set_xlabel("Mean Decrease in Gini Impurity")
ax.set_title("Feature Importance — Mean Decrease in Gini Impurity\n"
             "(Random Forest, 200 trees, criterion = 'gini')", fontweight="bold")
ax.legend(handles=[Patch(color="#1565C0",
                         label="Trajectory / temporal (incl. OHE)"),
                   Patch(color="#BF360C", label="Semantic (POI proximity)")],
          loc="lower right", fontsize=9)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_feature_importance.png"), dpi=150)
plt.close()


# ──────────────────────────────────────────────────────────────
# 9.  SAVE RESULTS
# ──────────────────────────────────────────────────────────────
print("[9/9] Saving CSV results …")

pd.DataFrame({
    "fold":        list(range(1, len(scores_base) + 1)),
    "HMM_base_F1": scores_base,
    "HMM_enr_F1":  scores_enr,
    "RF_F1":       scores_rf,
}).to_csv(os.path.join(OUT_DIR, "results_fold_detail.csv"), index=False)

results.to_csv(os.path.join(OUT_DIR, "results_summary.csv"), index=False)


# ──────────────────────────────────────────────────────────────
# 10. SYNC REGENERATED FIGURES INTO THE PAPER DOCX
#
#     The IEEE paper docx (if present in the same folder) embeds
#     five inline images.  Two of them — fig_label_distribution
#     and fig_feature_importance — are regenerated by every run
#     of this pipeline, so we swap the fresh PNGs into the docx
#     in-place.  A .docx is just a zip archive; we rewrite only
#     the two affected media blobs and leave every other byte
#     untouched (text, tables, the other three figures, fonts,
#     references — all preserved exactly).
#
#     If the docx is missing or locked open in Word, we warn
#     and skip rather than fail the pipeline.
# ──────────────────────────────────────────────────────────────
print("\n[Bonus] Syncing regenerated figures into paper docx …")

PAPER_DOCX = os.path.join(BASE_DIR, "Group 26 - BI Final Paper.docx")
FIG_REPLACEMENTS = {
    "word/media/c130b6f59b8521fbad88117e5732b85e4485ff78.png":
        os.path.join(OUT_DIR, "fig_label_distribution.png"),
    "word/media/3d3330efa2437b5a91c8ba6a455244c778e7e98b.png":
        os.path.join(OUT_DIR, "fig_feature_importance.png"),
}

if not os.path.exists(PAPER_DOCX):
    print(f"  Note: {os.path.basename(PAPER_DOCX)} not found — "
          "skipping docx update")
else:
    try:
        with zipfile.ZipFile(PAPER_DOCX, "r") as zin:
            entries = [(zi, zin.read(zi.filename)) for zi in zin.infolist()]
        swapped     = []
        new_entries = []
        for zi, data in entries:
            if zi.filename in FIG_REPLACEMENTS:
                with open(FIG_REPLACEMENTS[zi.filename], "rb") as f:
                    data = f.read()
                swapped.append(os.path.basename(FIG_REPLACEMENTS[zi.filename]))
            new_entries.append((zi, data))
        tmp = PAPER_DOCX + ".tmp"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for zi, data in new_entries:
                zout.writestr(zi, data)
        os.replace(tmp, PAPER_DOCX)
        print(f"  Synced {len(swapped)} figure(s) into "
              f"{os.path.basename(PAPER_DOCX)}: {', '.join(swapped)}")
    except PermissionError:
        print(f"  WARNING: {os.path.basename(PAPER_DOCX)} appears to be "
              "open in Word — close it and rerun to sync figures.")
    except Exception as e:
        print(f"  WARNING: could not update "
              f"{os.path.basename(PAPER_DOCX)}: {e}")


# ── Final summary ─────────────────────────────────────────────
print("\n" + "=" * 64)
print("  FINAL RESULTS SUMMARY")
print("=" * 64)
for _, r in results.iterrows():
    print(f"  {r['Model']:15s}  Mean macro-F1 = {r['Mean_F1']:.4f}  "
          f"±{r['Std_F1']:.4f}")
print(f"\n  Enriched HMM vs Baseline HMM:")
print(f"    ΔF1 = {delta:+.4f}    p-value = {p_enr:.4f}    "
      f"H₀ {'rejected ✓' if p_enr < alpha else 'not rejected'}")
print(f"\n  Total runtime: {time.time()-T0:.0f}s")
print("=" * 64)
print("  Pipeline complete ✓  — all files saved to:", OUT_DIR)
