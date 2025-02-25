import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import streamlit as st
import plotly.express as px

##############################################
# 1. DATA LOADING, CLEANING & FEATURE ENGINEERING
##############################################

file_path = "AccountLevelFinal.csv"
df = pd.read_csv(file_path)

# 1A. Remove STM members if present
if "STM" in df.columns:
    df = df[df["STM"] != 1]

# 1B. Fill missing numeric values with column means
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# 1C. Winsorization function
def winsorize(series, quantile=0.99):
    cap = series.quantile(quantile)
    return np.where(series > cap, cap, series)

# Cap extreme values
df["AvgSpend_capped"] = winsorize(df["AvgSpend"])
df["GamesAttended_capped"] = winsorize(df["GamesAttended"])

# 1D. Min-Max scaling
scaler = MinMaxScaler()
df["AvgSpend_norm"] = scaler.fit_transform(df[["AvgSpend_capped"]])
df["GamesAttended_norm"] = scaler.fit_transform(df[["GamesAttended_capped"]])

# 1E. BasketballPropensity (0–1000 → 0–1)
df["BasketballPropensity_norm"] = df["BasketballPropensity"] / 1000

# 1F. SocialMediaEngagement
if "SocialMediaEngagement" in df.columns:
    sm_map = {"low": 0.3, "medium": 0.6, "high": 1.0}
    df["SocialMediaEngagement_norm"] = df["SocialMediaEngagement"].map(sm_map).fillna(0.3)
else:
    df["SocialMediaEngagement_norm"] = 0.3

# 1G. DistanceToArena
df["DistanceToArena_capped"] = winsorize(df["DistanceToArena"])
df["DistanceClose_norm"] = scaler.fit_transform(df[["DistanceToArena_capped"]])
df["DistanceFar_norm"] = 1 - df["DistanceClose_norm"]

# 1H. Ensure fraction columns exist
fraction_cols = ["TierCD_Weekday_Fraction", "Tier_AB_Fraction", "WeekendFraction", "GiveawayFraction"]
for col in fraction_cols:
    if col not in df.columns:
        df[col] = 0.0

##############################################
# 2. ORIGINAL PLAN SCORES
##############################################

# 2A. Define bonus mappings for segments
value_plan_bonus_mapping   = {"D": 0.5}  # placeholder
marquee_plan_bonus_mapping = {"G": 1.0, "F": 0.75, "C": 0.4, "E": 0.2}
weekend_plan_bonus_mapping = {"B": 0.8, "E": 1.0, "G": 0.5, "A": 0.4}
promo_plan_bonus_mapping   = {"G": 1.0, "F": 0.8, "E": 0.4}

df["FanSegmentBonus_Value"]   = df["FanSegment"].map(value_plan_bonus_mapping).fillna(0)
df["FanSegmentBonus_Marquee"] = df["FanSegment"].map(marquee_plan_bonus_mapping).fillna(0)
df["FanSegmentBonus_Weekend"] = df["FanSegment"].map(weekend_plan_bonus_mapping).fillna(0)
df["FanSegmentBonus_Promo"]   = df["FanSegment"].map(promo_plan_bonus_mapping).fillna(0)

# 2B. Calculate plan scores
df["ValuePlan_Score"] = (
    0.25 * (1 - df["AvgSpend_norm"]) +
    0.25 * df["TierCD_Weekday_Fraction"] +
    0.15 * df["DistanceClose_norm"] +
    0.15 * (1 - df["GamesAttended_norm"]) +
    0.20 * df["FanSegmentBonus_Value"]
)

df["MarqueePlan_Score"] = (
    0.30 * df["AvgSpend_norm"] +
    0.30 * df["Tier_AB_Fraction"] +
    0.20 * df["BasketballPropensity_norm"] +
    0.20 * df["FanSegmentBonus_Marquee"]
)

df["WeekendPlan_Score"] = (
    0.60 * df["WeekendFraction"] +
    0.10 * df["DistanceFar_norm"] +
    0.10 * df["AvgSpend_norm"] +
    0.20 * df["FanSegmentBonus_Weekend"]
)

df["PromoPlan_Score"] = (
    0.20 * df["SocialMediaEngagement_norm"] +
    0.60 * df["GiveawayFraction"] +
    0.20 * df["FanSegmentBonus_Promo"]
)

df["OldTotalScore"] = (
    df["ValuePlan_Score"] +
    df["MarqueePlan_Score"] +
    df["WeekendPlan_Score"] +
    df["PromoPlan_Score"]
)

##############################################
# 3. POINTS-BASED "TOTALPOINTS" FOR FAN RANKING
##############################################
def calc_new_total_points(row):
    points = 0
    # GamesAttended => 100 pts each
    games = row["GamesAttended"]
    points += games * 100
    # Bonus if >20
    if games > 20:
        points += 2000
    # Spend => 10 pts per $
    spend = row["AvgSpend"]
    points += spend * 10
    # SocialMedia bonus
    sm_norm = row["SocialMediaEngagement_norm"]  # 0.3, 0.6, 1.0
    if sm_norm <= 0.4:
        points += 300
    elif sm_norm <= 0.7:
        points += 600
    else:
        points += 1000
    return points

df["TotalPoints"] = df.apply(calc_new_total_points, axis=1)

# Quantile-based rank
q = df["TotalPoints"].quantile([0.33, 0.66]).values
cut1, cut2 = q[0], q[1]
def assign_rank_by_points(points):
    if points <= cut1:
        return "Rookie"
    elif points <= cut2:
        return "All-Star"
    else:
        return "Champion"

df["FanRank"] = df["TotalPoints"].apply(assign_rank_by_points)

##############################################
# 4. CLUSTERING & CLUSTER-BASED WEIGHTING
##############################################
features = [
    "AvgSpend_norm", 
    "GamesAttended_norm",
    "BasketballPropensity_norm",
    "SocialMediaEngagement_norm", 
    "DistanceClose_norm", 
    "WeekendFraction"
]

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(df[features])

# (A) Inspect cluster means to decide how you'd like to weight each plan for that cluster
#     For demonstration, let's do something simple:
#       - If a cluster has high average spend => boost MarqueePlan_Score 10%
#       - If a cluster has high weekend fraction => boost WeekendPlan_Score 10%
#       - If a cluster has high giveaway fraction => boost PromoPlan_Score 10%
#       - If a cluster has high TierCD_Weekday_Fraction => boost ValuePlan_Score 10%
#     In reality, you'd refine these from domain logic or more advanced modeling.

cluster_summary = df.groupby("Cluster")[
    ["AvgSpend_norm", "WeekendFraction", "TierCD_Weekday_Fraction", "GiveawayFraction"]
].mean()

# We'll define a dictionary for each cluster's *multiplier* on each plan
# (1.0 = no change, 1.1 = +10%, etc.)
cluster_plan_multipliers = {
    0: {"Value": 1.0, "Marquee": 1.0, "Weekend": 1.0, "Promo": 1.0},
    1: {"Value": 1.0, "Marquee": 1.0, "Weekend": 1.0, "Promo": 1.0},
    2: {"Value": 1.0, "Marquee": 1.0, "Weekend": 1.0, "Promo": 1.0},
}

# Example logic: if cluster 0 has the highest TierCD_Weekday_Fraction,
# add a 10% bump to ValuePlan. If cluster 1 has the highest WeekendFraction,
# add a 10% bump to WeekendPlan, etc. Here’s a simple procedure:
overall_means = {
    "spend": df["AvgSpend_norm"].mean(),
    "weekend": df["WeekendFraction"].mean(),
    "weekdayCD": df["TierCD_Weekday_Fraction"].mean(),
    "giveaway": df["GiveawayFraction"].mean()
}

for c in cluster_summary.index:
    row = cluster_summary.loc[c]
    # Compare each feature to overall mean
    # If it's more than +0.05 above mean, we bump the relevant plan
    if row["TierCD_Weekday_Fraction"] > overall_means["weekdayCD"] + 0.05:
        cluster_plan_multipliers[c]["Value"] += 0.1  # +10% to ValuePlan
    if row["AvgSpend_norm"] > overall_means["spend"] + 0.05:
        cluster_plan_multipliers[c]["Marquee"] += 0.1
    if row["WeekendFraction"] > overall_means["weekend"] + 0.05:
        cluster_plan_multipliers[c]["Weekend"] += 0.1
    if row["GiveawayFraction"] > overall_means["giveaway"] + 0.05:
        cluster_plan_multipliers[c]["Promo"] += 0.1

# (B) Apply the multipliers to create adjusted plan scores
def adjust_scores(row):
    c = row["Cluster"]
    m = cluster_plan_multipliers[c]
    val_adj = row["ValuePlan_Score"] * m["Value"]
    marq_adj = row["MarqueePlan_Score"] * m["Marquee"]
    wend_adj = row["WeekendPlan_Score"] * m["Weekend"]
    promo_adj = row["PromoPlan_Score"] * m["Promo"]
    return pd.Series([val_adj, marq_adj, wend_adj, promo_adj])

df[["ValuePlan_Score_Adj", 
    "MarqueePlan_Score_Adj", 
    "WeekendPlan_Score_Adj", 
    "PromoPlan_Score_Adj"]] = df.apply(adjust_scores, axis=1)

# You can define a "Top Plan" after adjustments
def top_plan(row):
    scores = {
        "ValuePlan": row["ValuePlan_Score_Adj"],
        "MarqueePlan": row["MarqueePlan_Score_Adj"],
        "WeekendPlan": row["WeekendPlan_Score_Adj"],
        "PromoPlan": row["PromoPlan_Score_Adj"]
    }
    best_plan = max(scores, key=scores.get)
    return best_plan

df["MostLikelyPlan"] = df.apply(top_plan, axis=1)

##############################################
# 5. STREAMLIT DASHBOARD
##############################################

bucks_green = "#00471B"
bucks_cream = "#EEE1C6"

st.set_page_config(page_title="Milwaukee Bucks Fan Dashboard", layout="wide")
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {bucks_cream};
    }}
    .sidebar .sidebar-content {{
        background-color: {bucks_green};
        color: white;
    }}
    .stButton>button {{
        background-color: {bucks_green};
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Milwaukee Bucks Fan Dashboard - Cluster-Based Plan Weighting")

st.markdown("""
**What's New?**  
We now factor **cluster insights** into the four partial-plan scores:

1. **ValuePlan_Score**  
2. **MarqueePlan_Score**  
3. **WeekendPlan_Score**  
4. **PromoPlan_Score**

After the initial score is computed, we check which cluster a fan belongs to. If that cluster tends to have:
- higher TierCD_Weekday_Fraction => we boost ValuePlan_Score by +10%
- higher AvgSpend => we boost MarqueePlan_Score by +10%
- higher WeekendFraction => we boost WeekendPlan_Score by +10%
- higher GiveawayFraction => we boost PromoPlan_Score by +10%

This yields "Adjusted" plan scores, helping us identify which plan each cluster is *truly* most likely to buy.
We also keep the points-based **TotalPoints** to rank fans (Rookie / All-Star / Champion).
""")

# Sidebar
selected_rank = st.sidebar.selectbox("Select Fan Rank", options=["All", "Rookie", "All-Star", "Champion"])
selected_cluster = st.sidebar.selectbox("Select Cluster", options=["All", "0", "1", "2"])

filtered_df = df.copy()
if selected_rank != "All":
    filtered_df = filtered_df[filtered_df["FanRank"] == selected_rank]
if selected_cluster != "All":
    filtered_df = filtered_df[filtered_df["Cluster"] == int(selected_cluster)]

st.subheader("Filtered Fan Data")
st.dataframe(filtered_df[[
    "AccountNumber","FanSegment","FanRank","Cluster",
    "ValuePlan_Score","ValuePlan_Score_Adj",
    "MarqueePlan_Score","MarqueePlan_Score_Adj",
    "WeekendPlan_Score","WeekendPlan_Score_Adj",
    "PromoPlan_Score","PromoPlan_Score_Adj",
    "MostLikelyPlan","TotalPoints"
]])

# Distribution of total points
st.subheader("Distribution of TotalPoints")
fig_pts = px.histogram(
    filtered_df, x="TotalPoints", nbins=20,
    title="Histogram of TotalPoints",
    color_discrete_sequence=[bucks_green]
)
st.plotly_chart(fig_pts, use_container_width=True)

# Leaderboard by adjusted plan score
st.subheader("Top 10 Fans by Adjusted Plan Scores (Highest 'MostLikelyPlan' Value)")
# We'll pick a plan to sort by, or we can just show the top "MarqueePlan_Score_Adj" as example
# Instead, let's sort by whichever plan is 'MostLikelyPlan' for each row (this is trickier).
# For simplicity, let's sort by the biggest of the 4 adjusted scores:
filtered_df["MaxAdjScore"] = filtered_df[[
    "ValuePlan_Score_Adj","MarqueePlan_Score_Adj",
    "WeekendPlan_Score_Adj","PromoPlan_Score_Adj"
]].max(axis=1)
leaderboard = filtered_df.sort_values("MaxAdjScore", ascending=False).head(10)

st.table(leaderboard[[
    "AccountNumber",
    "FanSegment",
    "FanRank",
    "Cluster",
    "MostLikelyPlan",
    "MaxAdjScore",
    "TotalPoints"
]])

# Cluster Insights
st.subheader("Cluster Insights")
cluster_summary_full = df.groupby("Cluster")[features + [
    "ValuePlan_Score","MarqueePlan_Score","WeekendPlan_Score","PromoPlan_Score"
]].mean().round(2)

st.markdown("**Average of Key Features & Plan Scores per Cluster**")
st.dataframe(cluster_summary_full)

st.markdown("""
*Interpretation*: 
- Compare each cluster's "AvgSpend_norm," "WeekendFraction," etc., and see why 
  certain plan scores might get a boost. 
- Note how the final "adjusted" plan recommendations can shift 
  if we see high or low metrics in certain clusters.
""")
