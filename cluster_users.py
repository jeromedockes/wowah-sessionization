"""
Testing clustering and sessionization approaches on WoWAH data.
"""

# %%
import skrub
import numpy as np
import polars as pl

# %%
data_all = pl.read_parquet("data/wowah_data_raw.parquet")

data = data_all.filter(
    pl.col("char").is_in(
        data_all.select(pl.col("char").unique())
        .sample(fraction=0.1, seed=42)["char"]
        .implode()
    )
)

# %%
# Load the raw data
data_raw = skrub.var("raw_data").skb.subsample()
data_var = data_raw.with_columns(guild=pl.col("guild").replace({-1: None}))
# %%
from skrub import SessionEncoder

# Create a session encoder with a 30 minute timeout
encoder = SessionEncoder(group_by="char", timestamp_col="timestamp", session_gap=30)

data_with_sessions = data_var.skb.apply(encoder)
# %%
data_with_sessions
# %%
from skrub import deferred


@deferred
def add_session_features(df):
    return df.with_columns(
        pl.col("timestamp").first().over("timestamp_session_id").alias("session_start"),
        pl.col("timestamp").last().over("timestamp_session_id").alias("session_end"),
        pl.col("level")
        .max()
        .over("timestamp_session_id")
        .alias("max_level_in_session"),
        pl.col("level")
        .n_unique()
        .over("timestamp_session_id")
        .alias("levels_in_session"),
        pl.col("zone")
        .n_unique()
        .over("timestamp_session_id")
        .alias("zones_in_session"),
    )


@deferred
def add_char_features(df):
    return df.with_columns(
        pl.col("timestamp").first().over("char").alias("char_first_seen"),
        pl.col("timestamp").last().over("char").alias("char_last_seen"),
        pl.col("level").max().over("char").alias("max_level"),
        pl.col("zone").n_unique().over("char").alias("unique_zones_visited"),
        pl.col("timestamp_session_id").count().over("char").alias("sessions_per_char"),
        pl.col("timestamp_session_id")
        .count()
        .over("timestamp_session_id")
        .alias("session_duration"),
        pl.col("guild").n_unique().over("char").alias("guilds_joined"),
    )


@deferred
def add_aggregated_features(df):
    return df.with_columns(
        pl.col("level").mean().over("race").alias("avg_level_by_race"),
        pl.col("level").mean().over("charclass").alias("avg_level_by_class"),
        pl.col("level").mean().over("guild").alias("avg_level_by_guild"),
        pl.col("level").mean().over("zone").alias("avg_level_by_zone"),
        pl.col("charclass").count().over("charclass").alias("count_by_charclass"),
        pl.col("race").count().over("race").alias("count_by_race"),
        pl.col("char")
        .count()
        .over("race", "charclass")
        .alias("count_by_race_charclass"),
    )


# %%
data_with_all_features = (
    data_with_sessions.skb.apply_func(add_session_features)
    .skb.apply_func(add_char_features)
    .skb.apply_func(add_aggregated_features)
)
# %%
from skrub import TableVectorizer, DatetimeEncoder, ApplyToCols
import skrub.selectors as s

timestamp_encoder = ApplyToCols(
    DatetimeEncoder(periodic_encoding="circular"), keep_original=True, cols="timestamp"
)
data_vectorized = (
    data_with_all_features.skb.apply(
        TableVectorizer(), exclude_cols=["char", "timestamp"]
    )
    .skb.apply(timestamp_encoder)
    .skb.set_name("data_vectorized")
)
data_vectorized

# %%
# Aggregate by user, for the moment just taking the mean of all features


@deferred
def aggregate_by_user(df):
    return df.group_by("char").agg(pl.all().mean())


data_aggregated = (
    data_vectorized.skb.apply_func(aggregate_by_user)
    .skb.apply(skrub.SquashingScaler(), exclude_cols="char")
    .skb.set_name("data_aggregated")
)
data_aggregated

# %% 
from sklearn.impute import SimpleImputer
data_imputed = data_aggregated.skb.apply(SimpleImputer(strategy="mean")).skb.drop("char")

# %%
from sklearn.cluster import HDBSCAN, KMeans, Birch

# clusterer = KMeans(n_clusters=5)
clusterer = HDBSCAN(min_cluster_size=5)
# clusterer = Birch(threshold=0.5, n_clusters=5)

clusters = data_imputed.skb.apply(clusterer, unsupervised=True, y=None)
# %%
out = data_imputed.skb.apply(HDBSCAN(min_cluster_size=100), unsupervised=True, y=None).skb.set_name('hdbs')
labels =out.skb.make_learner().fit({"raw_data": data}).find_fitted_estimator('hdbs').labels_

# %%
# labels = clusters.skb.make_learner(fitted=True).fit_predict({"raw_data": data})
#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data_pca = data_imputed.skb.apply(pca).skb.eval({"raw_data": data})
# %%
import matplotlib.pyplot as plt
plt.scatter(data_pca["pca0"], data_pca["pca1"], c=labels)
# %%
plt.plot(labels)
# %%
