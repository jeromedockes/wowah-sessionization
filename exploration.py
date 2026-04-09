# %%
import polars as pl
# %%
df = pl.read_parquet("data/wowah_data.parquet")
# %%
from skrub import TableReport

TableReport(df)
# %%
df.group_by("charclass").agg(pl.count()).sort("count", descending=True)
# %%
