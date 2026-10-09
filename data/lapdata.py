import matplotlib.pyplot as plt
import polars as pl

df = pl.read_parquet('software-data.parquet')
df = df.filter(
    (pl.col('VDM_GPS_Latitude') != 0) & (pl.col('VDM_GPS_Longitude') != 0)
).drop_nulls(['VDM_GPS_Latitude', 'VDM_GPS_Longitude'])

x = df['VDM_GPS_Longitude']
y = df['VDM_GPS_Latitude']

plt.figure(figsize=(8, 8))
plt.plot(x, y, linewidth=2)
plt.scatter(x[0], y[0], c='green', label='start', zorder=3)
plt.scatter(x[-1], y[-1], c='red', label='end', zorder=3)
plt.axis('equal')
plt.legend()
plt.savefig("laps.png", dpi=300)