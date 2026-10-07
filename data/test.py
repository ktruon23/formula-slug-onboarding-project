import polars as pl
import matplotlib.pyplot as plt
import math

# 1. Read the CSV file into a DataFrame
df = pl.read_parquet('software-data.parquet')
df_speed = df.select('SME_TRQSPD_Speed')
print(df)
print(df_speed)

#function applies the velocity calculations given the wheel rpm, could also be adjusted to motor rpm if needed
def velocity_calculations(wheel_rpm):
    return (wheel_rpm*(2*(math.pi)/60)*.2)

df_ground_velocity = df.with_columns(
    pl.col("SME_TRQSPD_Speed")
    .map_elements(velocity_calculations, return_dtype=pl.Float64)
    .alias("ground_velocity")
)
y_values = df_ground_velocity["ground_velocity"]
x_values = range(1,len(y_values)+1)

plt.plot(x_values,y_values,linestyle="-")
plt.xlabel("Time")
plt.ylabel("Velocity (m/s)")

plt.savefig("ground_velocity.png", dpi=500)

#this function will determine whether the car is accelerating, braking, or coasting. To determine this,
#I have set my parameters for coasting at -.1 m/s^2 to .05 m/s^2. I chose these parameters based on what 
#I found on sae vehicles as their typical coasting acceleration, and accelerating time frames are whenever
#the acceleration is greater than .05 m/s^2, and braking is when it is less than -.1 m/s^2


