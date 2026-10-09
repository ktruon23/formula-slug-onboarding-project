import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import math

# 1. Read csv file into a dataframe
df = pl.read_parquet('software-data.parquet')
df_speed = df.select('SME_TRQSPD_Speed')
print(df)
print(df_speed)

#function applies the velocity calculations given the wheel rpm, could also be adjusted to motor rpm 
#if needed (just add gear ratio)
def velocity_calculations(wheel_rpm):
    return (wheel_rpm*(2*(math.pi)/60)*.2)

#in order to find ground velocity, or just the velocity of the car, apply the velocity_calculations
#to convert the wheel rpm data into a dataframe with the ground velocity at each time to plot
df_ground_velocity = df.with_columns(
    pl.col("SME_TRQSPD_Speed")
    .map_elements(velocity_calculations, return_dtype=pl.Float64)
    .alias("ground_velocity")
)

y_values_velocity = df_ground_velocity["ground_velocity"]
x_values_velocity = df.select('Time')

plt.plot(x_values_velocity,y_values_velocity,linestyle="-")
plt.xlabel("Time")
plt.ylabel("Velocity (m/s)")

plt.savefig("ground_velocity.png", dpi=500)


#this function will determine whether the car is accelerating, braking, or coasting. To determine this,
#I have set my parameters for coasting at -.1 m/s^2 to .05 m/s^2. I chose these parameters based on what 
#I found on sae vehicles as their typical coasting acceleration, and accelerating time frames are whenever
#the acceleration is greater than .05 m/s^2, and braking is when it is less than -.1 m/s^2

WINDOW_S = .5   #this right here sets the window for which I will be evaluating acceleration, or rather 

accel_df = (
    df_ground_velocity
    .with_columns((pl.col("Time") / WINDOW_S).cast(pl.Int64).alias("window"))
    .group_by("window", maintain_order=True)
    .agg(
        pl.col("Time").first().alias("time_start"),
        pl.col("Time").last().alias("time_end"),
        pl.col("ground_velocity").first().alias("car_start"),
        pl.col("ground_velocity").last().alias("car_end"),
    )
    .with_columns(
        ((pl.col("car_end") - pl.col("car_start")) / (pl.col("time_end") - pl.col("time_start")))
        .alias("accel"),
        ((pl.col("time_start") + pl.col("time_end")) / 2).alias("time_mid"),
    )
    .drop_nulls("accel")
)

#based off of data online about SAE vehicles, the typical coasting acceleration of the car should be
#in between about -0.4 m/s^2 and .05 m/s^2, so I made these my parameters for the different states of
#the car in order to account for wind resistance, drag, and regen braking
COAST_LOW = -.5
COAST_HIGH = 0.1

# this uses polars version of an if-ekse-if statement in order to draw the boundaries for when the 
#car is accelerating, braking, or coasting, and added my 
accel_df = accel_df.with_columns(
    pl.when(pl.col("accel") > COAST_HIGH).then(pl.lit("accelerating"))
      .when(pl.col("accel") < COAST_LOW).then(pl.lit("braking"))
      .otherwise(pl.lit("coasting"))
      .alias("state")
)

# Time frames: merge consecutive windows with the same state
segments = (
    accel_df
    .with_columns(pl.col("state").rle_id().alias("run_id"))
    .group_by("run_id", maintain_order=True)
    .agg(
        pl.col("state").first(),
        pl.col("time_start").min().alias("start"),
        pl.col("time_end").max().alias("end"),
    )
)


# Three graphs, one per state
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9), constrained_layout=True)
for ax, state, color in zip(
    axes, ["accelerating", "braking", "coasting"], ["tab:green", "tab:red", "tab:blue"]
):
    sub = accel_df.filter(pl.col("state") == state)
    ax.scatter(sub["time_mid"], sub["accel"], color=color, s=12)
    ax.axhline(COAST_HIGH, color="gray", linestyle="-")
    ax.axhline(COAST_LOW, color="gray", linestyle="-")
    ax.set_title(state)
    ax.set_ylabel("Acceleration (m/s²)")

axes[-1].set_xlabel("Time (s)")
plt.savefig("states_over_time.png", dpi=500)

accel_df = accel_df.with_columns(
    ((pl.col("car_start") + pl.col("car_end")) / 2).alias("v_mean")
)

accelerating_df = accel_df.filter(pl.col("state") == "accelerating")
braking_df      = accel_df.filter(pl.col("state") == "braking")
coasting_df     = accel_df.filter(pl.col("state") == "coasting")

# Overall stats
print("Max speed (m/s):", df_ground_velocity["ground_velocity"].max())
print("Max acceleration (m/s^2):", accelerating_df["accel"].max())

print("Time accelerating (s):", accelerating_df.height * WINDOW_S)
print("Time coasting (s):", coasting_df.height * WINDOW_S)

# This will remove anything below 5 m/s
coasting_fast = coasting_df.filter(pl.col("v_mean") >= 5).sort("window")

# 2. a new stretch starts whenever consecutive windows aren't exactly 1 apart
coasting_fast = coasting_fast.with_columns(
    (pl.col("window") - pl.col("window").shift() != 1)
    .fill_null(True)
    .cum_sum()
    .alias("run_id")
)

# 3. measure each stretch and drop anything under 1 second
coast_segments = (
    coasting_fast
    .group_by("run_id", maintain_order=True)
    .agg(
        pl.col("time_start").min().alias("start"),
        pl.col("time_end").max().alias("end"),
        (pl.len() * WINDOW_S).alias("duration_s"),
    )
    .filter(pl.col("duration_s") >= 1)
)

# 4. keep only coasting rows that belong to a surviving stretch
coasting_clean = coasting_fast.filter(pl.col("run_id").is_in(coast_segments["run_id"]))

print(coast_segments)
print("Cleaned coasting time (s):", coasting_clean.height * WINDOW_S)
print("Max speed while coasting (m/s):", coasting_clean["car_end"].max())

car_mass     = 221.4     # kg
effective_mass = 244.08    # kg
G     = 9.81
air_density   = 1.225     

# x = v^2, y = dv/dt
x = (coasting_clean["v_mean"] ** 2).to_numpy()
y = coasting_clean["accel"].to_numpy()

slope, intercept = np.polyfit(x, y, 1)   # y = slope * x + intercept

drag_term = -slope * effective_mass               # 0.5 * rho * Cd * A
CdA       = drag_term / (0.5 * air_density)
Crr       = -intercept * effective_mass / (car_mass * G)

print(f"Drag Term = {drag_term:.4f}")
print(f"CdA = {CdA:.3f} m^2")
print(f"Rolling Resistance = {Crr:.4f}")

plt.figure()
plt.scatter(x, y, s=12)
plt.plot(x, slope * x + intercept, color="red")
plt.xlabel("v² (m²/s²)")
plt.ylabel("dv/dt (m/s²)")
plt.savefig("coastdown_fit.png", dpi=300)
