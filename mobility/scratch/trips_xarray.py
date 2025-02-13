"""Shifting from passenger-kilometres to trips for the mobility DLS.

This code demonstrates a simple calculation based on number of trips by different trip
types, and an average length for each type of trip.

This older version was originally authored 2020-03-20 by Paul Kishimoto in the (private)
https://github.com/iiasa/DLE-scaleup repository and uses xarray directly. trips.py is an
updated version using genno.
"""


import xarray as xr

# 1. Existing approach ---

# Some regions
ds = xr.Dataset()
ds["region"] = ["JP", "BR", "IN"]

# Urban population fraction [0] as an example of a region attribute
ds["urban frac"] = xr.DataArray([0.9, 0.5, 0.75], dims="region")

# Current decent mobility standard
ds["base distance"] = 10000  # [motorized passenger-kilometres / year]


def scale_by_urban_frac(var):
    """Adjust *var* based on urban population fraction.

    Functional form and coefficients mean nothing; simply for illustration.
    """
    return var * (1 + 0.4 * (ds["urban frac"].loc["JP"] - ds["urban frac"]))


# Produces distances in the range (10000, 11600) km
ds["distance 1"] = scale_by_urban_frac(ds["base distance"])
print(ds["distance 1"])


# 2. Adding trips

# Trip categories
ds["trip_type"] = [
    "work",
    "other",  # to access shopping, services, etc.
    "leisure",  # within the 'daily life area'
    "tourism",  # long-distance leisure trips, outside of the 'daily life area'
]

# Decent living standard expressed as number of trips. Same for every region.
# These numbers could be drawn from NHTS or other travel surveys.
ds["trips"] = xr.DataArray(
    [
        52 * 5 * 2,  # work:  52 weeks × 5 days/week × 2 trips/day
        52 * 4,  # other: 4 trips/week
        52 * 1,  # leisure: 1 trip per week
        5,  # tourism: a little less than 1 trip every 2 months
    ],
    dims="trip_type",
)

# Empty variable (all zeroes) for trip distances [kilometres / trip]
ds["trip length"] = xr.DataArray([[0.0] * 4] * 3, dims=("region", "trip_type"))

# Distances for the reference case
ds["trip length"].loc["JP", :] = [10, 10, 30, 232]

# Scale the trip lengths by region—exactly the same method as above
ds["trip length"] = scale_by_urban_frac(ds["trip length"].loc["JP", :])
print(ds["trip length"])

# Compute the distances—same as above
ds["distance 2"] = (ds["trips"] * ds["trip length"]).sum("trip_type")
print(ds["distance 2"])


# 3. Use trip-based calculation to add nuance
#
# We assume that while greater urban population fraction affects the trip length in the
# daily life area, it does not affect the length of long-distance tourism trips, which
# are governed by the (non-changing) distances between cities.

# Long-distance trips all the same length; others remain as scaled above
idx = dict(region="JP", trip_type="tourism")
ds["trip length"].loc[dict(trip_type="tourism")] = ds["trip length"].loc[idx]

# Compute distances; slightly smaller range than above
ds["distance 3"] = (ds["trips"] * ds["trip length"]).sum("trip_type")
print(ds["distance 3"])


# 4. Use trip-based calculation to capture geographic/scenario differences
#
# We illustrate by construction of variables that would be used in calculations similar
# to the above. The calculations are not shown; but in all cases they result in distance
# per person per year that feeds into the remainder of the DLE analysis workflow.

# Add the 'hidden' walking segments at the beginning end of the reference trips (to and
# from a transit stop in Japan).
ds["trip_mode"] = ["active", "motorized"]
ds["reference"] = xr.DataArray(
    [[0.2, 0.2, 0.2, 0.2], [10, 10, 30, 232]], dims=("trip_mode", "trip_type")
)

# Other scenarios: person with no access to motorized transport at all = not decent
# mobility, because they do *too much* walking to access work, other services, etc.
ds["long walk"] = xr.DataArray(
    [[10, 10, 10, 10], [0, 0, 0, 0]], dims=("trip_mode", "trip_type")
)

# Person living in a dense, active-transport-friendly community, making full use of
# telecommuting to reduce work travel, but still taking vacations, etc.
ds["long walk"] = xr.DataArray(
    [[2, 2, 0.2, 0.2], [0, 0, 30, 232]], dims=("trip_mode", "trip_type")
)

# Not the main focus of DLE, but this person takes excess leisure and tourism trips,
# perhaps driving up the demand for energy/low-carbon transport equipment, thereby
# making it less affordable
ds["frequent flyer"] = ds["trips"] + [0, 0, 52, 10]

# etc.
