"""Shifting from passenger-kilometres to trips for the mobility DLS

This code demonstrates a simple calculation based on number of trips by different trip
types, and an average length for each type of trip.

This is updated version of trips_xarray.py using the genno package.
"""
import pint
import xarray as xr
from genno import Computer, Key, Quantity

# Structure: concept/dimension IDs and lists of codes.
#
# These are illustrative only. For the project we should select dimensions as needed and
# code lists based on literature.

# Regions, using ISO 3166-1 alpha-2 for example
r = ["JP", "BR", "IN"]

# Trip types or categories
tt = [
    "work",
    "other",  # to access shopping, services, etc.
    "leisure",  # within the 'daily life area'
    "tourism",  # long-distance leisure trips, outside of the 'daily life area'
]

# Trip mode
tm = ["active", "motorized"]

DIMS = {"r": r, "tt": tt, "tm": tm}  # All dimensions used in this file

# Shorthand functions


def q(data, dims, **kwargs):
    """Shorthand to create a N-dimensional quantity."""

    coords = {dim: DIMS[dim] for dim in dims.split()}
    return Quantity(xr.DataArray(data, coords=coords), **kwargs)


def show(key: Key):
    """Show `key` and its computed value."""
    print(key, c.get(key), sep="\n", end="\n\n")


# 1. Existing approach

c = Computer()

# Threshold used in prior DLE work
c.add("base distance:", Quantity(10000, units="km / a"))

# Urban population fraction [dimensionless] as an example of an attribute with an `r`
# dimension
c.add("urban frac:r", q([0.9, 0.5, 0.75], dims="r"))


def scale_by_urban_frac(qty: Quantity, urban_frac: Quantity) -> Quantity:
    """Adjust `qty` (scalar) based on urban population fraction (dimension `r`).

    The functional form and coefficients mean nothing; simply for illustration. The
    result has the same dimensionality as `urban_frac`; the value for r="JP" is used as
    a reference.
    """
    return qty * (1 + 0.4 * (urban_frac.sel(r="JP").item() - urban_frac))


# Produces distances in the range (10000, 11600) km
k = Key("distance 1:r")
c.add(k, scale_by_urban_frac, "base distance:", "urban frac:r")
show(k)


# 2. Adding trips
#
# Alternate expression of a decent living standard by number of trips. In this example,
# there is no r dimension; values are thus assumed the same for each `r`, but they need
# not be. These numbers could be drawn from NHTS or other travel surveys.
trips = q(
    [
        52 * 5 * 2,  # work:  52 weeks × 5 days/week × 2 trips/day
        52 * 4,  # other: 4 trips/week
        52 * 1,  # leisure: 1 trip per week
        5,  # tourism: a little less than 1 trip every 2 months
    ],
    dims="tt",
)
c.add("trips:tt", trips)

# Length of a single trip of each trip_type in the reference region
c.add("trip length:tt:JP", q([10, 10, 30, 232], dims="tt", units="km"))

# Scale the trip lengths by region. The function is the same one used above, but instead
# of scaling the 0-d "base distance" to get 1-D, we scale a 1-D quantity and get 2-D.
k = Key("trip length:r-tt")
c.add(k, scale_by_urban_frac, "trip length:tt:JP", "urban frac:r")
show(k)

# Compute total distance
k = Key("distance 2:r-tt")
c.add(k, "mul", "trips:tt", "trip length:r-tt")

# Drop the `tt` dimension (but retain `r`) to indicate a partial sum across trip_types
show(k / "tt")


# 3. Use trip-based calculation to add nuance
#
# For example: we assume that while greater urban population fraction affects the trip
# length in the daily life area, it does not affect the length of long-distance tourism
# trips, which are governed by the (non-changing) distances between cities.
#
# Again, only a minimal, contrived example: in principle the function(s) used can be
# more complex.


def same_distance_tourism(trip_length: Quantity) -> Quantity:
    """tt="tourism" all the same as r="JP"; all others distinct as scaled."""
    from genno.operator import concat

    return concat(
        trip_length.sel({"tt": tt[:-1]}),
        trip_length.sel({"r": "JP", "tt": ["tourism"]}).expand_dims({"r": r}),
    )


tl2 = Key("trip length 2:r-tt")
c.add(tl2, same_distance_tourism, "trip length:r-tt")
show(tl2)

# Same as above
k = Key("distance 3:r-tt")
c.add(k, "mul", "trips:tt", tl2)
show(k / "tt")

# Display how this calculation is structured
print(c.describe(k / "tt"))


# 4. Use trip-based calculation to capture geographic/scenario differences
#
# We illustrate by construction of variables that would be used in calculations similar
# to the above. The calculations are not shown; but in all cases they result in distance
# per person per year that feeds into the remainder of the DLE analysis workflow.

# Add the 'hidden' walking segments at the beginning end of the reference trips (to and
# from a transit stop in Japan).
c.add("trips:tm-tt:ref", q([[0.2, 0.2, 0.2, 0.2], [10, 10, 30, 232]], dims="tm tt"))

# Other scenarios: person with no access to motorized transport at all = not decent
# mobility, because they do *too much* walking to access work, other services, etc.
c.add("trips:tm-tt:long walk", q([[10, 10, 10, 10], [0, 0, 0, 0]], dims="tm tt"))

# Person living in a dense, active-transport-friendly community, making full use of
# telecommuting to reduce work travel, but still taking vacations, etc.
c.add("trips:tm-tt:dense TOD", q([[2, 2, 0.2, 0.2], [0, 0, 30, 232]], dims="tm tt"))

# Not the main focus of DLE, but this person takes excess leisure and tourism trips,
# perhaps driving up the demand for energy/low-carbon transport equipment, thereby
# making it less affordable
c.add("frequent flyer:tt", "add", "trips:tt", q([0, 0, 52, 10], dims="tt"))

# 5. Add other outputs: time, money
#
# Here these have dimension `tt`, but they more likely would have `tm` and other
# dimensions

pint.get_application_registry().define("USD = [money]")
c.add("price:tt", q([0.1, 0.15, 0.2, 1.0], dims="tt", units="USD/km"))
c.add("speed:tt", q([50.0, 50.0, 40.0, 100.0], dims="tt", units="km/h"))

# Total travel time = distance / speed
k_distance = k
k_time = Key("time:r-tt")
c.add(k_time, "div", k_distance, "speed:tt")

# Total travel cost = distance * price
k_cost = Key("cost:r-tt")
c.add(k_cost, "mul", k_distance, "price:tt")

# Compute distance, time, and cost at once
k_all = Key("all")
c.add(k_all, [k_distance / "tt", k_time / "tt", k_cost / "tt"])

# Show what would be done
print(c.describe(k_all))

# Show the result
print(c.get(k_all))
