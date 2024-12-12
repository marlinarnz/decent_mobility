"""Minimal operationalization of the decent mobility criteria.

As discussed at the 2024-12 writing retreat
"""

from collections.abc import Collection
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional

# Coded concepts
Gender = Enum("Gender", "FLINT* male")
Purpose = Enum("Purpose", "work leisure")


@dataclass
class Person:
    """A single/individual person.

    A person has 0 or more measurable characteristics or attributes. Currently the class
    has only 1 example: :attr:`gender`.
    """

    gender: Gender

    #: Counts of trips needed by purpose.
    trip_needs: dict[Purpose, int] = field(default_factory=dict)

    def trip_needs_from_persona(self) -> None:
        """Set :attr:`trip_needs` by identifying an applicable :class:`Persona`.

        .. todo:: Complete implementation.
        """
        # TODO Identify the persona applicable to this individual
        # Currently this uses the first element in PERSONA.
        pa = PERSONA[0]

        # Adopt the trip needs of this persona
        self.trip_needs = copy(pa.trip_needs)


@dataclass
class Persona:
    """A ‘persona’: a set of personal characteristics shared by 1 or more people.

    Its attributes are exact duplicates of a subset of the attributes of a Person.
    The :meth:`is_member` method allows to classify 1 or more people as being members,
    or not, of group of people associated with the persona.
    """

    gender: Gender

    def is_member(self, person: Person) -> bool:
        """Return :any:`True` if `person` is a member of this group."""
        return person.gender == self.gender

    @property
    def trip_needs(self) -> dict[Purpose, int]:
        """Decent mobility trip needs for all people in this persona.

        .. todo:: Apply the logic from our shared spreadsheet.
        """
        return {
            Purpose["work"]: 4,
            Purpose["leisure"]: 1,
        }


@dataclass(frozen=True)
class POI:
    """A point of interest."""

    #: The needs that can be satisfied by a trip to this place.
    needs_served: Purpose


@dataclass(frozen=True)
class Trip:
    """A single trip."""

    #: Distance travelled [kilometre].
    distance: float
    #: Duration of the trip [hours].
    time: float
    #: Place or point of interest that is visited.
    destination: POI


@dataclass
class TravelPlan:
    """A travel plan for a :class:`.Person`."""

    #: Duration of the time period covered by the travel plan. Default: 1 week.
    period_covered: int = 7

    #: Collection of trips that constitute the travel plan.
    trips: Collection[Trip] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"<Travel plan with {len(self.trips)} trips in {self.period_covered} days>"
        )


#: A collection of personas.
PERSONA = (
    Persona(Gender["male"]),
    Persona(Gender["FLINT*"]),
)

#: Travel time maximum for decent mobility [hours / day].
TIME_BUDGET = 1.2

# Utility functions that give information about a travel plan


def travel_distance(
    tp: TravelPlan, base: Optional[Literal["day", "year"]] = None
) -> float:
    """Sum of the travel distance for trips `tp` [kilometre / {time}].

    Parameters
    ----------
    base
        Compute the travel distance for a given time period. If :any:`None`, simply give
        the total. If "day", divide by :attr:`TravelPlan.period_covered` to give the
        travel distance per day. If "year", give the distance per year.
    """
    d = sum(t.distance for t in tp.trips)
    t = {
        None: 1.0,
        "day": tp.period_covered,
        "year": tp.period_covered / 365,
    }[base]
    return d / t


def travel_time(tp: TravelPlan) -> float:
    """Sum of the travel distance for trips `tp` [kilometre]."""
    return sum(t.time for t in tp.trips)


def trip_count(tp: TravelPlan, purpose: Purpose) -> int:
    """Count of trips for a certain `purpose`."""
    return sum(map(lambda t: t.destination.needs_served == purpose, tp.trips))


def has_decent_mobility(person: Person, tp: TravelPlan) -> bool:
    """:any:`True` if `tp` provides decent mobility for `person`.

    The criteria applied are:

    1. Needs satisfaction: for every :class:`Purpose`, the :func:`trip_count` of trips
       in `tp` for that purpose is equal to, or greater than, the
       :attr:`Person.trip_needs`.
    2. Constraint on travel time is satisfied: :func:`travel_time` of the `tp` less than
       or equal to :data:`TIME_BUDGET`.
    """
    # Compute data needed for the decent mobility criteria
    # Count trips by each Purpose
    counts = {p: trip_count(tp, p) for p in Purpose}

    # Travel time [hours / day] = Total travel time [hours] / period covered [day]
    daily_time = travel_time(tp) / tp.period_covered

    # Person's trip needs
    needs = person.trip_needs

    # Apply the decent mobility criteria
    return all(counts[p] >= needs[p] for p in Purpose) and (daily_time <= TIME_BUDGET)


def make_travel_plan(data: dict[Purpose, tuple[int, float, float]]) -> TravelPlan:
    """Construct a :class:`TravelPlan` using aggregate `data`.

    The returned plan includes a number of identical :class:`.Trips` for each purpose.
    For a real person, each trip for a given purpose, for instance Purpose.leisure,
    would differ in length in duration. Using this function is the same as assuming that
    the given values are averages across all trips for each purpose.

    Parameters
    ----------
    data
        Maps :class:`.Purpose` to 3-tuples with elements:

        1. Number of such trips in the travel plan.
        2. :attr:`.Trip.distance`.
        3. :attr:`.Trip.duration`.
    """
    trips: list[Trip] = []

    for purpose, (count, distance, time) in data.items():
        poi = POI(needs_served=purpose)
        trip = Trip(distance, time, destination=poi)
        trips.extend(copy(trip) for _ in range(count))

    return TravelPlan(trips=trips)


def demo():
    # Construct a person
    person = Person(Gender["FLINT*"])

    # Identify the PERSONA to which this person belongs; adopt that persona's trip needs
    person.trip_needs_from_persona()

    # Make up a set of trips that this person might take
    tp = make_travel_plan(
        {
            Purpose["work"]: (4, 1.0, 0.1),
            Purpose["leisure"]: (1, 2.0, 0.2),
        }
    )

    # Apply the decent mobility criteria, and show the associated travel distance:
    print(
        tp,
        f"Provides decent mobility: {has_decent_mobility(person, tp)}",
        f"Travel distance (PDT): {travel_distance(tp)} km",
        f"                     = {travel_distance(tp, 'year'):.0f} km / year",
        sep="\n",
        end="\n\n",
    )

    # Different set of trips that doesn't provide decent mobility, yet has a longer
    # total travel distance
    tp = make_travel_plan(
        {
            Purpose["work"]: (3, 1.0, 0.1),
            Purpose["leisure"]: (10, 2.0, 0.2),
        }
    )

    print(
        tp,
        f"Provides decent mobility: {has_decent_mobility(person, tp)}",
        f"Travel distance (PDT): {travel_distance(tp)} km",
        f"                     = {travel_distance(tp, 'year'):.0f} km / year",
        sep="\n",
        end="\n\n",
    )


if __name__ == "__main__":
    demo()
