from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Generator, List, Optional, Tuple, Union

#: Types of locations.
LocationType = Enum("LocationType", "home work")

#: Purposes for trips.
TripPurpose = Enum("TripPurpose", "commute other")


class Location(ABC):
    """Abstract base class for a location."""


@dataclass(frozen=True)
class GridLocation(Location):
    """A location denoted as (x, y) coordinates on a square grid."""

    x: float
    y: float

    def distance_to(self, other: "GridLocation") -> float:
        """Return the Euclidean (straight-line) distance to another GridLocation."""
        assert isinstance(other, GridLocation)
        return ((other.y - self.y) ** 2 + (other.x - self.x) ** 2) ** 0.5


@dataclass
class Need:
    """(Derived) Need for mobility."""

    purpose: TripPurpose
    origin: LocationType
    destination: LocationType

    #: Number of times the trip must be made in a typical week.
    count: int


@dataclass
class Trip(ABC):
    """Trip to be made, abstract of how it is made."""

    origin: Location
    destination: Location


@dataclass
class Alternative(Trip):
    """Specific transport/mobility alternative for a trip."""

    mode: str

    cost: float = 0.0
    distance: float = 0.0
    energy: float = 0.0
    time: float = 0.0

    def __post_init__(self):
        self.update_distance()

    def update_distance(self) -> None:
        """Set :attr:`.distance`.

        The true distance for a Trip or Alternative depends on the path available and
        taken. This method uses the Euclidean distance a placeholder/simplifying
        assumption.
        """
        assert isinstance(self.origin, GridLocation) and isinstance(
            self.destination, GridLocation
        )
        self.distance = self.origin.distance_to(self.destination)


@dataclass
class Agent:
    """Agent.

    This agent may be representative of a specific individual, a persona, or a
    representative agent in a population.
    """

    #: Collection of locations by type for this agent.
    location: Dict[LocationType, Location] = field(default_factory=dict)

    #: Trip needs of this agent.
    need: List[Need] = field(default_factory=list)

    #: Travel plan of this agent to satisfy their trip needs.
    plan: List[Alternative] = field(default_factory=list)

    def has_decent_mobility(self) -> bool:
        """Return :any:`.True` if the agent has decent mobility.

        This method may compute any number of sub-criteria, but must return a single
        :class:`bool` value.

        Currently, this uses the placeholder criterion that the agent has decent
        mobility **if and only if** there is 1 alternative for each need.
        """
        result = True

        for need, alternative in self.iter_np():
            if need and alternative is None:
                # No alternative for this need → it cannot be met
                result = False
                break

        return result

    def iter_np(
        self,
    ) -> Generator[Tuple[Optional[Need], Optional[Alternative]], None, None]:
        """Iterate over tuples of (Need, Alternative)."""
        # Transform needs into specific (o, d) tuples
        result: Dict[
            Tuple[Location, Location], Tuple[Optional[Need], Optional[Alternative]]
        ] = {
            (self.location[n.origin], self.location[n.destination]): (n, None)
            for n in self.need
        }

        # Match planned alternative to these needs
        for a in self.plan:
            key = (a.origin, a.destination)
            if key in result:
                result[key] = (result[key][0], a)
            else:
                result[key] = (None, a)

        yield from result.values()

    def total_distance(self) -> float:
        """Return the total travel distance of the agent's mobility :attr:`.plan`."""
        result = 0.0
        for need, alternative in self.iter_np():
            if need is None or alternative is None:
                continue
            result += need.count * alternative.distance

        return result


@dataclass
class Model:
    """A collection of agents."""

    agent: List[Agent]

    def universal_decent_mobility(self) -> bool:
        """Return :any:`.True` if all agents have decent mobility."""
        return all(a.has_decent_mobility() for a in self.agent)
