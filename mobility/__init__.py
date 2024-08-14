from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

#: Types of locations.
LocationType = Enum("LocationType", "home work")

#: Purposes for trips.
TripPurpose = Enum("TripPurpose", "commute other")


class Location(ABC):
    """Abstract base class for a location."""


@dataclass
class GridLocation(Location):
    """A location denoted as (x, y) coordinates on a square grid."""

    x: float
    y: float


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
        """
        # TEMPORARY criterion: there are at least as many alternative as there are needs
        # TODO
        # - Match alternatives to needs by (origin, destination)
        return len(self.plan) >= len(self.need)


@dataclass
class Model:
    """A collection of agents."""

    agent: List[Agent]

    def universal_decent_mobility(self) -> bool:
        """Return :any:`.True` if all agents have decent mobility."""
        return all(a.has_decent_mobility() for a in self.agent)
