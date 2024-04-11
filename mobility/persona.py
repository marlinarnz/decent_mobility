import random
from typing import Dict, List

from alternative import Alternative

class Persona:
    """
    A class to characterize a persona with mobility behavior.

    Attributes:
        name (str): The name of the persona.
        typ_travel_time (float): The typical travel time of the persona.
        demand (dict): A dictionary specifying the discrete number of trips for each destination.
        trips (dict): To be computed: A dictionary containing information about trips taken by the persona.
    """

    def __init__(self, name: str, typ_travel_time: float, demand: Dict[str, int]):
        """
        Initialize a Persona instance.

        Args:
            name (str): The name of the persona.
            typ_travel_time (float): The typical travel time of the persona.
            demand (dict): A dictionary containing information about trips the persona wants to take.
                           Keys represent destinations and values represent the number of trips.
        """
        self.name = name
        self.typ_travel_time = typ_travel_time
        self.demand = demand
        self.trips = {key: [] for key in demand.keys()}

    def get_name(self) -> str:
        """
        Get the name of the persona.

        Returns:
            str: The name of the persona.
        """
        return self.name

    def get_typ_travel_time(self) -> float:
        """
        Get the typical travel time of the persona.

        Returns:
            float: The typical travel time of the persona.
        """
        return self.typ_travel_time

    def get_demand(self) -> Dict[str, int]:
        """
        Get the dictionary containing information about trips demand.

        Returns:
            dict: A dictionary where keys represent destinations and values represent
                  the number of desired trips.
        """
        return self.demand

    def get_trips(self) -> Dict[str, List['Alternative']]:
        """
        Get the dictionary containing information about trips taken by the persona.

        Returns:
            dict: A dictionary where keys represent destinations and values represent
                  the set of alternatives chosen.
        """
        return self.trips

    def compute_trips(self, alternatives: List['Alternative'], method: str, modes_unavailable=[]):
        """
        Compute a set of trips from the given list of alternatives
        that fulfills the persona's trips demand, saved into the trips attribute.
        Certain modes can be marked unavailable and won't be chosen.

        Args:
            alternatives (List[Alternative]): A list of Alternative objects.
            method (str): "random" chooses alternatives randomly.
                          "min_energy_typ_time" minimizes energy demand while not diverging
                          more than 10 minutes from the persona's typical travel time.
            modes_unavailable (List[str]): A list of mode names that are not available.

        Raises:
            ValueError: If there is not a single alternative for a specific key in the persona's trips dict.
        """

        for destination, count in self.demand.items():
            # Filter alternatives that match the destination
            destination_alternatives = [alt for alt in alternatives
                                        if alt.destination == destination
                                        and not alt.mode in modes_unavailable]

            if not destination_alternatives:
                raise ValueError(f"No alternative found for destination: {destination}")

            # Select trips from available alternatives based on the given method
            if method == 'random':
                selected_alternatives = random.choices(
                    destination_alternatives, k=count)
            elif method == 'min_energy_typ_time':
                selected_alternatives = self._select_min_energy_typ_time(
                    destination_alternatives, count)
            else:
                raise ValueError(method + " is not a valid method. Choose 'random' or 'min_energy_typ_time'")

            # Update trips dictionary with selected alternatives
            self.trips[destination] = list(selected_alternatives)


    def _select_min_energy_typ_time(self, destination_alternatives, count):
        raise NotImplementedError("Not yet implemented")
