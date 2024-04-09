from dataclasses import dataclass

@dataclass
class Alternative:
    """
    A class to characterize a travel alternative to a specific destination.

    Attributes:
        destination (str): The destination for the travel alternative.
                           The same name as in a Persona's demand dict.
        mode (str): The transport mode of the alternative.
        distance (float): The distance [km] to the destination using that mode.
        time (float): The travel time [minutes] of the trip.
        energy_demand (float): The final energy demand [kJ] of the trip.
    """
    destination: str
    mode: str
    distance: float
    time: float
    energy_demand: float
