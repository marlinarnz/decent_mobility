"""Tests of toy/example cases."""


def test_example0():
    """Simplest possible example.

    - 1 agent.
    - 2 locations.
    - Mobility needs between those 2 locations: outbound and inbound.
    - Minimum alternatives to meet those needs.

    This example demonstrates that :meth:`Agent.has_decent_mobility()`
    :meth:`Model.universal_decent_mobility()` return either :any:`True` or :any:`False`.
    """

    from mobility import Agent, Alternative, GridLocation, Model, Need
    from mobility import LocationType as LT
    from mobility import TripPurpose as TP

    # Two locations
    l1 = GridLocation(x=0, y=0)
    l2 = GridLocation(x=1, y=0)

    # One agent
    a = Agent(
        location={LT.home: l1, LT.work: l2},
        need=[
            Need(purpose=TP.commute, origin=LT.home, destination=LT.work, count=5),
            Need(purpose=TP.commute, origin=LT.work, destination=LT.home, count=5),
        ],
        plan=[],
    )
    # A model with the single agent
    m = Model(agent=[a])

    # This agent has no alternatives to meet their needs, so they do not have decent
    # mobility
    assert False is a.has_decent_mobility()
    # Universal decent mobility is not achieved
    assert False is m.universal_decent_mobility()

    # Now provide the agent with some alternatives
    a.plan = [
        Alternative(origin=l1, destination=l2, mode="bus"),
        Alternative(origin=l2, destination=l1, mode="bus"),
    ]

    # The agent now has decent mobility
    assert True is a.has_decent_mobility()
    # Universal decent mobility is achieved
    assert True is m.universal_decent_mobility()
