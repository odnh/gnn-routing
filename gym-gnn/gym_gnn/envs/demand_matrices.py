from typing import Callable, Type, Iterator
import networkx as nx
import numpy as np
from numpy.random import RandomState

Demand = Type[np.ndarray]

# NB: demands are not matrices, but 1D arrays in deterministic ordering

def sparsify(
        demands: Demand,
        sparsity: float,
        random_state: RandomState) -> Demand:
    """
    Sparsifies the given demands by the given sparsity (probability of
    dropping a flow demand between two nodes)
    Args:
      demands: demand matrix
      sparsity: between 0 and 1
    """
    sparsified_demands = demands.copy()
    for i in range(len(demands)):
        if random_state.uniform() < sparsity:
            sparsified_demands[i] = 0
    return sparsified_demands

def gravity_demand(number_of_flows: int, graph: nx.Graph) -> Demand:
    """
    Generates gravity demand (deterministic, based on bandwidth) for one time
    step
    """
    #TODO: implement
    pass

def bimodal_demand(
        number_of_flows: int,
        random_state: RandomState = RandomState()) -> Demand:
    """
    Generates bimodal demand (probabilistic) for one time step
    """
    demand = np.zeros(number_of_flows, dtype=float)
    for i in range(number_of_flows):
        # Coin flip
        if random_state.uniform() < 0.8:
            # Standard flow
            demand[i] = random_state.normal(150, 20)
        else:
            # Elephant flow
            demand[i] = random_state.normal(400, 20)
    return demand

def cyclical_sequence(
        demand_generator: Callable[[], Demand],
        length: int,
        q: int,
        sparsity: float,
        random_state: RandomState = RandomState()) -> Iterator[Demand]:
    """
    Creates a sequence of length `length` which is a continuous cycle for a
    sequence of demands length q. Demands are all sparsified.
    Args:
      demand_generator: returns a demand
      flows: number of flows
      length: overall sequence length
      q: length of cycle used to build sequence
      sparsity: value used for sparsification of demands
      random_state: so that a shared or new seed can be used
    Returns:
      sequence as an iterator
    """
    demand = demand_generator()
    short_sequence = [sparsify(demand_generator(), sparsity, random_state) for _ in range(q)]
    i = 0
    for _ in range(length):
        yield short_sequence[i]
        i = (i + 1) % q

def average_sequence(
        demand_generator: Callable[[], Demand],
        length: int,
        q: int,
        sparsity: float,
        random_state: RandomState = RandomState()) -> Iterator[Demand]:
    """
    Creates a sequence of length `length` where each demand is the average
    over the previous q demands.
    Args:
      demand_generator: returns a demand
      length: overall sequence length
      q: length of averaging history
      sparsity: value used for sparsification of demands
      random_state: so that a shared or new seed can be used
    Returns:
      sequence as an iterator
    """
    # initialise history to length q
    history = [sparsify(demand_generator(), sparsity, random_state) for _ in range(q)]
    for _ in range(length):
        yield np.mean(history, axis=1)
        history.pop(0)
        history.append(sparsify(demand_generator(), sparsity, random_state))

def random_sequence(
        demand_generator: Callable[[], Demand],
        length: int,
        sparsity: float,
        random_state: RandomState = RandomState()) -> Iterator[Demand]:
    """
    Creates a sequence of length `length` from demands generated using the
    given function. There should be no dependency between items in the
    sequence. Demands are all sparsified.
    Args:
      demand_generator: returns a demand
      length: overall sequence length
      sparsity: value used for sparsification of demands
      random_state: so that a shared or new seed can be used
    Returns:
      sequence as an iterator
    """
    for _ in range(length):
        yield sparsify(demand_generator(), sparsity, random_state)
