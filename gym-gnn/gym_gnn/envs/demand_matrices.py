from typing import Tuple, List, Dict, Callable, Generator, Type, Iterator
import networkx as nx
import numpy as np

Demand = Type[np.ndarray]

# NB: demands are not matrices, but 1D arrays in deterministic ordering

def sparsify(
        demands: Demand, 
        sparsity: float, random_state: np.random.RandomState) -> Demand:
    """
    Sparsifies the given demands by the given sparsity (probability of
    dropping a flow demand between two nodes)
    Args:
      demands: demand matrix
      sparsity: between 0 and 1
    """
    sparsified_demands = demands.copy()
    for i in range(len(demands)):
        if random_state.uniform() < sparsity :
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
    number_of_flows: int, random_state: np.random.RandomState) -> Demand:
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

def cyclical_sequence(demand_generator: Callable, length: int, q: int, sparsity: float, random_state: np.random.RandomState) -> Iterator[Demand]:
    """
    Creates a sequence of length `length` which is a continuous cycle for a
    sequence of demands length q. Demands are all sparsified.
    Args:
      demand_generator: returns a demand
      length: overall sequence length
      q: length of cycle used to build sequence
      sparsity: value used for sparsification of demands
      random_state: so that a shared or new seed can be used
    Returns:
      sequence as an iterator
    """
    pass

def average_sequence(demand_generator: Callable, length: int, q: int, random_state: np.random.RandomState) -> Iterator[Demand]:
    """
    Creates a sequence of length `length` which is a continuous cycle for a
    sequence of demands length q. Demands are all sparsified.
    Args:
      demand_generator: returns a demand
      length: overall sequence length
      q: length of cycle used to build sequence
      sparsity: value used for sparsification of demands
      random_state: so that a shared or new seed can be used
    Returns:
      sequence as an iterator
    """
    pass

def random_sequence(demand_generator: Callable, length: int, random_state: np.random.RandomState) -> Iterator[Demand]:
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
        #TODO: fill this in (need to decide on generator function api)
        yield demand_generator()
    pass