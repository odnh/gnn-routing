from typing import Type, Callable, Iterator

import networkx as nx
import numpy as np
from numpy.random import RandomState

Demand = Type[np.ndarray]


# NB: demands are not matrices, but 1D arrays in deterministic ordering

def sparsify(
        demands: Demand,
        sparsity: float,
        random_state: RandomState = RandomState()) -> Demand:
    """
    Sparsifies the given demands by the given sparsity (probability of
    dropping a flow demand between two nodes)
    Args:
      demands: demand matrix
      sparsity: between 0 and 1
      random_state: a numpy RandomState to control seed
    Returns:
        A sparsified demand array
    """
    sparsified_demands = demands.copy()
    for i in range(len(demands)):
        if random_state.uniform() < sparsity:
            sparsified_demands[i] = 0
    return sparsified_demands


def gravity_demand(graph: nx.DiGraph) -> Demand:
    """
    Generates gravity demand (deterministic, based on bandwidth) for one time
    step
    Args:
        graph: Networkx DiGraph with 'weight' on edges to generate demand from
    Returns:
        A demand array
    """
    num_nodes = graph.number_of_nodes()
    sorted_edges = sorted(graph.edges(data=True))
    edge_weights = [e[2]['weight'] for e in sorted_edges]

    total_flow = sum(edge_weights)
    node_in_flow = np.zeros(num_nodes, np.float32)
    node_out_flow = np.zeros(num_nodes, np.float32)

    for i, edge in enumerate(sorted_edges):
        node_in_flow[edge[1]] += edge_weights[i]
        node_out_flow[edge[0]] += edge_weights[i]

    return np.divide(np.array(
        [node_out_flow[i] * node_in_flow[j] for i in range(num_nodes) for j in
         range(num_nodes) if i != j]), total_flow * 2)


def bimodal_demand(
        number_of_flows: int,
        random_state: RandomState = RandomState()) -> Demand:
    # TODO: temper size of demand based on network bandwidth
    """
    Generates bimodal demand (probabilistic) for one time step
    Args:
        number_of_flows: the number of flows to create
        random_state: numpy RandomState (for seeding)
    Returns:
        A demand array
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
        demand_generator: Callable[[np.random.RandomState], Demand],
        length: int,
        q: int,
        sparsity: float,
        seed: int) -> Iterator[Demand]:
    """
    Creates a sequence of length `length` which is a continuous cycle for a
    sequence of demands length q. Demands are all sparsified.
    Args:
      demand_generator: returns a demand
      flows: number of flows
      length: overall sequence length
      q: length of cycle used to build sequence
      sparsity: value used for sparsification of demands
      seed: seed randomness for dm generation
    Returns: sequence as an iterator
    """
    random_state = np.random.RandomState(seed=seed)
    short_sequence = [
        sparsify(demand_generator(random_state), sparsity, random_state) for _
        in range(q)]
    i = 0
    for _ in range(length):
        yield short_sequence[i]
        i = (i + 1) % q


def average_sequence(
        demand_generator: Callable[[], Demand],
        length: int,
        q: int,
        sparsity: float,
        seed: int) -> Iterator[Demand]:
    """
    Creates a sequence of length `length` where each demand is the average
    over the previous q demands.
    Args:
      demand_generator: returns a demand
      length: overall sequence length
      q: length of averaging history
      sparsity: value used for sparsification of demands
      seed: seed randomness for dm generation
    Returns:
      sequence as an iterator
    """
    random_state = np.random.RandomState(seed=seed)
    # initialise history to length q
    history = [sparsify(demand_generator(random_state), sparsity, random_state)
               for _ in
               range(q)]
    for _ in range(length):
        yield np.mean(history, axis=0)
        history.pop(0)
        history.append(
            sparsify(demand_generator(random_state), sparsity, random_state))


def random_sequence(
        demand_generator: Callable[[], Demand],
        length: int,
        sparsity: float,
        seed: int) -> Iterator[Demand]:
    """
    Creates a sequence of length `length` from demands generated using the
    given function. There should be no dependency between items in the
    sequence. Demands are all sparsified.
    Args:
      demand_generator: returns a demand
      length: overall sequence length
      sparsity: value used for sparsification of demands
      seed: seed randomness for dm generation
    Returns:
      sequence as an iterator
    """
    random_state = np.random.RandomState(seed=seed)
    for _ in range(length):
        yield sparsify(demand_generator(), sparsity, random_state)
