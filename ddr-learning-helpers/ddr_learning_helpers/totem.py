"""
Functions to read from the TOTEM dataset (graph and traffic matrices) and
convert the data into a form usable by the environment.
"""
import pathlib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Iterator

import networkx as nx
import numpy as np

Demand = np.array

totem_path = "{}/../../data/totem".format(
    pathlib.Path(__file__).parent.absolute())


class Totem:
    """
    Class to read in the TOTEM topology and traffic matrix files for use by the
    ddr environment
    """
    def __init__(self, weight: int = 100):
        self.node_mapping = {}
        self.graph = nx.DiGraph()
        self.read_graph(weight)  # Sets self.graph and self.node_mapping

    def read_graph(self, weight: int):
        """
        Reads in the topology graph from the xml file and sets the relevant
        class variables
        Args:
            A weight to put on all the edges
        """
        et = ET.parse("{}/topology-anonymised.xml".format(totem_path))
        nodes = et.find('topology').find('nodes').findall('node')
        edges = et.find('topology').find('links').findall('link')

        node_ids = [int(n.attrib['id']) for n in nodes]
        edge_ids = [
            (int(e.find('from').attrib['node']),
             int(e.find('to').attrib['node']))
            for e in edges]

        normalised_graph = nx.OrderedGraph()

        self.node_mapping = {node: i for i, node in enumerate(node_ids)}

        # Add nodes in order
        normalised_graph.add_nodes_from(range(len(node_ids)))
        # Add edges
        for src, dst in edge_ids:
            normalised_graph.add_edge(
                self.node_mapping[src], self.node_mapping[dst], weight=weight)

        self.graph = normalised_graph.to_directed()

    def get_traffic_matrix(self, time: datetime) -> Demand:
        """
        Reads in the traffic matrix associated with the given time
        Args:
            time: Time of the data to read
        Returns:
            A traffic matrix / demand
        """
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        demands = np.zeros(num_nodes * (num_nodes - 1), dtype=float)

        file_path = "{}/traffic-matrices/IntraTM-{}.xml" \
            .format(totem_path, time.strftime("%Y-%m-%d-%H-%M"))

        traffic_data = ET.parse(file_path)
        srcs = traffic_data.find('IntraTM').findall('src')

        for src in srcs:
            src_mapped = self.node_mapping[int(src.attrib['id'])]
            for dst in src.findall('dst'):
                dst_mapped = self.node_mapping[int(dst.attrib['id'])]
                if dst_mapped != src_mapped:
                    flow = float(dst.text)
                    offset = 0 if dst_mapped < src_mapped else -1
                    idx = src_mapped * (num_nodes - 1) + (
                            dst_mapped + offset)
                    demands[idx] = flow

        return demands

    def get_demands(self, start_date: datetime, end_date: datetime)\
            -> Iterator[Demand]:
        """
        Reads in the demand cml files from data folder. As data is in 15 minute
        intervals this is what while be returned.
        Args:
            start_date: start datetime (not before 2005-01-01-00-30)
            end_date: end datetime (not after 2005-04-29-16-45)
        Returns:
            An iterator of traffic matrices between the given datetimes
        """
        current_time = start_date
        while current_time < end_date:
            traffic_matrix = self.get_traffic_matrix(current_time)
            yield traffic_matrix
            current_time += timedelta(minutes=15)
