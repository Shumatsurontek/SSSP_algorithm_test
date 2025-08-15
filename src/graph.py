"""
Graph data structure for SSSP algorithms.
"""

from typing import Dict, List, Tuple, Optional
import heapq


class DirectedGraph:
    """
    A directed graph representation optimized for SSSP algorithms.
    
    This implementation supports weighted edges and is designed to work
    efficiently with the algorithms described in the research paper.
    """
    
    def __init__(self, vertices: Optional[int] = None):
        """
        Initialize a directed graph.
        
        Args:
            vertices: Number of vertices (optional, can grow dynamically)
        """
        self.vertices = vertices if vertices else 0
        self.edges: Dict[int, List[Tuple[int, float]]] = {}
        self.edge_count = 0
    
    def add_vertex(self, vertex: int) -> None:
        """Add a vertex to the graph."""
        if vertex not in self.edges:
            self.edges[vertex] = []
            self.vertices = max(self.vertices, vertex + 1)
    
    def add_edge(self, source: int, destination: int, weight: float = 1.0) -> None:
        """
        Add a directed edge to the graph.
        
        Args:
            source: Source vertex
            destination: Destination vertex  
            weight: Edge weight (default 1.0)
        """
        self.add_vertex(source)
        self.add_vertex(destination)
        self.edges[source].append((destination, weight))
        self.edge_count += 1
    
    def get_neighbors(self, vertex: int) -> List[Tuple[int, float]]:
        """Get all neighbors of a vertex with their edge weights."""
        return self.edges.get(vertex, [])
    
    def get_vertices(self) -> List[int]:
        """Get all vertices in the graph."""
        return list(self.edges.keys())
    
    def __str__(self) -> str:
        """String representation of the graph."""
        result = f"DirectedGraph({self.vertices} vertices, {self.edge_count} edges)\n"
        for vertex, neighbors in self.edges.items():
            if neighbors:
                result += f"  {vertex}: {neighbors}\n"
        return result
