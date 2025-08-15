"""
Tests for the DirectedGraph class.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graph import DirectedGraph


class TestDirectedGraph:
    """Test cases for DirectedGraph class."""
    
    def test_empty_graph(self):
        """Test creating an empty graph."""
        graph = DirectedGraph()
        assert graph.vertices == 0
        assert graph.edge_count == 0
        assert len(graph.get_vertices()) == 0
    
    def test_add_vertex(self):
        """Test adding vertices to the graph."""
        graph = DirectedGraph()
        graph.add_vertex(0)
        graph.add_vertex(1)
        
        assert graph.vertices == 2
        assert 0 in graph.get_vertices()
        assert 1 in graph.get_vertices()
    
    def test_add_edge(self):
        """Test adding edges to the graph."""
        graph = DirectedGraph()
        graph.add_edge(0, 1, 2.5)
        graph.add_edge(1, 2, 1.0)
        
        assert graph.edge_count == 2
        assert graph.get_neighbors(0) == [(1, 2.5)]
        assert graph.get_neighbors(1) == [(2, 1.0)]
        assert graph.get_neighbors(2) == []
    
    def test_graph_with_preset_vertices(self):
        """Test creating graph with preset number of vertices."""
        graph = DirectedGraph(vertices=5)
        assert graph.vertices == 5
        
        graph.add_edge(0, 4, 1.0)
        assert graph.vertices == 5  # Should not change
    
    def test_string_representation(self):
        """Test string representation of graph."""
        graph = DirectedGraph()
        graph.add_edge(0, 1, 1.0)
        graph.add_edge(0, 2, 2.0)
        
        str_repr = str(graph)
        assert "DirectedGraph" in str_repr
        assert "2 vertices" in str_repr
        assert "2 edges" in str_repr
