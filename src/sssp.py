import random
import time
from typing import Dict, Tuple
from graph import DirectedGraph
import heapq
import matplotlib.pyplot as plt

def generate_random_graph(num_vertices: int, num_edges: int, weight_range: Tuple[float, float] = (1.0, 10.0)) -> DirectedGraph:
    graph = DirectedGraph()
    for _ in range(num_edges):
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)
        while v == u:
            v = random.randint(0, num_vertices - 1)
        w = random.uniform(*weight_range)
        graph.add_edge(u, v, w)
    return graph

def dijkstra(graph: DirectedGraph, source: int) -> Dict[int, float]:
    distances = {v: float('inf') for v in graph.get_vertices()}
    distances[source] = 0.0
    heap = [(0.0, source)]
    visited = set()
    while heap:
        dist_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, weight in graph.get_neighbors(u):
            if v not in visited and dist_u + weight < distances[v]:
                distances[v] = dist_u + weight
                heapq.heappush(heap, (distances[v], v))
    return distances

# Version simplifiée de "frontier reduction" (illustrative, pas l'algo optimal du papier)
def frontier_reduction_sssp(graph: DirectedGraph, source: int, k: int = 3) -> Dict[int, float]:
    # Partitionne les sommets, puis fait des passes Bellman-Ford sur des sous-ensembles
    distances = {v: float('inf') for v in graph.get_vertices()}
    distances[source] = 0.0
    for _ in range(k):
        updated = False
        for u in graph.get_vertices():
            for v, weight in graph.get_neighbors(u):
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    updated = True
        if not updated:
            break
    return distances

def benchmark_algorithms():
    sizes = [100, 200, 400, 800]
    dijkstra_times = []
    frontier_times = []

    for n in sizes:
        g = generate_random_graph(n, n * 4)
        # Dijkstra
        start = time.perf_counter()
        dijkstra(g, 0)
        dijkstra_times.append(time.perf_counter() - start)
        # Frontier reduction
        start = time.perf_counter()
        frontier_reduction_sssp(g, 0, k=5)
        frontier_times.append(time.perf_counter() - start)

    plt.plot(sizes, dijkstra_times, label="Dijkstra")
    plt.plot(sizes, frontier_times, label="Frontier Reduction (simplified)")
    plt.xlabel("Number of vertices")
    plt.ylabel("Execution time (s)")
    plt.legend()
    plt.title("Dijkstra vs Frontier Reduction SSSP (random graphs)")
    plt.show()
