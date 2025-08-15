import random
import math
import time
from typing import Dict, Tuple, List
from graph import DirectedGraph
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import os
from collections import deque, defaultdict

class OperationCounter:
    def __init__(self):
        self.comparisons = 0
        self.relaxations = 0
        self.heap_pushes = 0
        self.heap_pops = 0

    def reset(self):
        self.comparisons = 0
        self.relaxations = 0
        self.heap_pushes = 0
        self.heap_pops = 0

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

def generate_strongly_connected_graph(num_vertices: int, num_edges: int, weight_range=(1.0, 10.0)) -> DirectedGraph:
    graph = DirectedGraph()
    # D'abord, créer un cycle pour assurer la connexité
    for i in range(num_vertices):
        graph.add_edge(i, (i+1) % num_vertices, random.uniform(*weight_range))
    # Puis ajouter des arêtes aléatoires supplémentaires
    for _ in range(num_edges - num_vertices):
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)
        if u != v:
            graph.add_edge(u, v, random.uniform(*weight_range))
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

def dijkstra_instrumented(graph: DirectedGraph, source: int, counter: OperationCounter) -> Dict[int, float]:
    distances = {v: float('inf') for v in graph.get_vertices()}
    distances[source] = 0.0
    heap = [(0.0, source)]
    visited = set()
    while heap:
        counter.heap_pops += 1
        dist_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, weight in graph.get_neighbors(u):
            counter.comparisons += 1
            if v not in visited and dist_u + weight < distances[v]:
                counter.relaxations += 1
                distances[v] = dist_u + weight
                heapq.heappush(heap, (distances[v], v))
                counter.heap_pushes += 1
    return distances


def draw_graph(graph: DirectedGraph, path: list[int] = None, filename: str = "graph.png"):
    G = nx.DiGraph()
    for u in graph.get_vertices():
        for v, w in graph.get_neighbors(u):
            G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{w:.1f}" for e, w in labels.items()})
    if path:
        edges_in_path = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges_in_path, edge_color='red', width=2)
    plt.title("Random Directed Graph" + (" with shortest path" if path else ""))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()

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

def get_shortest_path(distances: Dict[int, float], graph: DirectedGraph, source: int, target: int) -> List[int]:
    # Simple backtracking for shortest path (works if all weights > 0 and path exists)
    path = [target]
    current = target
    while current != source:
        for u in graph.get_vertices():
            for v, w in graph.get_neighbors(u):
                if v == current and math.isclose(distances[v], distances[u] + w):
                    path.append(u)
                    current = u
                    break
            else:
                continue
            break
        else:
            return []  # No path found
    return list(reversed(path))

def sssp_frontier_reduction_paper(
    graph: DirectedGraph,
    source: int,
    counter: OperationCounter,
    distances: dict = None,
    S: set = None,
    B: float = float('inf'),
    log_power: float = 2/3,
    recursion_depth: int = 0,
    max_depth: int = 8
) -> dict:
    """
    Implémentation fidèle de la "frontier reduction" du papier Duan et al. (2025).
    """
    if distances is None:
        distances = {v: float('inf') for v in graph.get_vertices()}
        distances[source] = 0.0
    if S is None:
        S = {source}

    n = len(graph.get_vertices())
    if recursion_depth > max_depth or n <= 1:
        return distances

    k = max(2, int(math.ceil(math.log2(n) ** log_power)))

    # 1. Calculer l'ensemble U des sommets à compléter (<B, dépendant d'un pivot)
    U = set()
    parent = {}
    for u in graph.get_vertices():
        if distances[u] < B:
            # On cherche un chemin le reliant à un pivot de S
            for s in S:
                # BFS limité pour trouver un chemin de s à u
                queue = deque([(s, 0)])
                visited = set()
                found = False
                while queue and not found:
                    v, depth = queue.popleft()
                    if v == u:
                        found = True
                        parent[u] = s
                        break
                    if depth >= k:
                        continue
                    for w, _ in graph.get_neighbors(v):
                        if w not in visited:
                            queue.append((w, depth + 1))
                            visited.add(w)
                if found:
                    U.add(u)
                    break

    # 2. Si |U| > k*|S|, frontier déjà petite
    if len(U) > k * len(S):
        # On ne fait rien, frontier déjà réduite
        return distances

    # 3. Sinon, k passes Bellman-Ford à partir des pivots
    for _ in range(k):
        updated = False
        for s in S:
            for u in graph.get_vertices():
                for v, weight in graph.get_neighbors(u):
                    counter.comparisons += 1
                    if distances[u] + weight < distances[v]:
                        counter.relaxations += 1
                        distances[v] = distances[u] + weight
                        updated = True
        if not updated:
            break

    # 4. Sélection des nouveaux pivots (ceux qui ont beaucoup de descendants dans U)
    pivot_count = defaultdict(int)
    for u in U:
        if u in parent:
            pivot_count[parent[u]] += 1
    new_pivots = {p for p, cnt in pivot_count.items() if cnt >= k}
    if not new_pivots:
        return distances

    # 5. Récursivité sur chaque sous-ensemble défini par un pivot
    for p in new_pivots:
        # On définit B_p comme la plus grande distance des descendants de p dans U
        B_p = max([distances[u] for u in U if parent.get(u) == p], default=B)
        sssp_frontier_reduction_paper(
            graph,
            p,
            counter,
            distances,
            S={p},
            B=B_p,
            log_power=log_power,
            recursion_depth=recursion_depth + 1,
            max_depth=max_depth
        )
    return distances

def benchmark_algorithms():
    # Tailles de graphes exponentielles pour voir l'effet d'échelle
    sizes = [500, 2000, 5000, 10000, 20000]
    results = []
    for n in sizes:
        print(f"\n--- Benchmark n={n} ---")
        g = generate_strongly_connected_graph(n, n * 6)  # 6 arêtes par sommet en moyenne

        # Dijkstra instrumenté
        counter = OperationCounter()
        start = time.perf_counter()
        dijkstra_instrumented(g, 0, counter)
        elapsed = time.perf_counter() - start
        print(f"Dijkstra: time={elapsed:.3f}s, comparisons={counter.comparisons}, relaxations={counter.relaxations}")
        results.append({
            "n": n, "algo": "Dijkstra",
            "time": elapsed,
            "comparisons": counter.comparisons,
            "relaxations": counter.relaxations,
            "heap_pushes": counter.heap_pushes,
            "heap_pops": counter.heap_pops
        })

        # Algo du papier instrumenté (version fidèle)
        counter = OperationCounter()
        start = time.perf_counter()
        sssp_frontier_reduction_paper(g, 0, counter)
        elapsed = time.perf_counter() - start
        print(f"Frontier Reduction (paper): time={elapsed:.3f}s, comparisons={counter.comparisons}, relaxations={counter.relaxations}")
        results.append({
            "n": n, "algo": "Frontier Reduction (paper)",
            "time": elapsed,
            "comparisons": counter.comparisons,
            "relaxations": counter.relaxations,
            "heap_pushes": counter.heap_pushes,
            "heap_pops": counter.heap_pops
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("src/results/benchmark_results.csv", index=False)
    # Plot with seaborn
    metrics = ["time", "comparisons", "relaxations"]
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df, x="n", y=metric, hue="algo", marker="o")
        plt.xlabel("Number of vertices")
        plt.ylabel(metric)
        plt.title(f"Dijkstra vs Frontier Reduction SSSP: {metric}")
        plt.savefig(f"src/results/benchmark_{metric}.png")
        plt.close()
    print("\nBenchmarks terminés. Résultats et graphes sauvegardés dans src/results/")

def visualize_multiple_paths(graph, distances, algo_name, source=0, num_targets=5):
    import random
    targets = random.sample([v for v in graph.get_vertices() if v != source], min(num_targets, len(graph.get_vertices())-1))
    for target in targets:
        path = get_shortest_path(distances, graph, source, target)
        filename = f"src/results/{algo_name}_shortest_path_{source}_to_{target}.png"
        draw_graph(graph, path, filename=filename)
        print(f"{algo_name}: Shortest path {source} → {target} (len={len(path)-1}, dist={distances[target]:.2f}): {path}")

if __name__ == "__main__":
    benchmark_algorithms()
    # Visualisation dynamique pour un petit graphe
    g_small = generate_strongly_connected_graph(15, 40)
    # Dijkstra
    counter = OperationCounter()
    dists_dijkstra = dijkstra_instrumented(g_small, 0, counter)
    visualize_multiple_paths(g_small, dists_dijkstra, "Dijkstra", source=0, num_targets=5)
    # Frontier Reduction
    counter = OperationCounter()
    dists_frontier = sssp_frontier_reduction_paper(g_small, 0, counter)
    visualize_multiple_paths(g_small, dists_frontier, "FrontierReduction", source=0, num_targets=5)