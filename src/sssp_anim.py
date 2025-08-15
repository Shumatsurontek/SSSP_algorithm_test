import matplotlib.pyplot as plt
import networkx as nx
import imageio
import os
from typing import Dict, List
from graph import DirectedGraph
import heapq
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import os

def animate_sssp_progression(
    graph: DirectedGraph,
    progression_steps: List[Dict[int, float]],
    source: int,
    filename: str = "src/results/sssp_animation.gif"
):
    """
    Crée une animation GIF montrant la progression de la découverte des plus courts chemins.
    progression_steps : liste d'états du dict {sommet: distance} à chaque étape.
    """
    G = nx.DiGraph()
    for u in graph.get_vertices():
        for v, w in graph.get_neighbors(u):
            G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, seed=42)
    images = []
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    for step, distances in enumerate(progression_steps):
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{w:.1f}" for e, w in labels.items()})
        # Colorier les sommets atteints à ce step
        reached = [v for v, d in distances.items() if d < float('inf')]
        nx.draw_networkx_nodes(G, pos, nodelist=reached, node_color='orange')
        plt.title(f"SSSP Progression - Step {step}")
        temp_path = f"src/results/_frame_{step}.png"
        plt.savefig(temp_path)
        plt.close()
        images.append(imageio.imread(temp_path))

    imageio.mimsave(filename, images, duration=0.7)
    # Nettoyage des frames temporaires
    for step in range(len(progression_steps)):
        os.remove(f"src/results/_frame_{step}.png")
    print(f"Animation sauvegardée : {filename}")

def dijkstra_with_progression(graph: DirectedGraph, source: int):
    import heapq
    distances = {v: float('inf') for v in graph.get_vertices()}
    distances[source] = 0.0
    heap = [(0.0, source)]
    visited = set()
    progression = [distances.copy()]
    while heap:
        dist_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        for v, weight in graph.get_neighbors(u):
            if v not in visited and dist_u + weight < distances[v]:
                distances[v] = dist_u + weight
                heapq.heappush(heap, (distances[v], v))
        progression.append(distances.copy())
    return distances, progression

def animate_dijkstra_verbose(graph: DirectedGraph, source: int, filename: str = "src/results/dijkstra_verbose.gif"):
    G = nx.DiGraph()
    for u in graph.get_vertices():
        for v, w in graph.get_neighbors(u):
            G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, seed=42)
    distances = {v: float('inf') for v in graph.get_vertices()}
    distances[source] = 0.0
    heap = [(0.0, source)]
    visited = set()
    images = []
    step = 0
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    while heap:
        dist_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        # Draw current state
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', arrows=True)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f"{w:.1f}" for e, w in labels.items()})
        # Color nodes
        nx.draw_networkx_nodes(G, pos, nodelist=list(visited), node_color='lightgreen')
        nx.draw_networkx_nodes(G, pos, nodelist=[u], node_color='red')
        # Show distances
        dist_labels = {v: f"{distances[v]:.1f}" if distances[v] < float('inf') else "∞" for v in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=dist_labels, font_color='black')
        plt.title(f"Dijkstra Progression - Step {step} (processing {u})")
        temp_path = f"src/results/_frame_{step}.png"
        plt.savefig(temp_path)
        plt.close()
        images.append(imageio.imread(temp_path))
        step += 1
        # Relax neighbors
        for v, weight in graph.get_neighbors(u):
            if v not in visited and dist_u + weight < distances[v]:
                distances[v] = dist_u + weight
                heapq.heappush(heap, (distances[v], v))
    imageio.mimsave(filename, images, duration=0.7)
    for i in range(step):
        os.remove(f"src/results/_frame_{i}.png")
    print(f"Animation sauvegardée : {filename}")

if __name__ == "__main__":
    from sssp import generate_random_graph
    g = generate_random_graph(12, 25)
    dists, progression = dijkstra_with_progression(g, 0)
    animate_sssp_progression(g, progression, source=0, filename="src/results/dijkstra_progression.gif")
