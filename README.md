# Single Source Shortest Path (SSSP) Algorithm Implementation

This repository contains an implementation and testing suite for Single Source Shortest Path algorithms, particularly focusing on the research presented in:

**"Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"**  
*Authors: Ran Duan, Jiayi Mao, Xiao Mao, Xinkai Shu, Longhui Yin*  
*ArXiv: https://arxiv.org/pdf/2504.17033v2*

## About

This project implements and tests various SSSP algorithms for directed graphs, with a focus on algorithms that break the traditional sorting barrier.

## Structure

```
├── src/           # Source code for SSSP implementations
├── tests/         # Test cases and benchmarks
├── data/          # Sample graphs and datasets
├── docs/          # Documentation
└── requirements.txt
```

## Requirements

- Python 3.11+
- See `requirements.txt` for additional dependencies

## Usage

Coming soon...

## References

- [Breaking the Sorting Barrier for Directed Single-Source Shortest Paths](https://arxiv.org/pdf/2504.17033v2)

## SSSP Benchmark: Dijkstra vs Frontier Reduction (Duan et al., 2025)

Ce projet compare l’algorithme classique de Dijkstra et l’algorithme "Frontier Reduction" du papier [Breaking the Sorting Barrier for Directed Single-Source Shortest Paths](https://arxiv.org/pdf/2504.17033v2).

### Algorithmes comparés

- **Dijkstra** : Complexité O((m+n) log n), optimal si on veut l’ordre des sommets par distance.
- **Frontier Reduction (papier)** : Complexité O(m log^{2/3} n), meilleure sur les grands graphes dirigés clairsemés, en réduisant la taille de la frontier à chaque étape via partitionnement et passes Bellman-Ford récursives.

### Ce que montrent les benchmarks

- **Temps d’exécution** et **nombre d’opérations** sont mesurés pour chaque algo.
- Sur de petits graphes, Dijkstra peut être plus rapide à cause de l’overhead Python et de la récursivité.
- Sur de grands graphes, la version du papier est censée surpasser Dijkstra (voir la théorie dans le papier).
- Les graphes générés et les chemins calculés sont visualisés dans `src/results/`.

### Limitations pratiques

- Les benchmarks Python ne reflètent pas toujours la théorie (overhead, gestion mémoire, etc.).
- Pour observer le gain théorique, il faut des graphes très grands (plusieurs milliers de sommets).
- L’implémentation suit la structure du papier, mais peut être optimisée en C++/Rust pour des benchmarks à grande échelle.

### Visualisation

- Les graphes et chemins calculés sont sauvegardés dans `src/results/`.
- Les métriques (temps, comparaisons, relaxations, heap ops) sont tracées avec seaborn.

---
