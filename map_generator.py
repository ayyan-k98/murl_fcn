"""
Map Generator for Coverage Environment

Generates diverse map types for curriculum learning.
"""

import random
from typing import Tuple, Set
import numpy as np
import networkx as nx


class MapGenerator:
    """
    Generates diverse map types for curriculum learning.
    All maps are represented as NetworkX graphs.
    """

    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size

    def generate(self, map_type: str) -> Tuple[nx.Graph, Set[Tuple[int, int]]]:
        """
        Generate a map of specified type.

        Returns:
            graph: NetworkX graph with nodes and edges
            obstacles: Set of obstacle positions
        """
        if map_type == "empty":
            return self._generate_empty()
        elif map_type == "random":
            return self._generate_random()
        elif map_type == "room":
            return self._generate_room()
        elif map_type == "corridor":
            return self._generate_corridor()
        elif map_type == "cave":
            return self._generate_cave()
        elif map_type == "lshape":
            return self._generate_lshape()
        else:
            raise ValueError(f"Unknown map type: {map_type}")

    def _generate_empty(self) -> Tuple[nx.Graph, Set[Tuple[int, int]]]:
        """Empty map (no obstacles)."""
        G = nx.grid_2d_graph(self.grid_size, self.grid_size)
        obstacles = set()

        # Add node attributes
        for node in G.nodes():
            G.nodes[node]['type'] = 'free'
            G.nodes[node]['pc'] = 0.0  # Coverage

        return G, obstacles

    def _generate_random(self, obstacle_density: float = 0.15) -> Tuple[nx.Graph, Set[Tuple[int, int]]]:
        """Random scattered obstacles."""
        G = nx.grid_2d_graph(self.grid_size, self.grid_size)
        obstacles = set()

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if random.random() < obstacle_density:
                    obstacles.add((x, y))

        # Set attributes
        for node in list(G.nodes()):
            if node in obstacles:
                G.nodes[node]['type'] = 'obstacle'
                G.nodes[node]['pc'] = 0.0
            else:
                G.nodes[node]['type'] = 'free'
                G.nodes[node]['pc'] = 0.0

        return G, obstacles

    def _generate_room(self) -> Tuple[nx.Graph, Set[Tuple[int, int]]]:
        """Room with walls and doors."""
        G = nx.grid_2d_graph(self.grid_size, self.grid_size)
        obstacles = set()

        # Create 2-4 rooms
        num_rooms = random.randint(2, 4)

        if num_rooms == 2:
            # Vertical wall with door
            wall_x = self.grid_size // 2
            door_y = random.randint(self.grid_size // 4, 3 * self.grid_size // 4)
            for y in range(self.grid_size):
                if y != door_y and y != door_y + 1:
                    obstacles.add((wall_x, y))
        else:
            # 4 rooms (cross pattern)
            wall_x = self.grid_size // 2
            wall_y = self.grid_size // 2

            # Vertical wall with 2 doors
            door1_y = random.randint(2, wall_y - 2)
            door2_y = random.randint(wall_y + 2, self.grid_size - 3)
            for y in range(self.grid_size):
                if y not in [door1_y, door2_y]:
                    obstacles.add((wall_x, y))

            # Horizontal wall with 2 doors
            door1_x = random.randint(2, wall_x - 2)
            door2_x = random.randint(wall_x + 2, self.grid_size - 3)
            for x in range(self.grid_size):
                if x not in [door1_x, door2_x] and (x, wall_y) not in obstacles:
                    obstacles.add((x, wall_y))

        # Set attributes
        for node in G.nodes():
            if node in obstacles:
                G.nodes[node]['type'] = 'obstacle'
            else:
                G.nodes[node]['type'] = 'free'
            G.nodes[node]['pc'] = 0.0

        return G, obstacles

    def _generate_corridor(self) -> Tuple[nx.Graph, Set[Tuple[int, int]]]:
        """Narrow corridors."""
        G = nx.grid_2d_graph(self.grid_size, self.grid_size)
        obstacles = set()

        # Create maze-like corridors
        corridor_width = 2

        # Horizontal corridors every 5 rows
        for i in range(0, self.grid_size, 5):
            for x in range(self.grid_size):
                # Block rows above corridor
                if i > 0:
                    obstacles.add((x, i - 1))

        # Vertical connections every 5 columns
        for i in range(2, self.grid_size, 5):
            for y in range(self.grid_size):
                if (i, y) in obstacles:
                    obstacles.remove((i, y))

        # Set attributes
        for node in G.nodes():
            if node in obstacles:
                G.nodes[node]['type'] = 'obstacle'
            else:
                G.nodes[node]['type'] = 'free'
            G.nodes[node]['pc'] = 0.0

        return G, obstacles

    def _generate_cave(self) -> Tuple[nx.Graph, Set[Tuple[int, int]]]:
        """Cave-like structure using cellular automata."""
        G = nx.grid_2d_graph(self.grid_size, self.grid_size)

        # Initialize with random noise
        grid = np.random.random((self.grid_size, self.grid_size)) < 0.45

        # Cellular automata smoothing (3 iterations)
        for _ in range(3):
            new_grid = grid.copy()
            for x in range(1, self.grid_size - 1):
                for y in range(1, self.grid_size - 1):
                    # Count neighbors
                    neighbors = grid[x-1:x+2, y-1:y+2].sum() - grid[x, y]
                    if neighbors > 4:
                        new_grid[x, y] = True
                    elif neighbors < 4:
                        new_grid[x, y] = False
            grid = new_grid

        obstacles = set()
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if grid[x, y]:
                    obstacles.add((x, y))

        # Set attributes
        for node in G.nodes():
            if node in obstacles:
                G.nodes[node]['type'] = 'obstacle'
            else:
                G.nodes[node]['type'] = 'free'
            G.nodes[node]['pc'] = 0.0

        return G, obstacles

    def _generate_lshape(self) -> Tuple[nx.Graph, Set[Tuple[int, int]]]:
        """L-shaped free space."""
        G = nx.grid_2d_graph(self.grid_size, self.grid_size)
        obstacles = set()

        # Block one corner randomly
        corner = random.choice(['NE', 'SE', 'SW', 'NW'])
        mid = self.grid_size // 2

        if corner == 'NE':
            for x in range(mid, self.grid_size):
                for y in range(0, mid):
                    obstacles.add((x, y))
        elif corner == 'SE':
            for x in range(mid, self.grid_size):
                for y in range(mid, self.grid_size):
                    obstacles.add((x, y))
        elif corner == 'SW':
            for x in range(0, mid):
                for y in range(mid, self.grid_size):
                    obstacles.add((x, y))
        else:  # NW
            for x in range(0, mid):
                for y in range(0, mid):
                    obstacles.add((x, y))

        # Set attributes
        for node in G.nodes():
            if node in obstacles:
                G.nodes[node]['type'] = 'obstacle'
            else:
                G.nodes[node]['type'] = 'free'
            G.nodes[node]['pc'] = 0.0

        return G, obstacles


if __name__ == "__main__":
    # Test map generator
    gen = MapGenerator(grid_size=20)

    for map_type in ["empty", "random", "room", "corridor", "cave", "lshape"]:
        graph, obstacles = gen.generate(map_type)
        print(f"✓ {map_type:10s}: {len(graph.nodes()):3d} nodes, {len(obstacles):3d} obstacles")
