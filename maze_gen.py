#!/usr/bin/env python3
"""
maze_gen.py + Pygame viewer

- Generates a perfect grid maze using DFS (recursive backtracking, iterative).
- Displays the maze in a Pygame window.


Make sure Pygame is installed:
    python3 -m pip install pygame
"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import pygame


# Maze data structures and generation

DIRS = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}

OPPOSITE = {
    "N": "S",
    "S": "N",
    "E": "W",
    "W": "E",
}


@dataclass
class Cell:
    x: int
    y: int
    walls: Dict[str, bool] = field(
        default_factory=lambda: {"N": True, "S": True, "E": True, "W": True}
    )
    visited: bool = False


class Maze:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # 2D array [y][x]
        self.cells: List[List[Cell]] = [
            [Cell(x, y) for x in range(width)] for y in range(height)
        ]

        # order cells were first visited
        self.visit_order: List[Tuple[int, int]] = []

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, cell: Cell):
        """Return list of (direction, neighbor_cell)."""
        result = []
        for d, (dx, dy) in DIRS.items():
            nx, ny = cell.x + dx, cell.y + dy
            if self.in_bounds(nx, ny):
                result.append((d, self.cells[ny][nx]))
        return result

    def generate(self, rng_seed: Optional[int] = None, start_x: int = 0, start_y: int = 0):
        """Iterative DFS maze generation."""
        if rng_seed is not None:
            random.seed(rng_seed)

        start = self.cells[start_y][start_x]
        start.visited = True
        self.visit_order.append((start.x, start.y))
        stack = [start]

        while stack:
            current = stack[-1]
            unvisited = [(d, n) for d, n in self.neighbors(current) if not n.visited]

            if not unvisited:
                stack.pop()
                continue

            direction, nxt = random.choice(unvisited)

            # knock down walls between current and next
            current.walls[direction] = False
            nxt.walls[OPPOSITE[direction]] = False

            nxt.visited = True
            self.visit_order.append((nxt.x, nxt.y))
            stack.append(nxt)

    def to_grid(self) -> List[List[int]]:
        """
        Convert cells to a 2D grid of 0/1 where:
          0 = wall, 1 = floor.
        Grid size = (2*height + 1) rows by (2*width + 1) columns.
        """
        grid_w = self.width * 2 + 1
        grid_h = self.height * 2 + 1

        grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]

        for y in range(self.height):
            for x in range(self.width):
                cell = self.cells[y][x]
                gx, gy = 2 * x + 1, 2 * y + 1

                # cell center is floor
                grid[gy][gx] = 1

                if not cell.walls["N"]:
                    grid[gy - 1][gx] = 1
                if not cell.walls["S"]:
                    grid[gy + 1][gx] = 1
                if not cell.walls["W"]:
                    grid[gy][gx - 1] = 1
                if not cell.walls["E"]:
                    grid[gy][gx + 1] = 1

        return grid


# Pygame rendering

def run_pygame_viewer(width_cells: int = 10, height_cells: int = 10, tile_size: int = 24):
    """
    Generate a maze and display it with Pygame.
    width_cells, height_cells are in *maze cells*, not pixels.
    tile_size is pixel size of each grid square.
    """

    #generate maze 
    maze = Maze(width_cells, height_cells)
    maze.generate(rng_seed=42, start_x=0, start_y=0)
    grid = maze.to_grid()

    grid_rows = len(grid)
    grid_cols = len(grid[0])

    print(f"Grid size: {grid_cols} x {grid_rows}")

    #pygame setup
    pygame.init()
    screen_width = grid_cols * tile_size
    screen_height = grid_rows * tile_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("DFS Maze")

    clock = pygame.time.Clock()

    WALL_COLOR = (20, 20, 20)
    FLOOR_COLOR = (230, 230, 230)
    START_COLOR = (50, 200, 50)
    GOAL_COLOR = (200, 50, 50)

    #choose start/goal in cell coordinates
    start_cell = (0, 0)
    goal_cell = (width_cells - 1, height_cells - 1)

    #helper: convert cell coords -> grid coords (center)
    def cell_to_grid(x: int, y: int) -> Tuple[int, int]:
        return 2 * x + 1, 2 * y + 1

    start_gx, start_gy = cell_to_grid(*start_cell)
    goal_gx, goal_gy = cell_to_grid(*goal_cell)

    running = True
    while running:
        #events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        #draw
        screen.fill(WALL_COLOR)

        for gy, row in enumerate(grid):
            for gx, val in enumerate(row):
                if val == 1:
                    color = FLOOR_COLOR
                    # override for start/goal
                    if (gx, gy) == (start_gx, start_gy):
                        color = START_COLOR
                    elif (gx, gy) == (goal_gx, goal_gy):
                        color = GOAL_COLOR
                else:
                    color = WALL_COLOR

                rect = pygame.Rect(gx * tile_size, gy * tile_size, tile_size, tile_size)
                pygame.draw.rect(screen, color, rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# Main entry point

if __name__ == "__main__":
    #Choose maze size here (in cells)
    WIDTH_CELLS = 10
    HEIGHT_CELLS = 10

    #Each grid square's pixel size
    TILE_SIZE = 24

    run_pygame_viewer(WIDTH_CELLS, HEIGHT_CELLS, TILE_SIZE)