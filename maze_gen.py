#!/usr/bin/env python3
"""Research-grade maze solver visualizer with time-dependent doors."""

from __future__ import annotations

import heapq
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Generator, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pygame

# ---------------------------------------------------------------------------
# Maze primitives
# ---------------------------------------------------------------------------

DIRS = {
    "N": (0, -1),
    "S": (0, 1),
    "E": (1, 0),
    "W": (-1, 0),
}

OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def within_bounds(width: int, height: int, x: int, y: int) -> bool:
    return 0 <= x < width and 0 <= y < height


@dataclass
class Cell:
    x: int
    y: int
    walls: Dict[str, bool] = field(
        default_factory=lambda: {"N": True, "S": True, "E": True, "W": True}
    )
    visited: bool = False


class Maze:
    """Perfect grid maze generated via iterative DFS."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.cells: List[List[Cell]] = [
            [Cell(x, y) for x in range(width)] for y in range(height)
        ]

    def generate(self, rng_seed: Optional[int] = None, start_x: int = 0, start_y: int = 0):
        rng = random.Random(rng_seed)
        stack = [self.cells[start_y][start_x]]
        stack[0].visited = True

        while stack:
            cell = stack[-1]
            unvisited = [
                (direction, self.cells[ny][nx])
                for direction, (dx, dy) in DIRS.items()
                if within_bounds(self.width, self.height, nx := cell.x + dx, ny := cell.y + dy)
                and not self.cells[ny][nx].visited
            ]
            if not unvisited:
                stack.pop()
                continue
            direction, nxt = rng.choice(unvisited)
            cell.walls[direction] = False
            nxt.walls[OPPOSITE[direction]] = False
            nxt.visited = True
            stack.append(nxt)

    def cell_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        return [
            (nx, ny)
            for direction, (dx, dy) in DIRS.items()
            if not self.cells[y][x].walls[direction]
            and within_bounds(self.width, self.height, nx := x + dx, ny := y + dy)
        ]

    def corridor_edges(self) -> Iterable[Tuple[Tuple[int, int], Tuple[int, int]]]:
        seen = set()
        for y in range(self.height):
            for x in range(self.width):
                for nx, ny in self.cell_neighbors(x, y):
                    edge = tuple(sorted(((x, y), (nx, ny))))
                    if edge not in seen:
                        seen.add(edge)
                        yield edge

    def to_grid(self) -> List[List[int]]:
        grid_w = self.width * 2 + 1
        grid_h = self.height * 2 + 1
        grid = [[0 for _ in range(grid_w)] for _ in range(grid_h)]
        for y in range(self.height):
            for x in range(self.width):
                cell = self.cells[y][x]
                gx, gy = 2 * x + 1, 2 * y + 1
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


# ---------------------------------------------------------------------------
# Door scheduling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Door:
    a: Tuple[int, int]
    b: Tuple[int, int]
    open_duration: int
    closed_duration: int
    offset: int

    def cycle(self) -> int:
        return max(1, self.open_duration + self.closed_duration)

    def is_open(self, t: int) -> bool:
        return (t + self.offset) % self.cycle() < self.open_duration

    def key(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return tuple(sorted((self.a, self.b)))  # type: ignore[return-value]


@dataclass
class DoorConfig:
    count: int = 6
    open_range: Tuple[int, int] = (2, 5)
    closed_range: Tuple[int, int] = (1, 4)
    deterministic: bool = False
    deterministic_period: Tuple[int, int] = (3, 2)
    deterministic_offset_step: int = 1
    seed: Optional[int] = None


class DoorController:
    """Places and evaluates time-dependent doors along maze corridors."""

    def __init__(
        self,
        maze: Maze,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        config: Optional[DoorConfig] = None,
    ):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.config = config or DoorConfig()
        self.rng = random.Random(self.config.seed)
        allowed_edges = [
            edge
            for edge in maze.corridor_edges()
            if start not in edge and goal not in edge
        ]
        if self.config.deterministic:
            allowed_edges.sort()
        else:
            self.rng.shuffle(allowed_edges)
        self.doors: List[Door] = []
        for idx, edge in enumerate(allowed_edges[: self.config.count]):
            door = self._create_door(edge, idx)
            self.doors.append(door)
        self.lookup = {door.key(): door for door in self.doors}

    def _create_door(self, edge: Tuple[Tuple[int, int], Tuple[int, int]], idx: int) -> Door:
        if self.config.deterministic:
            open_duration, closed_duration = self.config.deterministic_period
            total = max(1, open_duration + closed_duration)
            offset = (idx * self.config.deterministic_offset_step) % total
        else:
            open_duration = self.rng.randint(*self.config.open_range)
            closed_duration = self.rng.randint(*self.config.closed_range)
            total = max(1, open_duration + closed_duration)
            offset = self.rng.randint(0, total - 1)
        return Door(edge[0], edge[1], open_duration, closed_duration, offset)

    def can_traverse(self, src: Tuple[int, int], dst: Tuple[int, int], t: int) -> bool:
        door = self.lookup.get(tuple(sorted((src, dst))))
        return True if door is None else door.is_open(t)

    def states_at(self, t: int) -> List[Tuple[Door, bool]]:
        return [(door, door.is_open(t)) for door in self.doors]


# ---------------------------------------------------------------------------
# Solver utilities
# ---------------------------------------------------------------------------


def reconstruct_path(parent: Dict[Tuple[int, int], Tuple[int, int]], start, goal):
    if goal != start and goal not in parent:
        return []
    cur = goal
    result = [cur]
    while cur != start:
        cur = parent[cur]
        result.append(cur)
    result.reverse()
    return result


def reconstruct_time_path(
    parent: Dict[Tuple[int, int, int], Tuple[int, int, int]],
    start: Tuple[int, int, int],
    goal: Tuple[int, int, int],
) -> List[Tuple[int, int]]:
    if goal != start and goal not in parent:
        return []
    cur = goal
    result = [cur]
    while cur != start:
        cur = parent[cur]
        result.append(cur)
    result.reverse()
    return [(x, y) for (x, y, _) in result]


def static_shortest_path(maze: Maze, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    q = deque([start])
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = {start}
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return reconstruct_path(parent, start, goal)
        for nx, ny in maze.cell_neighbors(x, y):
            if (nx, ny) in visited:
                continue
            visited.add((nx, ny))
            parent[(nx, ny)] = (x, y)
            q.append((nx, ny))
    return []


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def time_limit(maze: Maze) -> int:
    return max(maze.width * maze.height * 10, 200)


def time_neighbors(
    maze: Maze,
    state: Tuple[int, int, int],
    doors: DoorController,
    max_time: int,
) -> List[Tuple[int, int, int]]:
    x, y, t = state
    next_t = t + 1
    if next_t > max_time:
        return []
    moves = []
    for nx, ny in maze.cell_neighbors(x, y):
        if doors.can_traverse((x, y), (nx, ny), next_t):
            moves.append((nx, ny, next_t))
    moves.append((x, y, next_t))  # wait action
    return moves


@dataclass
class SolverSnapshot:
    visited: Set[Tuple[int, int]]
    frontier: Set[Tuple[int, int]]
    current: Optional[Tuple[int, int]]
    path: List[Tuple[int, int]]
    done: bool
    success: bool
    expanded: int
    elapsed: float
    time_step: int


@dataclass
class SolverVisualizer:
    name: str
    color: Tuple[int, int, int]
    generator: Iterator[SolverSnapshot]
    snapshot: SolverSnapshot = field(init=False)
    finished: bool = field(default=False, init=False)

    def __post_init__(self):
        self.snapshot = SolverSnapshot(set(), set(), None, [], False, False, 0, 0.0, 0)
        self.advance()

    def advance(self, steps: int = 1):
        if self.finished:
            return
        for _ in range(steps):
            try:
                self.snapshot = next(self.generator)
            except StopIteration:
                self.finished = True
                break


# ---------------------------------------------------------------------------
# Solver generators
# ---------------------------------------------------------------------------


def bfs_solver(maze: Maze, start: Tuple[int, int], goal: Tuple[int, int], doors: DoorController) -> Generator[SolverSnapshot, None, None]:
    start_state = (start[0], start[1], 0)
    q = deque([start_state])
    parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    visited_states = {start_state}
    visited_cells: Set[Tuple[int, int]] = {start}
    expanded = 0
    limit = time_limit(maze)
    start_time = time.perf_counter()

    while q:
        current = q.popleft()
        x, y, t = current
        expanded += 1
        visited_cells.add((x, y))
        frontier = {(cx, cy) for (cx, cy, _) in q}
        elapsed = time.perf_counter() - start_time
        success = (x, y) == goal
        path = reconstruct_time_path(parent, start_state, current) if success else []
        yield SolverSnapshot(set(visited_cells), frontier, (x, y), path, success, success, expanded, elapsed, t)
        if success:
            return
        if t >= limit:
            continue
        for nxt in time_neighbors(maze, current, doors, limit):
            if nxt in visited_states:
                continue
            visited_states.add(nxt)
            parent[nxt] = current
            q.append(nxt)

    elapsed = time.perf_counter() - start_time
    yield SolverSnapshot(set(visited_cells), set(), None, [], True, False, expanded, elapsed, limit)


def dfs_solver(maze: Maze, start: Tuple[int, int], goal: Tuple[int, int], doors: DoorController) -> Generator[SolverSnapshot, None, None]:
    limit = time_limit(maze)
    wait_budget = max(2, (maze.width + maze.height) // 2)
    stack: List[Tuple[int, int, int, int]] = [(start[0], start[1], 0, 0)]
    visited_cells: Set[Tuple[int, int]] = {start}
    expanded = 0
    start_time = time.perf_counter()

    while stack:
        x, y, t, wait_used = stack[-1]
        expanded += 1
        frontier = {(sx, sy) for sx, sy, _, _ in stack}
        path = [(sx, sy) for sx, sy, _, _ in stack]
        success = (x, y) == goal
        elapsed = time.perf_counter() - start_time
        yield SolverSnapshot(set(visited_cells), frontier, (x, y), path, success, success, expanded, elapsed, t)
        if success:
            return
        if t >= limit:
            stack.pop()
            continue

        moved = False
        for nx, ny in reversed(maze.cell_neighbors(x, y)):
            if (nx, ny) in visited_cells:
                continue
            if not doors.can_traverse((x, y), (nx, ny), t + 1):
                continue
            visited_cells.add((nx, ny))
            stack.append((nx, ny, t + 1, 0))
            moved = True
            break

        if moved:
            continue

        if wait_used < wait_budget:
            stack[-1] = (x, y, t + 1, wait_used + 1)
            continue

        stack.pop()

    elapsed = time.perf_counter() - start_time
    yield SolverSnapshot(set(visited_cells), set(), None, [], True, False, expanded, elapsed, limit)


def dijkstra_solver(maze: Maze, start: Tuple[int, int], goal: Tuple[int, int], doors: DoorController) -> Generator[SolverSnapshot, None, None]:
    start_state = (start[0], start[1], 0)
    open_heap: List[Tuple[float, Tuple[int, int, int]]] = [(0.0, start_state)]
    parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    g_score: Dict[Tuple[int, int, int], float] = {start_state: 0.0}
    visited_cells: Set[Tuple[int, int]] = set()
    expanded = 0
    limit = time_limit(maze)
    start_time = time.perf_counter()

    while open_heap:
        cost, state = heapq.heappop(open_heap)
        if cost > g_score.get(state, float("inf")):
            continue
        x, y, t = state
        visited_cells.add((x, y))
        expanded += 1
        frontier = {node[:2] for _, node in open_heap}
        success = (x, y) == goal
        elapsed = time.perf_counter() - start_time
        path = reconstruct_time_path(parent, start_state, state) if success else []
        yield SolverSnapshot(set(visited_cells), frontier, (x, y), path, success, success, expanded, elapsed, t)
        if success:
            return
        if t >= limit:
            continue
        for nxt in time_neighbors(maze, state, doors, limit):
            new_cost = cost + 1.0
            if new_cost >= g_score.get(nxt, float("inf")):
                continue
            g_score[nxt] = new_cost
            parent[nxt] = state
            heapq.heappush(open_heap, (new_cost, nxt))

    elapsed = time.perf_counter() - start_time
    yield SolverSnapshot(set(visited_cells), set(), None, [], True, False, expanded, elapsed, limit)


def astar_solver(maze: Maze, start: Tuple[int, int], goal: Tuple[int, int], doors: DoorController) -> Generator[SolverSnapshot, None, None]:
    start_state = (start[0], start[1], 0)
    open_heap: List[Tuple[float, Tuple[int, int, int]]] = [
        (manhattan(start, goal), start_state)
    ]
    parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    g_score: Dict[Tuple[int, int, int], float] = {start_state: 0.0}
    visited_cells: Set[Tuple[int, int]] = set()
    expanded = 0
    limit = time_limit(maze)
    start_time = time.perf_counter()

    while open_heap:
        f_cost, state = heapq.heappop(open_heap)
        g = g_score.get(state, float("inf"))
        if f_cost - manhattan(state[:2], goal) > g:
            continue
        x, y, t = state
        visited_cells.add((x, y))
        expanded += 1
        frontier = {node[:2] for _, node in open_heap}
        success = (x, y) == goal
        elapsed = time.perf_counter() - start_time
        path = reconstruct_time_path(parent, start_state, state) if success else []
        yield SolverSnapshot(set(visited_cells), frontier, (x, y), path, success, success, expanded, elapsed, t)
        if success:
            return
        if t >= limit:
            continue
        for nxt in time_neighbors(maze, state, doors, limit):
            tentative_g = g + 1.0
            if tentative_g >= g_score.get(nxt, float("inf")):
                continue
            g_score[nxt] = tentative_g
            parent[nxt] = state
            f = tentative_g + manhattan(nxt[:2], goal)
            heapq.heappush(open_heap, (f, nxt))

    elapsed = time.perf_counter() - start_time
    yield SolverSnapshot(set(visited_cells), set(), None, [], True, False, expanded, elapsed, limit)


def door_wait_solver(maze: Maze, start: Tuple[int, int], goal: Tuple[int, int], doors: DoorController) -> Generator[SolverSnapshot, None, None]:
    path = static_shortest_path(maze, start, goal)
    if not path:
        snapshot = SolverSnapshot(set(), set(), None, [], True, False, 0, 0.0, 0)
        yield snapshot
        return

    idx = 0
    t = 0
    expanded = 0
    visited: Set[Tuple[int, int]] = set()
    start_time = time.perf_counter()

    while True:
        current = path[idx]
        visited.add(current)
        frontier = {path[idx + 1]} if idx + 1 < len(path) else set()
        done = idx == len(path) - 1
        elapsed = time.perf_counter() - start_time
        yield SolverSnapshot(set(visited), frontier, current, path[: idx + 1], done, done, expanded, elapsed, t)
        if done:
            return
        next_cell = path[idx + 1]
        t += 1
        expanded += 1
        if doors.can_traverse(current, next_cell, t):
            idx += 1


ALGORITHMS = {
    "BFS": (bfs_solver, (66, 135, 245)),
    "DFS": (dfs_solver, (240, 144, 41)),
    "Dijkstra": (dijkstra_solver, (177, 110, 235)),
    "A*": (astar_solver, (62, 201, 115)),
    "Follower": (door_wait_solver, (255, 215, 0)),
}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def run_pygame_viewer(
    width_cells: int = 14,
    height_cells: int = 10,
    tile_size: int = 24,
    door_config: Optional[DoorConfig] = None,
):
    maze = Maze(width_cells, height_cells)
    maze.generate(rng_seed=1, start_x=0, start_y=0)
    start_cell = (0, 0)
    goal_cell = (width_cells - 1, height_cells - 1)
    doors = DoorController(maze, start_cell, goal_cell, door_config)

    grid = maze.to_grid()
    grid_rows = len(grid)
    grid_cols = len(grid[0])

    pygame.init()
    view_width = grid_cols * tile_size
    stats_height = 120

    num_algs = len(ALGORITHMS)
    num_cols = 3
    num_rows = (num_algs + num_cols - 1) // num_cols  # ceil division
    panel_height = grid_rows * tile_size + stats_height

    screen_width = view_width * num_cols
    screen_height = panel_height * num_rows

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Time-dependent Maze Solvers")
    font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()

    def cell_to_grid(x: int, y: int) -> Tuple[int, int]:
        return 2 * x + 1, 2 * y + 1

    solver_visualizers = [
        SolverVisualizer(name, color, solver_fn(maze, start_cell, goal_cell, doors))
        for name, (solver_fn, color) in ALGORITHMS.items()
    ]

    colors = {
        "wall": (20, 20, 20),
        "floor": (230, 230, 230),
        "start": (50, 200, 90),
        "goal": (210, 60, 60),
        "door_open": (80, 180, 250),
        "door_closed": (220, 90, 90),
    }

    def draw_alpha_rect(surface, color, rect, alpha):
        overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        overlay.fill((*color, alpha))
        surface.blit(overlay, rect.topleft)

    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False

        for solver in solver_visualizers:
            solver.advance()

        screen.fill((10, 10, 10))

        for idx, solver in enumerate(solver_visualizers):
            col = idx % num_cols
            row = idx // num_cols
            offset_x = col * view_width
            offset_y = row * panel_height

            # draw maze background
            for gy, grid_row in enumerate(grid):
                for gx, value in enumerate(grid_row):
                    color = colors["floor"] if value == 1 else colors["wall"]
                    rect = pygame.Rect(
                        offset_x + gx * tile_size,
                        offset_y + gy * tile_size,
                        tile_size,
                        tile_size,
                    )
                    pygame.draw.rect(screen, color, rect)

            snapshot = solver.snapshot

            # visited cells
            for cell in snapshot.visited:
                gx, gy = cell_to_grid(*cell)
                rect = pygame.Rect(
                    offset_x + gx * tile_size,
                    offset_y + gy * tile_size,
                    tile_size,
                    tile_size,
                )
                draw_alpha_rect(screen, solver.color, rect, 60)

            # frontier cells
            for cell in snapshot.frontier:
                gx, gy = cell_to_grid(*cell)
                rect = pygame.Rect(
                    offset_x + gx * tile_size,
                    offset_y + gy * tile_size,
                    tile_size,
                    tile_size,
                )
                draw_alpha_rect(screen, solver.color, rect, 110)

            # path cells
            for cell in snapshot.path:
                gx, gy = cell_to_grid(*cell)
                rect = pygame.Rect(
                    offset_x + gx * tile_size,
                    offset_y + gy * tile_size,
                    tile_size,
                    tile_size,
                )
                draw_alpha_rect(screen, solver.color, rect, 180)

            # current cell
            if snapshot.current:
                gx, gy = cell_to_grid(*snapshot.current)
                rect = pygame.Rect(
                    offset_x + gx * tile_size,
                    offset_y + gy * tile_size,
                    tile_size,
                    tile_size,
                )
                draw_alpha_rect(screen, solver.color, rect, 230)

            # start and goal
            sx, sy = cell_to_grid(*start_cell)
            gx, gy = cell_to_grid(*goal_cell)
            pygame.draw.rect(
                screen,
                colors["start"],
                pygame.Rect(offset_x + sx * tile_size, offset_y + sy * tile_size, tile_size, tile_size),
            )
            pygame.draw.rect(
                screen,
                colors["goal"],
                pygame.Rect(offset_x + gx * tile_size, offset_y + gy * tile_size, tile_size, tile_size),
            )

            # doors
            for door, is_open in doors.states_at(snapshot.time_step):
                agx, agy = cell_to_grid(*door.a)
                bgx, bgy = cell_to_grid(*door.b)
                dgx = (agx + bgx) // 2
                dgy = (agy + bgy) // 2
                rect = pygame.Rect(
                    offset_x + dgx * tile_size,
                    offset_y + dgy * tile_size,
                    tile_size,
                    tile_size,
                )
                draw_alpha_rect(
                    screen,
                    colors["door_open" if is_open else "door_closed"],
                    rect,
                    160 if is_open else 230,
                )

            # stats area below each panel
            stats_rect = pygame.Rect(
                offset_x,
                offset_y + grid_rows * tile_size,
                view_width,
                stats_height,
            )
            pygame.draw.rect(screen, (25, 25, 25), stats_rect)

            lines = [
                f"{solver.name}",
                f"elapsed: {snapshot.elapsed:.2f}s",
                f"success: {snapshot.success}",
                f"expanded: {snapshot.expanded}",
                f"time step: {snapshot.time_step}",
                f"path length: {len(snapshot.path) if snapshot.path else '-'}",
            ]
            for i, text in enumerate(lines):
                surface = font.render(text, True, (235, 235, 235))
                screen.blit(
                    surface,
                    (
                        offset_x + 8,
                        offset_y + grid_rows * tile_size + 8 + i * 18,
                    ),
                )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    door_cfg = DoorConfig(deterministic=False, seed=42)
    run_pygame_viewer(14, 10, 24, door_cfg)
