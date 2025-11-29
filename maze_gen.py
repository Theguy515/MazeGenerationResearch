#Fall 2025 Algorithm Analysis Project
#Made by Daniel Wong and Jordan Andrews
#Maze generation and solvers
#Runs are done multiple times with different variables on the same mazes
#A run is done for each algorithm in the following 3 categories:
#On a normal maze with no doors and no monster
#A run with time dependent "doors"
#And lastly a run with a monster that follows the shortest path, essentially blocking the main path at points

#To run this code, open terminal, follow directories to where the files are then run "python3 maze_gen.py"
#To save multiple runs to a csv file, run "python3 maze_gen.py --mode cli --scenario all --runs 10 --csv-output results.csv"
#You can increase or decrease the number of runs as you wish in the command
 
from __future__ import annotations

import argparse
import heapq
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Generator, Iterable, Iterator, List, Optional, Sequence, Set, Tuple



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
#Grid maze generated with DFS generation algorithm
#This code ensures theres always one path from the start to the finish
#Then, we punch extra small links through the walls to create two paths from start to finish

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.cells: List[List[Cell]] = [
            [Cell(x, y) for x in range(width)] for y in range(height)
        ]

    def generate(
        self,
        rng_seed: Optional[int] = None,
        start_x: int = 0,
        start_y: int = 0,
        extra_links: int = 0,
    ):
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

        #Delete walls to create more than one path to the end
        for _ in range(extra_links):
            y = rng.randrange(self.height)
            x = rng.randrange(self.width)
            dirs = list(DIRS.items())
            rng.shuffle(dirs)
            for direction, (dx, dy) in dirs:
                nx, ny = x + dx, y + dy
                if not within_bounds(self.width, self.height, nx, ny):
                    continue
                if self.cells[y][x].walls[direction]:
                    self.cells[y][x].walls[direction] = False
                    self.cells[ny][nx].walls[OPPOSITE[direction]] = False
                    break

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


#Time intervals for doors

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
        return tuple(sorted((self.a, self.b)))


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
    #Places time dependent doors along corridors
    #Each door has an open/closed cycle, at time (t) the edge can only be traversed if a door is at open state
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


class NullDoorController:
    def can_traverse(self, src: Tuple[int, int], dst: Tuple[int, int], t: int) -> bool:
        return True

    def states_at(self, t: int) -> List[Tuple[Door, bool]]:
        return []


#Scheduling for monsters along shortest path

@dataclass
class MonsterConfig:
    count: int = 1
    step_interval: int = 1
    seed: Optional[int] = None


class MonsterController:

    #Controls the patrolling monster
    #Monsters walk back and forth along the static shortest path from start to goal
    #Monsters essentially behave as guard patrolling the main corrridor while the solvers run
    def __init__(
        self,
        maze: Maze,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        config: Optional[MonsterConfig] = None,
    ):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.config = config or MonsterConfig()
        self.rng = random.Random(self.config.seed)
        base_path = static_shortest_path(maze, start, goal)
        if len(base_path) < 2:
            base_path = [(0, 0)]
        patrol = base_path + list(reversed(base_path[1:-1]))  # back and forth loop
        self.paths: List[List[Tuple[int, int]]] = []
        for _ in range(self.config.count):
            self.paths.append(list(patrol))

    def positions_at(self, t: int) -> List[Tuple[int, int]]:
        positions = []
        if self.config.step_interval <= 0:
            return positions
        for path in self.paths:
            if not path:
                continue
            idx = (t // self.config.step_interval) % len(path)
            positions.append(path[idx])
        return positions

    def occupies(self, cell: Tuple[int, int], t: int) -> bool:
        return cell in self.positions_at(t)


class NullMonsterController:

    def positions_at(self, t: int) -> List[Tuple[int, int]]:
        return []

    def occupies(self, cell: Tuple[int, int], t: int) -> bool:
        return False


# Solver utilities


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


def static_shortest_path_forbidden_edge(
    maze: Maze,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    forbidden_edge: Tuple[Tuple[int, int], Tuple[int, int]],
) -> List[Tuple[int, int]]:
    blocked = {tuple(sorted(forbidden_edge))}
    q = deque([start])
    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
    visited = {start}
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return reconstruct_path(parent, start, goal)
        for nx, ny in maze.cell_neighbors(x, y):
            edge = tuple(sorted(((x, y), (nx, ny))))
            if edge in blocked:
                continue
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
    doors: Optional[DoorController],
    max_time: int,
    monsters: Optional["MonsterController"] = None,
) -> List[Tuple[int, int, int]]:
    
    #Given our x,y,t we return all safe states at time t+1

    #This is how we enforce door timing and monster collisions so all algorithms-
    # - besides the follower algorithm share the same movement rules

    x, y, t = state
    next_t = t + 1
    if next_t > max_time:
        return []
    monster_pos_t = set(monsters.positions_at(t)) if monsters else set()
    monster_pos_next = set(monsters.positions_at(next_t)) if monsters else set()
    moves = []
    for nx, ny in maze.cell_neighbors(x, y):
        if doors and not doors.can_traverse((x, y), (nx, ny), next_t):
            continue
        if monsters:
            if (nx, ny) in monster_pos_next:
                continue  # monster occupies destination
            if (x, y) in monster_pos_next:
                continue  # monster moves into our current cell
            if (nx, ny) in monster_pos_t and (x, y) in monster_pos_next:
                continue  # swap positions with monster
        moves.append((nx, ny, next_t))
    if not monsters or (x, y) not in monster_pos_next:
        moves.append((x, y, next_t))  # wait action if monster not moving onto us
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


#Solver generators


def bfs_solver(maze: Maze, start: Tuple[int, int], goal: Tuple[int, int], doors: Optional[DoorController], monsters: Optional["MonsterController"] = None) -> Generator[SolverSnapshot, None, None]:
    
    #BFS on the time-expanded maze (states are x,y, and t)

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
        for nxt in time_neighbors(maze, current, doors, limit, monsters):
            if nxt in visited_states:
                continue
            visited_states.add(nxt)
            parent[nxt] = current
            q.append(nxt)

    elapsed = time.perf_counter() - start_time
    yield SolverSnapshot(set(visited_cells), set(), None, [], True, False, expanded, elapsed, limit)


def dfs_solver(maze: Maze, start: Tuple[int, int], goal: Tuple[int, int], doors: Optional[DoorController], monsters: Optional["MonsterController"] = None) -> Generator[SolverSnapshot, None, None]:

    #DFS that explores one path in the time-expanded maze with limited to no waiting
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
        monster_pos_t = set(monsters.positions_at(t)) if monsters else set()
        monster_pos_next = set(monsters.positions_at(t + 1)) if monsters else set()

        for nx, ny in reversed(maze.cell_neighbors(x, y)):
            if (nx, ny) in visited_cells:
                continue
            if doors and not doors.can_traverse((x, y), (nx, ny), t + 1):
                continue
            if monsters and (
                (nx, ny) in monster_pos_next
                or (x, y) in monster_pos_next
                or ((nx, ny) in monster_pos_t and (x, y) in monster_pos_next)
            ):
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


def dijkstra_solver(maze: Maze, start: Tuple[int, int], goal: Tuple[int, int], doors: Optional[DoorController], monsters: Optional["MonsterController"] = None) -> Generator[SolverSnapshot, None, None]:

    #Dijkstra's algorith, on (x,y,t) 

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
        for nxt in time_neighbors(maze, state, doors, limit, monsters):
            new_cost = cost + 1.0
            if new_cost >= g_score.get(nxt, float("inf")):
                continue
            g_score[nxt] = new_cost
            parent[nxt] = state
            heapq.heappush(open_heap, (new_cost, nxt))

    elapsed = time.perf_counter() - start_time
    yield SolverSnapshot(set(visited_cells), set(), None, [], True, False, expanded, elapsed, limit)


def astar_solver(maze: Maze, start: Tuple[int, int], goal: Tuple[int, int], doors: Optional[DoorController], monsters: Optional["MonsterController"] = None) -> Generator[SolverSnapshot, None, None]:

    #A* search on (x,y,t) using Manhattan distance on (x,y) as the heuristic

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
        for nxt in time_neighbors(maze, state, doors, limit, monsters):
            tentative_g = g + 1.0
            if tentative_g >= g_score.get(nxt, float("inf")):
                continue
            g_score[nxt] = tentative_g
            parent[nxt] = state
            f = tentative_g + manhattan(nxt[:2], goal)
            heapq.heappush(open_heap, (f, nxt))

    elapsed = time.perf_counter() - start_time
    yield SolverSnapshot(set(visited_cells), set(), None, [], True, False, expanded, elapsed, limit)


def door_wait_solver(maze: Maze, start: Tuple[int, int], goal: Tuple[int, int], doors: Optional[DoorController], monsters: Optional["MonsterController"] = None) -> Generator[SolverSnapshot, None, None]:
    
    #A follower baseline, precomputes the shortest path possible, then follows it
    #If a door is closed, it waits, if the monster is in front of it and traversing towards the goal, it waits behind
    #Easily fails monster cases due to no redirecting/detours allowed
    path_compute_start = time.perf_counter()
    path = static_shortest_path(maze, start, goal)
    path_compute_elapsed = time.perf_counter() - path_compute_start
    if not path:
        snapshot = SolverSnapshot(set(), set(), None, [], True, False, 0, path_compute_elapsed, 0)
        yield snapshot
        return

    idx = 0
    t = 0
    expanded = 0
    visited: Set[Tuple[int, int]] = set()
    start_time = time.perf_counter() - path_compute_elapsed  # include path compute time

    while True:
        monster_pos_t = set(monsters.positions_at(t)) if monsters else set()
        monster_pos_next = set(monsters.positions_at(t + 1)) if monsters else set()

        if monsters and t > 0 and path[idx] in monster_pos_t:
            elapsed = time.perf_counter() - start_time
            yield SolverSnapshot(set(visited), set(), path[idx], path[: idx + 1], True, False, expanded, elapsed, t)
            return
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
        if doors and not doors.can_traverse(current, next_cell, t):
            continue
        if monsters:
            if next_cell in monster_pos_next:
                continue
            if current in monster_pos_next:
                elapsed = time.perf_counter() - start_time
                yield SolverSnapshot(set(visited), set(), current, path[: idx + 1], True, False, expanded, elapsed, t)
                return
            if (next_cell in monster_pos_t) and (current in monster_pos_next):
                elapsed = time.perf_counter() - start_time
                yield SolverSnapshot(set(visited), set(), current, path[: idx + 1], True, False, expanded, elapsed, t)
                return
        idx += 1


ALGORITHMS = {
    "BFS": (bfs_solver, (66, 135, 245)),
    "DFS": (dfs_solver, (240, 144, 41)),
    "Dijkstra": (dijkstra_solver, (177, 110, 235)),
    "A*": (astar_solver, (62, 201, 115)),
    "Follower": (door_wait_solver, (255, 215, 0)),
}


#CLI + visualization

def carve_random_link(maze: Maze, rng: random.Random) -> None:
    for _ in range(20):
        y = rng.randrange(maze.height)
        x = rng.randrange(maze.width)
        dirs = list(DIRS.items())
        rng.shuffle(dirs)
        for direction, (dx, dy) in dirs:
            nx, ny = x + dx, y + dy
            if not within_bounds(maze.width, maze.height, nx, ny):
                continue
            if maze.cells[y][x].walls[direction]:
                maze.cells[y][x].walls[direction] = False
                maze.cells[ny][nx].walls[OPPOSITE[direction]] = False
                return


def ensure_two_distinct_paths(
        
    #Carve out extra links until there exists at least a second distinct path
    #We compute a shortest path then try to block edges, if there exists a path still we keep it -
    # -if a path doesnt exist we carve another link out then retry
    maze: Maze,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    rng: random.Random,
    max_attempts: int = 30,
) -> None:
    for _ in range(max_attempts):
        primary = static_shortest_path(maze, start, goal)
        if len(primary) < 2:
            carve_random_link(maze, rng)
            continue
        alt = None
        for i in range(len(primary) - 1):
            edge = tuple(sorted((primary[i], primary[i + 1])))
            candidate = static_shortest_path_forbidden_edge(maze, start, goal, edge)
            if candidate and len(candidate) != len(primary):
                alt = candidate
                break
        if alt:
            return
        carve_random_link(maze, rng)


def make_environment(
        
    #Build the test environment

    width_cells: int,
    height_cells: int,
    door_config: Optional[DoorConfig],
    rng_seed: Optional[int],
    extra_links: int,
    enable_doors: bool,
    enable_monsters: bool,
    monster_config: Optional["MonsterConfig"],
):
    rng = random.Random(rng_seed)
    maze = Maze(width_cells, height_cells)
    maze.generate(rng_seed=rng_seed, start_x=0, start_y=0, extra_links=extra_links)
    start_cell = (0, 0)
    goal_cell = (width_cells - 1, height_cells - 1)
    ensure_two_distinct_paths(maze, start_cell, goal_cell, rng)
    doors: Optional[DoorController] = DoorController(maze, start_cell, goal_cell, door_config) if enable_doors else NullDoorController()
    monsters: Optional["MonsterController"] = MonsterController(maze, start_cell, goal_cell, monster_config) if enable_monsters else NullMonsterController()
    return maze, start_cell, goal_cell, doors, monsters


def build_solver_factories(maze: Maze, start_cell: Tuple[int, int], goal_cell: Tuple[int, int], doors: Optional[DoorController], monsters: Optional["MonsterController"] = None):
    factories: List[Callable[[], SolverVisualizer]] = []
    for name, (solver_fn, color) in ALGORITHMS.items():
        def factory(fn=solver_fn, alg_name=name, alg_color=color):
            return SolverVisualizer(alg_name, alg_color, fn(maze, start_cell, goal_cell, doors, monsters))
        factories.append(factory)
    return factories


def run_visual_mode(args):
    from visualizer import MazeVisualizer

    base_seed = args.base_seed if args.base_seed is not None else (args.seed if args.seed is not None else random.randint(0, 1_000_000_000))
    door_seed = args.door_seed if args.door_seed is not None else base_seed
    monster_seed = args.monster_seed if args.monster_seed is not None else base_seed

    scenarios = []
    if args.scenario in ("all", "static"):
        scenarios.append(("static", False, False))
    if args.scenario in ("all", "doors"):
        scenarios.append(("doors", True, False))
    if args.scenario in ("all", "monsters"):
        scenarios.append(("monsters", False, True))

    for scenario_name, enable_doors, enable_monsters in scenarios:
        print(f"Launching visualizer for scenario: {scenario_name}")
        maze, start_cell, goal_cell, doors, monsters = make_environment(
            args.width,
            args.height,
            build_door_config(args, door_seed),
            base_seed,
            args.extra_links,
            enable_doors,
            enable_monsters,
            build_monster_config(args, monster_seed),
        )
        factories = build_solver_factories(maze, start_cell, goal_cell, doors, monsters)
        viewer = MazeVisualizer(
            maze=maze,
            doors=doors,
            solver_factories=factories,
            start_cell=start_cell,
            goal_cell=goal_cell,
        monsters=monsters,
        tile_size=args.tile_size,
        stats_height=160,
        max_cols=3,
        title_suffix=f" - {scenario_name}",
    )
        viewer.run()


def consume_solver(generator: Iterator[SolverSnapshot]) -> SolverSnapshot:
    last = None
    for snapshot in generator:
        last = snapshot
        if snapshot.done:
            break
    return last if last is not None else SolverSnapshot(set(), set(), None, [], True, False, 0, 0.0, 0)


def run_cli_mode(args):
    scenarios = []
    if args.scenario in ("all", "static"):
        scenarios.append(("static", False, False))
    if args.scenario in ("all", "doors"):
        scenarios.append(("doors", True, False))
    if args.scenario in ("all", "monsters"):
        scenarios.append(("monsters", False, True))

    rows = []
    for run_idx in range(args.runs):
        base_seed = args.base_seed if args.base_seed is not None else (args.seed if args.seed is not None else random.randint(0, 1_000_000_000))
        door_seed = args.door_seed if args.door_seed is not None else base_seed
        monster_seed = args.monster_seed if args.monster_seed is not None else base_seed
        seed_desc = base_seed if args.base_seed or args.seed is not None else f"random({base_seed})"

        for name, enable_doors, enable_monsters in scenarios:
            maze, start_cell, goal_cell, doors, monsters = make_environment(
                args.width,
                args.height,
                build_door_config(args, door_seed),
                base_seed,
                args.extra_links,
                enable_doors,
                enable_monsters,
                build_monster_config(args, monster_seed),
            )
            print(f"\nRun {run_idx + 1}/{args.runs} Scenario: {name} | maze {args.width}x{args.height} | base_seed: {seed_desc} | door_seed: {door_seed} | monster_seed: {monster_seed} | extra_links={args.extra_links}")
            if enable_doors and isinstance(doors, DoorController):
                print(f"Doors: {doors.config.count} (deterministic={doors.config.deterministic})")
            if enable_monsters and isinstance(monsters, MonsterController):
                print(f"Monsters: count={monsters.config.count} step_interval={monsters.config.step_interval}")
            for alg_name, (solver_fn, _) in ALGORITHMS.items():
                snapshot = consume_solver(solver_fn(maze, start_cell, goal_cell, doors, monsters))
                success = "yes" if snapshot.success else "no"
                path_length = len(snapshot.path) if snapshot.path else "-"
                print(f"[{alg_name}] success={success} elapsed={snapshot.elapsed:.3f}s expanded={snapshot.expanded} steps={snapshot.time_step} path_len={path_length}")
                rows.append({
                    "run": run_idx + 1,
                    "scenario": name,
                    "base_seed": base_seed,
                    "door_seed": door_seed if enable_doors else "",
                    "monster_seed": monster_seed if enable_monsters else "",
                    "algorithm": alg_name,
                    "success": snapshot.success,
                    "elapsed": f"{snapshot.elapsed:.6f}",
                    "expanded": snapshot.expanded,
                    "time_steps": snapshot.time_step,
                    "path_length": path_length,
                })

    if args.csv_output:
        import csv

        fieldnames = [
            "run",
            "scenario",
            "base_seed",
            "door_seed",
            "monster_seed",
            "algorithm",
            "success",
            "elapsed",
            "expanded",
            "time_steps",
            "path_length",
        ]
        with open(args.csv_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.csv_output}")


def prompt_for_mode():
    response = input("Run visualizer? (y/n): ").strip().lower()
    return "visual" if response.startswith("y") else "cli"


def build_door_config(args, door_seed: Optional[int]) -> DoorConfig:
    return DoorConfig(
        deterministic=args.deterministic,
        seed=door_seed,
        count=args.door_count,
    )


def build_monster_config(args, monster_seed: Optional[int]) -> MonsterConfig:
    return MonsterConfig(
        count=args.monster_count,
        step_interval=args.monster_interval,
        seed=monster_seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Maze generator/solver with optional visualizer.")
    parser.add_argument("--mode", choices=["visual", "cli"], help="Choose 'visual' for pygame viewer or 'cli' for text metrics.")
    parser.add_argument("--scenario", choices=["all", "static", "doors", "monsters"], default="all", help="Which constraints to run.")
    parser.add_argument("--width", type=int, default=14, help="Maze width in cells.")
    parser.add_argument("--height", type=int, default=10, help="Maze height in cells.")
    parser.add_argument("--extra-links", type=int, default=6, help="Additional corridors to carve after generation to create multiple paths.")
    parser.add_argument("--tile-size", type=int, default=24, help="Base tile size for visual mode; auto-scales to fit the screen.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for maze and door generation (default: random).")
    parser.add_argument("--base-seed", type=int, default=None, help="Explicit base seed for maze generation.")
    parser.add_argument("--door-seed", type=int, default=None, help="Explicit seed for door generation.")
    parser.add_argument("--monster-seed", type=int, default=None, help="Explicit seed for monster generation.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic door timing.")
    parser.add_argument("--door-count", type=int, default=6, help="Number of doors to place when enabled.")
    parser.add_argument("--monster-count", type=int, default=1, help="Number of patrolling monsters when enabled.")
    parser.add_argument("--monster-interval", type=int, default=1, help="Monster step interval (time steps between moves).")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs to execute in CLI mode.")
    parser.add_argument("--csv-output", type=str, default=None, help="Path to write CSV metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    mode = args.mode or prompt_for_mode()
    if mode == "visual":
        run_visual_mode(args)
    else:
        run_cli_mode(args)


if __name__ == "__main__":
    main()
