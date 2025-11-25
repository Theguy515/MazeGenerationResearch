#!/usr/bin/env python3
"""Pygame-based visualization for the maze generators/solvers."""

from __future__ import annotations

import pygame


class MazeVisualizer:
    """Displays maze solver progress using pygame."""

    def __init__(
        self,
        maze,
        doors,
        solver_factories,
        start_cell,
        goal_cell,
        tile_size=24,
        stats_height=120,
        max_cols=3,
    ):
        self.maze = maze
        self.doors = doors
        self.solver_factories = solver_factories
        self.start_cell = start_cell
        self.goal_cell = goal_cell
        self.tile_size = tile_size
        self.stats_height = stats_height
        self.max_cols = max_cols

    def _cell_to_grid(self, x, y):
        return 2 * x + 1, 2 * y + 1

    def _compute_layout(self, grid_cols, grid_rows, num_visualizers):
        pygame.display.init()
        display_info = pygame.display.Info()
        usable_w = max(320, display_info.current_w - 80)
        usable_h = max(240, display_info.current_h - 120)

        num_cols = min(self.max_cols, num_visualizers)
        num_rows = (num_visualizers + num_cols - 1) // num_cols

        max_tile_w = usable_w // (grid_cols * num_cols)
        max_tile_h = max(1, (usable_h - self.stats_height * num_rows) // (grid_rows * num_rows))
        tile_size = max(4, min(self.tile_size, max_tile_w, max_tile_h))

        stats_height = max(70, int(self.stats_height * (tile_size / self.tile_size)))
        view_width = grid_cols * tile_size
        panel_height = grid_rows * tile_size + stats_height
        screen_width = view_width * num_cols
        screen_height = panel_height * num_rows

        return tile_size, stats_height, view_width, panel_height, screen_width, screen_height, num_cols, num_rows

    def run(self):
        grid = self.maze.to_grid()
        grid_rows = len(grid)
        grid_cols = len(grid[0])

        tile_size, stats_height, view_width, panel_height, screen_width, screen_height, num_cols, num_rows = (
            self._compute_layout(grid_cols, grid_rows, len(self.solver_factories))
        )

        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Time-dependent Maze Solvers")
        font = pygame.font.SysFont(None, max(14, int(18 * tile_size / 24)))
        clock = pygame.time.Clock()

        solver_visualizers = [
            factory()
            for factory in self.solver_factories
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

                for cell in snapshot.visited:
                    gx, gy = self._cell_to_grid(*cell)
                    rect = pygame.Rect(
                        offset_x + gx * tile_size,
                        offset_y + gy * tile_size,
                        tile_size,
                        tile_size,
                    )
                    draw_alpha_rect(screen, solver.color, rect, 60)

                for cell in snapshot.frontier:
                    gx, gy = self._cell_to_grid(*cell)
                    rect = pygame.Rect(
                        offset_x + gx * tile_size,
                        offset_y + gy * tile_size,
                        tile_size,
                        tile_size,
                    )
                    draw_alpha_rect(screen, solver.color, rect, 110)

                for cell in snapshot.path:
                    gx, gy = self._cell_to_grid(*cell)
                    rect = pygame.Rect(
                        offset_x + gx * tile_size,
                        offset_y + gy * tile_size,
                        tile_size,
                        tile_size,
                    )
                    draw_alpha_rect(screen, solver.color, rect, 180)

                if snapshot.current:
                    gx, gy = self._cell_to_grid(*snapshot.current)
                    rect = pygame.Rect(
                        offset_x + gx * tile_size,
                        offset_y + gy * tile_size,
                        tile_size,
                        tile_size,
                    )
                    draw_alpha_rect(screen, solver.color, rect, 230)

                sx, sy = self._cell_to_grid(*self.start_cell)
                gx, gy = self._cell_to_grid(*self.goal_cell)
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

                for door, is_open in self.doors.states_at(snapshot.time_step):
                    agx, agy = self._cell_to_grid(*door.a)
                    bgx, bgy = self._cell_to_grid(*door.b)
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
                            offset_y + grid_rows * tile_size + 8 + i * max(16, int(18 * tile_size / 24)),
                        ),
                    )

            pygame.display.flip()

        pygame.quit()
