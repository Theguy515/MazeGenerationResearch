#Pygame visualization for mazes+algorithms

from __future__ import annotations

import pygame


class MazeVisualizer:
    #For maze solver progress using pygame

    def __init__(
        self,
        maze,
        doors,
        solver_factories,
        start_cell,
        goal_cell,
        monsters=None,
        tile_size=24,
        stats_height=120,
        max_cols=3,
        title_suffix="",
    ):
        self.maze = maze
        self.doors = doors
        self.solver_factories = solver_factories
        self.start_cell = start_cell
        self.goal_cell = goal_cell
        self.monsters = monsters
        self.tile_size = tile_size
        self.stats_height = stats_height
        self.max_cols = max_cols
        self.title_suffix = title_suffix

    def _cell_to_grid(self, x, y):
        return 2 * x + 1, 2 * y + 1

    def _compute_layout(self, grid_cols, grid_rows, num_visualizers, container_w, container_h):
        num_cols = min(self.max_cols, max(1, num_visualizers))
        num_rows = (num_visualizers + num_cols - 1) // num_cols

        usable_w = max(320, container_w - 16)
        usable_h = max(240, container_h - 16)

        tile_size = self.tile_size
        stats_height = self.stats_height
        for _ in range(4):
            max_tile_w = max(4, usable_w // (grid_cols * num_cols))
            max_tile_h = max(4, (usable_h - stats_height * num_rows) // (grid_rows * num_rows))
            tile_size = max(4, min(max_tile_w, max_tile_h))
            line_height = max(16, int(18 * tile_size / 24))
            stats_height = max(70, line_height * 6)

        view_width = grid_cols * tile_size
        panel_height = grid_rows * tile_size + stats_height
        return tile_size, stats_height, view_width, panel_height, num_cols, num_rows

    def run(self):
        grid = self.maze.to_grid()
        grid_rows = len(grid)
        grid_cols = len(grid[0])

        pygame.init()
        display_info = pygame.display.Info()
        default_w = max(640, int(display_info.current_w * 0.9))
        default_h = max(480, int(display_info.current_h * 0.8))
        screen = pygame.display.set_mode((default_w, default_h), pygame.RESIZABLE)
        pygame.display.set_caption(f"Time-dependent Maze Solvers{self.title_suffix}")
        tile_size, stats_height, view_width, panel_height, num_cols, num_rows = (
            self._compute_layout(grid_cols, grid_rows, len(self.solver_factories), *screen.get_size())
        )
        font_size = max(14, int(18 * tile_size / 24))
        font = pygame.font.SysFont(None, font_size)
        clock = pygame.time.Clock()
        fullscreen = False
        last_window_size = screen.get_size()

        solver_visualizers = [
            factory()
            for factory in self.solver_factories
        ]

        #color schemes for visual aspects
        colors = {
            "wall": (20, 20, 20),
            "floor": (230, 230, 230),
            "start": (50, 200, 90),
            "goal": (210, 60, 60),
            "door_open": (80, 180, 250),
            "door_closed": (220, 90, 90),
            "monster": (255, 80, 130),
        }

        line_height = max(16, int(18 * tile_size / 24))

        #Draws semi transparent overlays for visited paths (helps visualize the algorithms working)
        def draw_alpha_rect(surface, color, rect, alpha):
            overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            overlay.fill((*color, alpha))
            surface.blit(overlay, rect.topleft)

        running = True
        while running:

            #Change this number to speed up or slow down the visualizer
            #Lower numbers correspond to lower frames per second
            #Higher numbers correspons to higher FPS, increase this number to have the visualizer run faster
            clock.tick(5)
            tile_size, stats_height, view_width, panel_height, num_cols, num_rows = (
                self._compute_layout(grid_cols, grid_rows, len(self.solver_factories), *screen.get_size())
            )
            new_font_size = max(14, int(18 * tile_size / 24))
            if new_font_size != font_size:
                font_size = new_font_size
                font = pygame.font.SysFont(None, font_size)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                if event.type == pygame.VIDEORESIZE and not fullscreen:
                    last_window_size = (event.w, event.h)
                    screen = pygame.display.set_mode(last_window_size, pygame.RESIZABLE)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    fullscreen = not fullscreen
                    if fullscreen:
                        display_info = pygame.display.Info()
                        screen = pygame.display.set_mode((display_info.current_w, display_info.current_h), pygame.FULLSCREEN)
                    else:
                        screen = pygame.display.set_mode(last_window_size, pygame.RESIZABLE)

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

                if self.doors:
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

                if self.monsters:
                    for mx, my in self.monsters.positions_at(snapshot.time_step):
                        gx, gy = self._cell_to_grid(mx, my)
                        rect = pygame.Rect(
                            offset_x + gx * tile_size,
                            offset_y + gy * tile_size,
                            tile_size,
                            tile_size,
                        )
                        draw_alpha_rect(screen, colors["monster"], rect, 240)

                path_len = len(snapshot.path) if snapshot.path else "-"
                lines = [
                    f"{solver.name}",
                    f"elapsed: {snapshot.elapsed:.2f}s",
                    f"success: {snapshot.success}",
                    f"expanded: {snapshot.expanded}",
                    f"time step: {snapshot.time_step}",
                    f"path length: {path_len}",
                ]
                pad = 6
                overlay_height = line_height * len(lines) + pad * 2
                stats_rect = pygame.Rect(
                    offset_x,
                    offset_y + grid_rows * tile_size,
                    view_width,
                    max(stats_height, overlay_height),
                )
                pygame.draw.rect(screen, (25, 25, 25), stats_rect)

                for i, text in enumerate(lines):
                    surface = font.render(text, True, (235, 235, 235))
                    screen.blit(
                        surface,
                        (
                            stats_rect.x + pad,
                            stats_rect.y + pad + i * line_height,
                        ),
                    )

            pygame.display.flip()

        pygame.quit()
