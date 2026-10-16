from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import time


class Mesh:

    def __init__(
        self,
        bounds,
        base_level: int = 7,
        max_level: int = 9,
        min_level: int = 5,
    ):
        self.xmin, self.xmax, self.ymin, self.ymax = map(float, bounds)
        self._xmax_open = np.nextafter(self.xmax, self.xmin)
        self._ymax_open = np.nextafter(self.ymax, self.ymin)
        self._x_width = self.xmax - self.xmin
        self._y_width = self.ymax - self.ymin
        self._inv_x_width = 1.0 / self._x_width
        self._inv_y_width = 1.0 / self._y_width

        self.base_level = base_level
        self.min_level = min_level
        self.max_level = max_level

        self.active_ids_by_level = [
            set() for _ in range(max_level + 1)
        ]

        n = 1 << base_level
        self.active_ids_by_level[base_level] = set(range(n * n))

    def _cell_size(self, level: int):
        n = 1 << level
        return self._x_width / n, self._y_width / n

    def _lvl_cell_id(self, level: int, x_idx: int, y_idx: int) -> int:
        return x_idx * (1 << level) + y_idx
        
    def cell_id_parent(self, level: int, cell_id: int) -> int:
        return self._lvl_cell_id(level-1, cell_id >> (level+1), (cell_id % (1<<level)) // 2)

    def cell_id_children(self, level: int, cell_id: int) -> int:
        new_x = (cell_id >> (level)) * 2
        new_y = (cell_id % (1<<level)) * 2
        return (self._lvl_cell_id(level+1, new_x+dx, new_y+dy) for dx, dy in [[0,0],[0,1],[1,0],[1,1]])

    def _unique_ids_at_level(
        self,
        ux: np.ndarray,
        uy: np.ndarray,
        level: int,
    ) -> Set[int]:
        active = self.active_ids_by_level[level]
        if not active:
            return set()

        n = 1 << level
        x_idx = (ux * n).astype(np.int64)
        y_idx = (uy * n).astype(np.int64)

        flat_idx = x_idx * n + y_idx
        unique_idx = np.unique(flat_idx)

        matching_idx = active.intersection(unique_idx.tolist())
        return {
            idx for idx in matching_idx
        }

    def refine(self, level, cell_id) -> bool:
        """
        Refine one active leaf patch by one level.
        """
        # print("Id", level, cell_id)
        if level >= self.max_level:
            # print("Level too high")
            return False

        if cell_id not in self.active_ids_by_level[level]:
            # print("Cell not active")
            return False

        self.active_ids_by_level[level].remove(cell_id)

        child_level = level + 1

        self.active_ids_by_level[child_level].update(
            self.cell_id_children(level, cell_id)
        )

        return True

    def can_coarsen(
        self,
        level,
        parent_id,
    ) -> bool:

        if level < self.min_level:
            return False

        child_level = level + 1
        if child_level > self.max_level:
            return False
        
        active = self.active_ids_by_level[child_level]

        for child_id in self.cell_id_children(level, parent_id):
            if child_id not in active:
                return False

        return True

    def coarsen(
        self,
        level,
        parent_id,
    ) -> bool:
        """
        Merge four active sibling cells into their parent.
        """
        if not self.can_coarsen(level, parent_id):
            return False

        child_level = level + 1
        child_n = 1 << child_level
        children = self.cell_id_children(level, parent_id)
        for child_id in children:
            self.active_ids_by_level[child_level].remove(child_id)

        self.active_ids_by_level[level].add(parent_id)
        # print("Coarsening point")
        return True

    def ids_containing_points(self, points):
        points = np.asarray(points, dtype=float)

        # if points.size == 0:
        #     return set()

        assert points.ndim == 2 and points.shape[1] == 2

        x = np.clip(points[:, 0], self.xmin, self._xmax_open)
        y = np.clip(points[:, 1], self.ymin, self._ymax_open)
        ux = (x - self.xmin) * self._inv_x_width
        uy = (y - self.ymin) * self._inv_y_width

        leaves = [set() for level in range(self.max_level+1)]
        for level in range(self.max_level, self.min_level - 1, -1):
            leaves[level] = self._unique_ids_at_level(ux, uy, level)

        if not leaves:
            raise RuntimeError("No active leaf found.")

        return leaves

    def parents_for_coarsen_points(self, points):
        leaves = self.ids_containing_points(points)
        parents = [{self.cell_id_parent(level, leaf) for leaf in leaves[level]} for level in range(1, self.max_level+1)]

        return parents

    def apply_refine_coarsen(
        self,
        refine_points,
        coarsen_points,
    ):
        """

        refine_points:
            Coordinates lying inside patches to refine.

        coarsen_points:
            Coordinates lying inside patches whose sibling group should be coarsened.

        Refinement takes priority over coarsening
        """

        # print("Points to refine", refine_points)
        # print("Points to coarsen", coarsen_points)
        # t1 = time.time()

        # List of sets
        refine_ids = [set() for level in range(self.max_level+1)]
        # print("refine_ids", refine_ids)
        if refine_points is not None:
            refine_ids = self.ids_containing_points(refine_points)
        # print("Refine ids", refine_ids)

        # print("Getting refinement cells took", time.time()-t1, "s")
        # t1 = time.time()

        # List of sets
        coarsen_parent_ids = [set() for level in range(self.max_level+1)]
        if coarsen_points is not None:
            coarsen_parent_ids = self.parents_for_coarsen_points(coarsen_points)

        # print("Getting coarsening cells took", time.time()-t1, "s")
        # t1 = time.time()

        coarsen_parent_ids = [{
            p
            for p in coarsen_parent_ids[level]
            if not any(child in refine_ids[level+1] for child in self.cell_id_children(level,p))
        } for level in range(len(coarsen_parent_ids)-1)]
        
        # print("Filtering coarsen cells took", time.time()-t1, "s")
        # t1 = time.time()

        for level in range(len(coarsen_parent_ids)):
            for parent in coarsen_parent_ids[level]:
                # print(level, parent)
                self.coarsen(level, parent)
        # print("Coarsening took", time.time()-t1, "s")
        # t1 = time.time()

        for level in range(len(refine_ids)):
            for cell in refine_ids[level]:
                self.refine(level, cell)
        # print("Refinement took", time.time()-t1, "s")


    def points(self):
        n = 1
        level_points = []

        for level, cell_ids in enumerate(self.active_ids_by_level):
            if not cell_ids:
                continue

            dx, dy = self._cell_size(level)
            x_idx = np.fromiter((cell_id>>level for cell_id in cell_ids), dtype=np.int64)
            y_idx = np.fromiter((cell_id & ((1<<level)-1) for cell_id in cell_ids), dtype=np.int64)

            # print(level, x_idx, y_idx)
            level_points.append(
                np.column_stack(
                    (
                        self.xmin + (x_idx + 0.5) * dx,
                        self.ymin + (y_idx + 0.5) * dy,
                    )
                )
            )

        if not level_points:
            return np.empty((0, 2), dtype=float)
        # print("Num points", len(np.concatenate(level_points, axis=0)))

        return np.concatenate(level_points, axis=0)

# unset OMP_NUM_THREADS
