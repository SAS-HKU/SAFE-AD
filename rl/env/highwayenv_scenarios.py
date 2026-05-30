from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
from gymnasium.envs.registration import register, registry


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIGHWAY_ENV_ROOT = os.path.join(REPO_ROOT, "HighwayEnv-master")
if HIGHWAY_ENV_ROOT not in sys.path:
    sys.path.insert(0, HIGHWAY_ENV_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from highway_env.envs.highway_env import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle


class _ScenarioBaseEnv(HighwayEnv):
    ROAD_LENGTH = 420.0
    EGO_LANE_ID = 2
    EGO_X0 = 20.0
    EGO_SPEED = 10.2
    TRUCK_LANE_ID = 1
    TRUCK_X0 = 45.0
    TRUCK_SPEED = 5.3
    BLOCKER_LANE_ID = 2
    BLOCKER_X0 = 80.0
    BLOCKER_SPEED = 10.0
    MERGER_LANE_ID = 0
    MERGER_X0 = 24.0
    MERGER_SPEED = 10.5
    MERGER_RELEASE_MARGIN = 9.0
    MERGER_REVEAL_FALLBACK = 50
    FAST_CAR_LANE_ID = 2
    FAST_CAR_X0 = 78.0
    FAST_CAR_SPEED = 14.0
    FAST_CAR_REVEAL_FALLBACK = 90
    MERGE_X_START = 30.0
    MERGE_X_END = 70.0

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        cfg = super().default_config()
        cfg.update(
            {
                "lanes_count": 3,
                "vehicles_count": 0,
                "controlled_vehicles": 1,
                "initial_lane_id": cls.EGO_LANE_ID,
                "duration": 30,
                "ego_spacing": 1.2,
                "vehicles_density": 0.0,
                "collision_reward": -1.0,
                "right_lane_reward": 0.0,
                "high_speed_reward": 0.5,
                "lane_change_reward": 0.0,
                "reward_speed_range": [6.0, 14.0],
                "normalize_reward": True,
                "offroad_terminal": False,
            }
        )
        return cfg

    def __init__(self, config: dict | None = None, render_mode: str | None = None) -> None:
        self._truck = None
        self._blocker = None
        self._hidden_merger = None
        self._hidden_fast_car = None
        self._merger_revealed = False
        self._fast_car_revealed = False
        self._merger_lc_started_step = None
        self._fast_car_reveal_step = None
        self._merger_reveal_step = None
        super().__init__(config=config, render_mode=render_mode)

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                lanes=3,
                start=0.0,
                length=self.ROAD_LENGTH,
                speed_limit=30.0,
                nodes_str=("a", "b"),
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        road = self.road
        self._merger_revealed = False
        self._fast_car_revealed = False
        self._merger_lc_started_step = None
        self._fast_car_reveal_step = None
        self._merger_reveal_step = None
        self._hidden_fast_car = None
        ego_lane = ("a", "b", self.EGO_LANE_ID)
        ego = self.action_type.vehicle_class(
            road,
            road.network.get_lane(ego_lane).position(self.EGO_X0, 0.0),
            heading=road.network.get_lane(ego_lane).heading_at(self.EGO_X0),
            speed=self.EGO_SPEED,
        )
        ego.target_lane_index = ego_lane
        ego.target_speed = self.EGO_SPEED
        self.vehicle = ego
        road.vehicles.append(ego)

        self._truck = self._spawn_idm_vehicle(
            lane_id=self.TRUCK_LANE_ID,
            x0=self.TRUCK_X0,
            speed=self.TRUCK_SPEED,
            target_speed=self.TRUCK_SPEED,
            reveal=True,
            tag="truck",
        )
        self._truck.LENGTH = 12.0
        self._truck.WIDTH = 2.5
        self._truck.is_truck = True
        self._truck.color = (255, 111, 0)

        self._blocker = self._spawn_idm_vehicle(
            lane_id=self.BLOCKER_LANE_ID,
            x0=self.BLOCKER_X0,
            speed=self.BLOCKER_SPEED,
            target_speed=self.BLOCKER_SPEED,
            reveal=True,
            tag="blocker",
        )

        self._hidden_merger = self._spawn_idm_vehicle(
            lane_id=self.MERGER_LANE_ID,
            x0=self.MERGER_X0,
            speed=self.MERGER_SPEED,
            target_speed=self.MERGER_SPEED,
            reveal=False,
            tag="hidden_merger",
        )
        self._hidden_merger.color = (233, 30, 99)

    def _spawn_idm_vehicle(
        self,
        lane_id: int,
        x0: float,
        speed: float,
        target_speed: float,
        reveal: bool,
        tag: str,
    ) -> IDMVehicle:
        lane_index = ("a", "b", int(lane_id))
        lane = self.road.network.get_lane(lane_index)
        vehicle = IDMVehicle(
            self.road,
            lane.position(float(x0), 0.0),
            heading=lane.heading_at(float(x0)),
            speed=float(speed),
            target_lane_index=lane_index,
            target_speed=float(target_speed),
            enable_lane_change=False,
        )
        vehicle.route = [lane_index]
        vehicle.enable_lane_change = False
        vehicle._scenario_tag = tag
        vehicle._revealed = bool(reveal)
        if reveal:
            self.road.vehicles.append(vehicle)
        return vehicle

    def _hidden_vehicles(self) -> list[Vehicle]:
        hidden = []
        if self._hidden_merger is not None and not self._merger_revealed:
            hidden.append(self._hidden_merger)
        if self._hidden_fast_car is not None and not self._fast_car_revealed:
            hidden.append(self._hidden_fast_car)
        return hidden

    def get_hidden_drift_vehicles(self) -> list[Vehicle]:
        return list(self._hidden_vehicles())

    def get_drift_overlay_config(self) -> dict[str, Any]:
        road = self.road
        return {
            "use_merge_source": True,
            "merge_x_start": float(self.MERGE_X_START),
            "merge_x_end": float(self.MERGE_X_END),
            "merge_from_lane_index": ("a", "b", self.MERGER_LANE_ID),
            "merge_to_lane_index": ("a", "b", self.TRUCK_LANE_ID),
            "road_length": float(self.ROAD_LENGTH),
            "lane_width": float(road.network.get_lane(("a", "b", 0)).width_at(0.0)),
        }

    def get_scenario_info(self) -> dict[str, Any]:
        return {
            "merger_revealed": bool(self._merger_revealed),
            "fast_car_revealed": bool(self._fast_car_revealed),
            "merger_lc_started_step": self._merger_lc_started_step,
            "merger_reveal_step": self._merger_reveal_step,
            "fast_car_reveal_step": self._fast_car_reveal_step,
        }

    def _simulate(self, action=None) -> None:
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        for frame in range(frames):
            if (
                action is not None
                and not self.config["manual_control"]
                and self.steps % frames == 0
            ):
                self.action_type.act(action)

            self._pre_step_hidden_logic()

            self.road.act()
            for hidden in self._hidden_vehicles():
                hidden.act()

            dt = 1 / self.config["simulation_frequency"]
            self.road.step(dt)
            for hidden in self._hidden_vehicles():
                hidden.step(dt)

            self.steps += 1
            self._post_step_hidden_logic()

            if frame < frames - 1:
                self._automatic_rendering()
        self.enable_auto_render = False

    def _pre_step_hidden_logic(self) -> None:
        if self._hidden_merger is not None and not self._merger_revealed:
            if self._hidden_merger.position[0] > self._truck.position[0] + self.MERGER_RELEASE_MARGIN:
                self._hidden_merger.target_lane_index = ("a", "b", self.TRUCK_LANE_ID)
                if self._merger_lc_started_step is None:
                    self._merger_lc_started_step = int(self.steps)

    def _post_step_hidden_logic(self) -> None:
        if self._hidden_merger is not None and not self._merger_revealed:
            current_y = float(self._hidden_merger.position[1])
            lane_gap = abs(
                self.road.network.get_lane(("a", "b", self.MERGER_LANE_ID)).position(0.0, 0.0)[1]
                - self.road.network.get_lane(("a", "b", self.TRUCK_LANE_ID)).position(0.0, 0.0)[1]
            )
            merge_target_y = float(
                self.road.network.get_lane(("a", "b", self.TRUCK_LANE_ID)).position(0.0, 0.0)[1]
            )
            halfway = abs(current_y - merge_target_y) <= lane_gap * 0.5
            if halfway or self.steps >= self.MERGER_REVEAL_FALLBACK:
                self._reveal_vehicle(self._hidden_merger)
                self._merger_revealed = True
                self._merger_reveal_step = int(self.steps)

        if self._hidden_fast_car is not None and not self._fast_car_revealed:
            if (not self._is_occluded_by_truck(self._hidden_fast_car)) or self.steps >= self.FAST_CAR_REVEAL_FALLBACK:
                self._reveal_vehicle(self._hidden_fast_car)
                self._fast_car_revealed = True
                self._fast_car_reveal_step = int(self.steps)

    def _reveal_vehicle(self, vehicle: Vehicle) -> None:
        if vehicle not in self.road.vehicles:
            vehicle._revealed = True
            self.road.vehicles.append(vehicle)

    def _is_occluded_by_truck(self, candidate: Vehicle) -> bool:
        ego = self.vehicle
        truck = self._truck
        if candidate is None or truck is None:
            return False
        dx = float(candidate.position[0] - truck.position[0])
        dy = float(candidate.position[1] - truck.position[1])
        if float(candidate.position[0]) <= float(ego.position[0]):
            return False
        if not (float(ego.position[0]) < float(truck.position[0]) < float(candidate.position[0])):
            return False
        heading = float(truck.heading)
        cos_h = float(np.cos(heading))
        sin_h = float(np.sin(heading))
        along_road = cos_h * dx + sin_h * dy
        if along_road <= 0.0:
            return False
        dist_from_truck = float(np.hypot(dx, dy))
        if dist_from_truck >= 60.0:
            return False
        lateral_dist = abs(-sin_h * dx + cos_h * dy)
        truck_width = float(getattr(truck, "WIDTH", 2.5))
        shadow_width = truck_width * 0.5 + dist_from_truck * 0.3
        return lateral_dist <= shadow_width


class OccludedMergerHighwayEnv(_ScenarioBaseEnv):
    pass


class TwoThreatHighwayEnv(_ScenarioBaseEnv):
    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        self._hidden_fast_car = self._spawn_idm_vehicle(
            lane_id=self.FAST_CAR_LANE_ID,
            x0=self.FAST_CAR_X0,
            speed=self.FAST_CAR_SPEED,
            target_speed=self.FAST_CAR_SPEED,
            reveal=False,
            tag="hidden_fast_car",
        )
        self._hidden_fast_car.color = (156, 39, 176)


def register_highwayenv_scenarios() -> None:
    specs = {
        "occluded-merger-v0": "rl.env.highwayenv_scenarios:OccludedMergerHighwayEnv",
        "two-threat-v0": "rl.env.highwayenv_scenarios:TwoThreatHighwayEnv",
    }
    for env_id, entry_point in specs.items():
        if env_id in registry:
            continue
        register(id=env_id, entry_point=entry_point)
