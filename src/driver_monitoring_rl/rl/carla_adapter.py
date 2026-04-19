from __future__ import annotations

"""CARLA adapter for the drowsiness-aware braking RL environment.

This module provides :class:`CARLABrakingEnv`, a drop-in alternative to
``DrowsyBrakingEnv`` from ``environment.py``. It keeps the same observable state
shape and discrete action mapping while sourcing relative distance/velocity from
CARLA radar detections clustered with DBSCAN.

The adapter is intentionally self-contained and optional:
- If CARLA is not installed, importing this module succeeds.
- A runtime error with setup guidance is raised only when the environment is
  instantiated.
"""

from dataclasses import dataclass
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

try:
    import carla  # type: ignore
except ImportError:  # pragma: no cover - exercised when CARLA is unavailable
    carla = None


class CarlaAdapterError(RuntimeError):
    """Base exception for CARLA adapter failures."""


class CarlaConnectionError(CarlaAdapterError):
    """Raised when CARLA server connection/setup fails."""


@dataclass
class EnvState:
    v_ego: float
    action_idx: int
    d_rel: float
    v_rel: float
    drowsy: int


@dataclass
class _ClusterEstimate:
    distance_m: float
    relative_velocity_mps: float
    num_points: int


class CARLABrakingEnv:
    """Drowsiness-aware braking environment backed by CARLA 0.9.x.

    Interface compatibility:
      - ``state_dim`` and ``action_dim`` properties
      - ``reset(drowsy: Optional[int]) -> np.ndarray``
      - ``step(action: int) -> Tuple[np.ndarray, float, bool, Dict]``

    Observation/state definition is intentionally identical to ``DrowsyBrakingEnv``:
      ``[v_ego, action_idx, d_rel, v_rel, drowsy]``.

    Notes
    -----
    - Radar sensor data are transformed into point clouds in ego coordinates and
      clustered with DBSCAN; nearest frontal cluster is treated as lead vehicle.
    - Drowsiness can be forced at reset and/or injected dynamically through
      ``set_drowsiness_state`` or a custom ``drowsy_injector`` callback.
    - Supports CARLA 0.9.x Python API conventions.
    """

    ACTION_TO_CONTROL = {
        0: (0.0, 1.0),
        1: (0.0, 0.7),
        2: (0.0, 0.4),
        3: (0.0, 0.2),
        4: (0.0, 0.0),
        5: (1.0, 0.0),
    }

    def __init__(
        self,
        dt: float = 0.1,
        max_steps: int = 300,
        drowsy_delay_s: float = 0.5,
        seed: int = 42,
        host: str = "127.0.0.1",
        port: int = 2000,
        timeout_s: float = 10.0,
        town: Optional[str] = None,
        synchronous_mode: bool = True,
        traffic_manager_port: int = 8000,
        radar_range_m: float = 80.0,
        radar_hfov_deg: float = 25.0,
        radar_vfov_deg: float = 5.0,
        radar_points_per_second: int = 1400,
        radar_lateral_limit_m: float = 3.0,
        radar_vertical_limit_m: float = 2.5,
        dbscan_eps_m: float = 1.75,
        dbscan_min_samples: int = 3,
        radar_velocity_sign: float = 1.0,
        lead_spawn_distance_m: float = 25.0,
        random_drowsy_flip_prob: float = 0.015,
        drowsy_injector: Optional[Callable[[int, EnvState], Optional[int]]] = None,
    ):
        if carla is None:
            raise CarlaAdapterError(
                "CARLA Python API is not installed. Install CARLA 0.9.x egg/wheel "
                "and ensure it is available in PYTHONPATH. See docs/CARLA_INTEGRATION.md."
            )

        self.dt = dt
        self.max_steps = max_steps
        self.delay_steps = max(1, int(round(drowsy_delay_s / dt)))
        self.rng = np.random.default_rng(seed)

        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.town = town
        self.synchronous_mode = synchronous_mode
        self.traffic_manager_port = traffic_manager_port

        self.radar_range_m = radar_range_m
        self.radar_hfov_deg = radar_hfov_deg
        self.radar_vfov_deg = radar_vfov_deg
        self.radar_points_per_second = radar_points_per_second
        self.radar_lateral_limit_m = radar_lateral_limit_m
        self.radar_vertical_limit_m = radar_vertical_limit_m
        self.dbscan_eps_m = dbscan_eps_m
        self.dbscan_min_samples = dbscan_min_samples
        self.radar_velocity_sign = radar_velocity_sign
        self.lead_spawn_distance_m = lead_spawn_distance_m
        self.random_drowsy_flip_prob = random_drowsy_flip_prob
        self.drowsy_injector = drowsy_injector

        self.step_count = 0
        self.prev_action = 4
        self.v_front = 10.0
        self.state = EnvState(0.0, 4, 30.0, 0.0, 0)
        self._forced_drowsy: Optional[int] = None

        self._client = None
        self._world = None
        self._map = None
        self._tm = None

        self._ego_vehicle = None
        self._lead_vehicle = None
        self._radar_sensor = None
        self._actors: List = []
        self._orig_world_settings = None

        self._radar_lock = Lock()
        self._latest_radar_points: List[Tuple[float, float, float, float]] = []

        self._connect()

    @property
    def state_dim(self) -> int:
        return 5

    @property
    def action_dim(self) -> int:
        return 6

    @classmethod
    def check_connection(
        cls,
        host: str = "127.0.0.1",
        port: int = 2000,
        timeout_s: float = 3.0,
    ) -> bool:
        """Return ``True`` if a CARLA server responds at ``host:port``."""
        if carla is None:
            return False
        try:
            client = carla.Client(host, port)
            client.set_timeout(timeout_s)
            _ = client.get_world()
            return True
        except Exception:
            return False

    def set_drowsiness_state(self, drowsy: int) -> None:
        """Force drowsiness state for subsequent steps.

        Parameters
        ----------
        drowsy:
            ``0`` (alert) or ``1`` (drowsy).
        """
        if drowsy not in (0, 1):
            raise ValueError("drowsy must be 0 or 1")
        self._forced_drowsy = int(drowsy)

    def clear_drowsiness_override(self) -> None:
        """Disable forced drowsiness and return to stochastic/callback updates."""
        self._forced_drowsy = None

    def reset(self, drowsy: int | None = None) -> np.ndarray:
        """Reset simulation episode and return the initial observation."""
        self.step_count = 0
        self.prev_action = 4
        self._latest_radar_points = []

        self._destroy_episode_actors()
        self._spawn_episode_actors()

        theta = int(self.rng.integers(0, 2) if drowsy is None else drowsy)
        self.state = EnvState(v_ego=self._ego_speed(), action_idx=4, d_rel=50.0, v_rel=0.0, drowsy=theta)

        # Warm-up a few ticks so radar starts streaming.
        for _ in range(3):
            self._tick_world()
        d_rel, v_rel, _ = self._estimate_front_from_radar(default_d=50.0, default_v_rel=0.0)
        self.state = EnvState(v_ego=self._ego_speed(), action_idx=4, d_rel=d_rel, v_rel=v_rel, drowsy=theta)
        return self._to_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Advance one step in CARLA using the same logic as ``DrowsyBrakingEnv``."""
        if action not in self.ACTION_TO_CONTROL:
            raise ValueError(f"Invalid action {action}. Expected one of {sorted(self.ACTION_TO_CONTROL)}")
        if self._ego_vehicle is None:
            raise CarlaAdapterError("Environment has no spawned ego vehicle. Call reset() before step().")

        self.step_count += 1

        # Drowsiness-induced execution delay: execute previous intended action.
        exec_action = self.prev_action if self.state.drowsy == 1 else action
        self.prev_action = action

        throttle, brake = self.ACTION_TO_CONTROL[exec_action]
        self._apply_ego_control(throttle=throttle, brake=brake)

        # Keep a simple stochastic lead-vehicle profile for a stable training target.
        self._update_lead_vehicle_behavior()

        self._tick_world()

        v_ego = self._ego_speed()
        d_rel, v_rel, cluster = self._estimate_front_from_radar(
            default_d=self.state.d_rel,
            default_v_rel=self.state.v_rel,
        )

        dmin, dmax = self._safe_range(v_ego)
        collision = d_rel <= 0.75
        delta_pressure = abs(
            self.ACTION_TO_CONTROL[exec_action][1] - self.ACTION_TO_CONTROL[self.state.action_idx][1]
        )

        reward = 0.0
        reward += -200.0 if collision else 0.0
        reward += -2.0 * delta_pressure
        if d_rel < dmin:
            reward += -10.0
        elif d_rel > dmax:
            reward += -2.0
        else:
            reward += +10.0

        if exec_action == 5 and self.state.v_ego < 0.5:
            reward += +1.0
        if dmin <= d_rel <= dmax and exec_action in (1, 2, 3):
            reward += +2.0
        if dmin <= d_rel <= dmax and abs(v_rel) < 1.0:
            reward += +5.0

        theta = self._next_drowsiness_state(step=self.step_count)

        done = collision or self.step_count >= self.max_steps
        if done and not collision:
            reward += +1.0

        self.state = EnvState(v_ego=v_ego, action_idx=exec_action, d_rel=d_rel, v_rel=v_rel, drowsy=theta)

        info = {
            "collision": collision,
            "safe_min": dmin,
            "safe_max": dmax,
            "throttle": throttle,
            "brake": brake,
            "delay_active": self.state.drowsy == 1,
            "radar_cluster_points": 0 if cluster is None else cluster.num_points,
            "carla_host": self.host,
            "carla_port": self.port,
        }
        return self._to_obs(self.state), float(reward), bool(done), info

    def close(self) -> None:
        """Destroy actors and restore world settings."""
        self._destroy_episode_actors()
        if self._world is not None and self._orig_world_settings is not None:
            try:
                self._world.apply_settings(self._orig_world_settings)
            except Exception:
                pass
        self._world = None
        self._map = None

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _connect(self) -> None:
        try:
            self._client = carla.Client(self.host, self.port)
            self._client.set_timeout(self.timeout_s)
            if self.town:
                self._world = self._client.load_world(self.town)
            else:
                self._world = self._client.get_world()
            self._map = self._world.get_map()

            self._orig_world_settings = self._world.get_settings()
            if self.synchronous_mode:
                settings = self._world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.dt
                self._world.apply_settings(settings)

                self._tm = self._client.get_trafficmanager(self.traffic_manager_port)
                self._tm.set_synchronous_mode(True)
        except Exception as exc:
            raise CarlaConnectionError(
                f"Failed to connect to CARLA server at {self.host}:{self.port}. "
                "Ensure the CARLA simulator is running and API version is 0.9.x. "
                f"Original error: {exc}"
            ) from exc

    def _spawn_episode_actors(self) -> None:
        if self._world is None or self._map is None:
            raise CarlaAdapterError("CARLA world is not initialized.")

        blueprints = self._world.get_blueprint_library()
        ego_bp = blueprints.filter("vehicle.tesla.model3")
        ego_bp = ego_bp[0] if ego_bp else blueprints.filter("vehicle.*")[0]

        spawn_points = self._map.get_spawn_points()
        if not spawn_points:
            raise CarlaAdapterError("No vehicle spawn points available in current CARLA map.")

        ego_transform = spawn_points[int(self.rng.integers(0, len(spawn_points)))]
        self._ego_vehicle = self._try_spawn_vehicle(ego_bp, ego_transform)
        if self._ego_vehicle is None:
            raise CarlaAdapterError("Could not spawn ego vehicle. Try restarting CARLA or changing town.")
        self._actors.append(self._ego_vehicle)

        lead_bp = blueprints.filter("vehicle.*")[int(self.rng.integers(0, len(blueprints.filter('vehicle.*'))))]
        lead_transform = self._lead_transform_ahead(ego_transform, self.lead_spawn_distance_m)
        self._lead_vehicle = self._try_spawn_vehicle(lead_bp, lead_transform)
        if self._lead_vehicle is None:
            # fallback to a random spawn point if ahead spawn is blocked
            lead_transform = spawn_points[int(self.rng.integers(0, len(spawn_points)))]
            self._lead_vehicle = self._try_spawn_vehicle(lead_bp, lead_transform)

        if self._lead_vehicle is not None:
            self._actors.append(self._lead_vehicle)
            self.v_front = float(self.rng.uniform(7.0, 14.0))
            self._set_actor_forward_speed(self._lead_vehicle, self.v_front)

        self._attach_radar_sensor()

    def _try_spawn_vehicle(self, blueprint, transform):
        try:
            return self._world.try_spawn_actor(blueprint, transform)
        except Exception:
            return None

    def _lead_transform_ahead(self, ego_transform, distance_m: float):
        waypoint = self._map.get_waypoint(ego_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        next_wps = waypoint.next(distance_m)
        if next_wps:
            return next_wps[0].transform
        return ego_transform

    def _attach_radar_sensor(self) -> None:
        blueprints = self._world.get_blueprint_library()
        radar_bp = blueprints.find("sensor.other.radar")
        radar_bp.set_attribute("horizontal_fov", str(self.radar_hfov_deg))
        radar_bp.set_attribute("vertical_fov", str(self.radar_vfov_deg))
        radar_bp.set_attribute("range", str(self.radar_range_m))
        radar_bp.set_attribute("points_per_second", str(self.radar_points_per_second))

        radar_transform = carla.Transform(carla.Location(x=2.2, z=1.0))
        self._radar_sensor = self._world.spawn_actor(radar_bp, radar_transform, attach_to=self._ego_vehicle)
        self._actors.append(self._radar_sensor)
        self._radar_sensor.listen(self._on_radar_data)

    def _on_radar_data(self, radar_measurement) -> None:
        points: List[Tuple[float, float, float, float]] = []
        for detection in radar_measurement:
            depth = float(getattr(detection, "depth", 0.0))
            azimuth = float(getattr(detection, "azimuth", 0.0))
            altitude = float(getattr(detection, "altitude", 0.0))
            velocity = float(getattr(detection, "velocity", 0.0))

            # Polar -> Cartesian in sensor frame (x forward).
            x = depth * np.cos(altitude) * np.cos(azimuth)
            y = depth * np.cos(altitude) * np.sin(azimuth)
            z = depth * np.sin(altitude)
            rel_v = self.radar_velocity_sign * velocity

            points.append((float(x), float(y), float(z), float(rel_v)))

        with self._radar_lock:
            self._latest_radar_points = points

    def _estimate_front_from_radar(
        self,
        default_d: float,
        default_v_rel: float,
    ) -> Tuple[float, float, Optional[_ClusterEstimate]]:
        with self._radar_lock:
            points = list(self._latest_radar_points)

        if not points:
            return float(default_d), float(default_v_rel), None

        arr = np.asarray(points, dtype=np.float32)
        x, y, z, vr = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        frontal_mask = (
            (x > 0.0)
            & (x <= self.radar_range_m)
            & (np.abs(y) <= self.radar_lateral_limit_m)
            & (np.abs(z) <= self.radar_vertical_limit_m)
        )
        if not np.any(frontal_mask):
            return float(default_d), float(default_v_rel), None

        frontal_xyz = arr[frontal_mask][:, :3]
        frontal_vr = vr[frontal_mask]

        if len(frontal_xyz) < self.dbscan_min_samples:
            nearest_idx = int(np.argmin(frontal_xyz[:, 0]))
            return float(frontal_xyz[nearest_idx, 0]), float(frontal_vr[nearest_idx]), _ClusterEstimate(
                distance_m=float(frontal_xyz[nearest_idx, 0]),
                relative_velocity_mps=float(frontal_vr[nearest_idx]),
                num_points=1,
            )

        labels = DBSCAN(eps=self.dbscan_eps_m, min_samples=self.dbscan_min_samples).fit_predict(frontal_xyz)
        valid_clusters = [lab for lab in np.unique(labels) if lab != -1]
        if not valid_clusters:
            nearest_idx = int(np.argmin(frontal_xyz[:, 0]))
            return float(frontal_xyz[nearest_idx, 0]), float(frontal_vr[nearest_idx]), _ClusterEstimate(
                distance_m=float(frontal_xyz[nearest_idx, 0]),
                relative_velocity_mps=float(frontal_vr[nearest_idx]),
                num_points=1,
            )

        best_cluster: Optional[_ClusterEstimate] = None
        for cluster_id in valid_clusters:
            cluster_mask = labels == cluster_id
            cluster_xyz = frontal_xyz[cluster_mask]
            cluster_vr = frontal_vr[cluster_mask]
            dist = float(np.median(cluster_xyz[:, 0]))
            rel_v = float(np.mean(cluster_vr))
            estimate = _ClusterEstimate(distance_m=dist, relative_velocity_mps=rel_v, num_points=int(cluster_mask.sum()))
            if best_cluster is None or estimate.distance_m < best_cluster.distance_m:
                best_cluster = estimate

        if best_cluster is None:
            return float(default_d), float(default_v_rel), None
        return best_cluster.distance_m, best_cluster.relative_velocity_mps, best_cluster

    def _update_lead_vehicle_behavior(self) -> None:
        if self._lead_vehicle is None:
            return
        front_accel = float(self.rng.normal(loc=0.0, scale=0.35))
        self.v_front = float(np.clip(self.v_front + front_accel * self.dt, 2.0, 20.0))
        self._set_actor_forward_speed(self._lead_vehicle, self.v_front)

    def _set_actor_forward_speed(self, actor, speed_mps: float) -> None:
        tf = actor.get_transform()
        forward = tf.get_forward_vector()
        vel = carla.Vector3D(x=forward.x * speed_mps, y=forward.y * speed_mps, z=forward.z * speed_mps)
        actor.set_target_velocity(vel)

    def _apply_ego_control(self, throttle: float, brake: float) -> None:
        control = carla.VehicleControl(throttle=float(throttle), brake=float(brake), steer=0.0, hand_brake=False)
        self._ego_vehicle.apply_control(control)

    def _ego_speed(self) -> float:
        if self._ego_vehicle is None:
            return 0.0
        v = self._ego_vehicle.get_velocity()
        return float(np.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

    def _tick_world(self) -> None:
        if self._world is None:
            raise CarlaAdapterError("CARLA world is not available.")
        try:
            if self.synchronous_mode:
                self._world.tick()
            else:
                self._world.wait_for_tick()
        except Exception as exc:
            raise CarlaAdapterError(f"Failed during CARLA world tick: {exc}") from exc

    def _next_drowsiness_state(self, step: int) -> int:
        if self._forced_drowsy is not None:
            return int(self._forced_drowsy)

        theta = int(self.state.drowsy)
        if self.drowsy_injector is not None:
            injected = self.drowsy_injector(step, self.state)
            if injected in (0, 1):
                return int(injected)

        if self.rng.random() < self.random_drowsy_flip_prob:
            theta = 1 - theta
        return theta

    def _safe_range(self, v_ego: float) -> Tuple[float, float]:
        dmin = max(5.0, 2.0 * v_ego)
        return dmin, dmin + 10.0

    def _to_obs(self, s: EnvState) -> np.ndarray:
        return np.array([s.v_ego, s.action_idx, s.d_rel, s.v_rel, s.drowsy], dtype=np.float32)

    def _destroy_episode_actors(self) -> None:
        for actor in reversed(self._actors):
            try:
                if hasattr(actor, "stop"):
                    actor.stop()
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass
        self._actors = []
        self._ego_vehicle = None
        self._lead_vehicle = None
        self._radar_sensor = None
