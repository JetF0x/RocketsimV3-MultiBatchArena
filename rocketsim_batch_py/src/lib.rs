//! Python bindings for `rocketsim_batch`, exposed as `rocketsim_batch._core`.

use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use rocketsim_batch::{
    BatchArena as RsBatchArena, BatchConfig, Scheduling, layout,
    rocketsim::{ArenaConfig, ArenaMemWeightMode, CarBodyConfig, GameMode, Vec3A},
};

fn parse_game_mode(name: &str) -> PyResult<GameMode> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "soccar" => GameMode::Soccar,
        "hoops" => GameMode::Hoops,
        "heatseeker" => GameMode::Heatseeker,
        "snowday" => GameMode::Snowday,
        "dropshot" => GameMode::Dropshot,
        "void" | "thevoid" | "the_void" => GameMode::TheVoid,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown game_mode {name:?}; expected soccar, hoops, heatseeker, snowday, dropshot or void"
            )));
        }
    })
}

fn parse_car_body(name: &str) -> PyResult<CarBodyConfig> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "octane" => CarBodyConfig::OCTANE,
        "dominus" => CarBodyConfig::DOMINUS,
        "plank" | "batmobile" => CarBodyConfig::PLANK,
        "breakout" => CarBodyConfig::BREAKOUT,
        "hybrid" => CarBodyConfig::HYBRID,
        "merc" => CarBodyConfig::MERC,
        "psyclops" => CarBodyConfig::PSYCLOPS,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown car_body {name:?}; expected octane, dominus, plank, breakout, hybrid, merc or psyclops"
            )));
        }
    })
}

fn parse_mem_mode(name: &str) -> PyResult<ArenaMemWeightMode> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "heavy" => ArenaMemWeightMode::Heavy,
        "balanced" => ArenaMemWeightMode::Balanced,
        "light" => ArenaMemWeightMode::Light,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown mem_weight_mode {name:?}; expected heavy, balanced or light"
            )));
        }
    })
}

/// Copies `data` into a new numpy array of the given shape.
fn to_numpy<'py, T: numpy::Element + Copy>(
    py: Python<'py>,
    data: &[T],
    shape: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyArray1::from_slice(py, data).reshape(shape)?.into_any())
}

fn vec3(v: [f32; 3]) -> Vec3A {
    Vec3A::from_array(v)
}

/// Many RocketSim v3 arenas stepped in parallel (see the package docstring).
#[pyclass(module = "rocketsim_batch._core")]
struct BatchArena {
    inner: RsBatchArena,
}

impl BatchArena {
    fn check_arena(&self, arena: usize) -> PyResult<()> {
        if arena >= self.inner.num_arenas() {
            return Err(PyIndexError::new_err(format!(
                "arena index {arena} out of range (num_arenas = {})",
                self.inner.num_arenas()
            )));
        }
        Ok(())
    }

    fn check_car(&self, arena: usize, car: usize) -> PyResult<()> {
        self.check_arena(arena)?;
        if car >= self.inner.cars_per_arena() {
            return Err(PyIndexError::new_err(format!(
                "car index {car} out of range (cars_per_arena = {})",
                self.inner.cars_per_arena()
            )));
        }
        Ok(())
    }

    fn per_arena<'py, T: numpy::Element + Copy>(
        &self,
        py: Python<'py>,
        data: &[T],
    ) -> PyResult<Bound<'py, PyAny>> {
        to_numpy(py, data, &[self.inner.num_arenas()])
    }

    fn per_car<'py, T: numpy::Element + Copy>(
        &self,
        py: Python<'py>,
        data: &[T],
    ) -> PyResult<Bound<'py, PyAny>> {
        to_numpy(
            py,
            data,
            &[self.inner.num_arenas(), self.inner.cars_per_arena()],
        )
    }
}

#[pymethods]
impl BatchArena {
    #[new]
    #[pyo3(signature = (
        num_arenas,
        blue_cars = 1,
        orange_cars = 1,
        game_mode = "soccar",
        car_body = "octane",
        num_threads = None,
        seed = 0,
        mem_weight_mode = "balanced",
        tick_skip = 8,
        auto_reset = false,
        max_episode_ticks = None,
        scheduling = "stealing",
        pin_threads = false,
        pin_cpus = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        num_arenas: usize,
        blue_cars: usize,
        orange_cars: usize,
        game_mode: &str,
        car_body: &str,
        num_threads: Option<usize>,
        seed: u64,
        mem_weight_mode: &str,
        tick_skip: u32,
        auto_reset: bool,
        max_episode_ticks: Option<u64>,
        scheduling: &str,
        pin_threads: bool,
        pin_cpus: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let scheduling = match scheduling.to_ascii_lowercase().as_str() {
            "static" => Scheduling::Static,
            "stealing" | "work_stealing" => Scheduling::WorkStealing,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown scheduling {scheduling:?}; expected static or stealing"
                )));
            }
        };
        let game_mode = parse_game_mode(game_mode)?;
        if num_arenas == 0 {
            return Err(PyValueError::new_err("num_arenas must be > 0"));
        }
        if blue_cars + orange_cars == 0 {
            return Err(PyValueError::new_err("need at least one car per arena"));
        }
        if game_mode != GameMode::TheVoid && !rocketsim_batch::is_initialized() {
            return Err(PyRuntimeError::new_err(
                "collision meshes are not loaded; call rocketsim_batch.init(path) first \
                 (only game_mode='void' works without them)",
            ));
        }
        let config = BatchConfig::new(game_mode, num_arenas)
            .with_arena_config(
                ArenaConfig::new(game_mode).with_mem_weight_mode(parse_mem_mode(mem_weight_mode)?),
            )
            .with_teams(blue_cars, orange_cars)
            .with_car_body(parse_car_body(car_body)?)
            .with_num_threads(num_threads)
            .with_seed(seed)
            .with_tick_skip(tick_skip)
            .with_auto_reset(auto_reset)
            .with_max_episode_ticks(max_episode_ticks)
            .with_scheduling(scheduling)
            .with_pin_threads(pin_threads);
        let config = match pin_cpus {
            Some(cpus) => config.with_pin_cpus(cpus),
            None => config,
        };
        let inner = py.detach(|| RsBatchArena::new(config));
        Ok(Self { inner })
    }

    /// Apply `actions` (any array-like with `num_arenas * cars_per_arena * 8`
    /// values, e.g. shape `(N, C, 8)`), then simulate `ticks` ticks (default
    /// `tick_skip`) in every arena. The GIL is released while stepping.
    #[pyo3(signature = (actions = None, ticks = None))]
    fn step(
        &mut self,
        py: Python<'_>,
        actions: Option<&Bound<'_, PyAny>>,
        ticks: Option<u32>,
    ) -> PyResult<()> {
        if let Some(actions) = actions {
            let np = py.import("numpy")?;
            let arr = np
                .call_method1("ascontiguousarray", (actions, np.getattr("float32")?))?
                .cast_into::<numpy::PyArrayDyn<f32>>()?;
            let arr = arr.readonly();
            let expected = self.inner.actions().len();
            if arr.len() != expected {
                return Err(PyValueError::new_err(format!(
                    "expected {expected} action values (shape ({}, {}, {})), got shape {:?}",
                    self.inner.num_arenas(),
                    self.inner.cars_per_arena(),
                    layout::ACTION_DIM,
                    arr.shape()
                )));
            }
            self.inner.set_actions(arr.as_slice()?);
        }
        let ticks = ticks.unwrap_or(self.inner.config().tick_skip);
        let inner = &mut self.inner;
        py.detach(|| inner.step_ticks(ticks));
        Ok(())
    }

    /// Reset arenas to a random kickoff. `indices=None` resets all arenas;
    /// `seed` makes this reset reproducible.
    #[pyo3(signature = (indices = None, seed = None))]
    fn reset(
        &mut self,
        py: Python<'_>,
        indices: Option<Vec<usize>>,
        seed: Option<u64>,
    ) -> PyResult<()> {
        if let Some(indices) = &indices {
            for &i in indices {
                self.check_arena(i)?;
            }
        }
        let inner = &mut self.inner;
        py.detach(|| inner.reset(indices.as_deref(), seed));
        Ok(())
    }

    /// Teleport the ball of one arena.
    #[pyo3(signature = (arena, pos, vel = [0.0; 3], ang_vel = [0.0; 3]))]
    fn set_ball_state(
        &mut self,
        arena: usize,
        pos: [f32; 3],
        vel: [f32; 3],
        ang_vel: [f32; 3],
    ) -> PyResult<()> {
        self.check_arena(arena)?;
        let a = self.inner.arena_mut(arena);
        let mut state = *a.get_ball_state();
        state.phys.pos = vec3(pos);
        state.phys.vel = vec3(vel);
        state.phys.ang_vel = vec3(ang_vel);
        a.set_ball_state(state);
        self.inner.refresh_arena_obs(arena);
        Ok(())
    }

    /// Edit one car. Arguments left as `None` keep their current value.
    #[pyo3(signature = (arena, car, pos = None, vel = None, ang_vel = None, boost = None))]
    fn set_car_state(
        &mut self,
        arena: usize,
        car: usize,
        pos: Option<[f32; 3]>,
        vel: Option<[f32; 3]>,
        ang_vel: Option<[f32; 3]>,
        boost: Option<f32>,
    ) -> PyResult<()> {
        self.check_car(arena, car)?;
        let a = self.inner.arena_mut(arena);
        let mut state = *a.get_car_state(car);
        if let Some(pos) = pos {
            state.phys.pos = vec3(pos);
        }
        if let Some(vel) = vel {
            state.phys.vel = vec3(vel);
        }
        if let Some(ang_vel) = ang_vel {
            state.phys.ang_vel = vec3(ang_vel);
        }
        if let Some(boost) = boost {
            state.boost = boost;
        }
        a.set_car_state(car, state);
        self.inner.refresh_arena_obs(arena);
        Ok(())
    }

    /// Positions of every boost pad, shape `(P, 3)`, and whether each is big.
    fn boost_pad_layout<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let configs = self.inner.arenas()[0].get_all_boost_pad_configs();
        let pos: Vec<f32> = configs.iter().flat_map(|c| c.pos.to_array()).collect();
        let big: Vec<bool> = configs.iter().map(|c| c.is_big).collect();
        Ok((
            to_numpy(py, &pos, &[configs.len(), 3])?,
            to_numpy(py, &big, &[configs.len()])?,
        ))
    }

    #[getter]
    fn num_arenas(&self) -> usize {
        self.inner.num_arenas()
    }

    #[getter]
    fn cars_per_arena(&self) -> usize {
        self.inner.cars_per_arena()
    }

    #[getter]
    fn num_pads(&self) -> usize {
        self.inner.num_pads()
    }

    #[getter]
    fn num_threads(&self) -> usize {
        self.inner.num_threads()
    }

    #[getter]
    fn tick_skip(&self) -> u32 {
        self.inner.config().tick_skip
    }

    /// `(N, BALL_DIM)` float32.
    #[getter]
    fn ball_obs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_numpy(
            py,
            self.inner.ball_obs(),
            &[self.inner.num_arenas(), layout::BALL_DIM],
        )
    }

    /// `(N, C, CAR_DIM)` float32.
    #[getter]
    fn car_obs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = [
            self.inner.num_arenas(),
            self.inner.cars_per_arena(),
            layout::CAR_DIM,
        ];
        to_numpy(py, self.inner.car_obs(), &shape)
    }

    /// `(N, P)` float32, `1.0` = pad active.
    #[getter]
    fn pad_obs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_numpy(
            py,
            self.inner.pad_obs(),
            &[self.inner.num_arenas(), self.inner.num_pads()],
        )
    }

    /// `(N, BALL_DIM)` ball before an auto-reset (valid where `done`).
    #[getter]
    fn final_ball_obs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_numpy(
            py,
            self.inner.final_ball_obs(),
            &[self.inner.num_arenas(), layout::BALL_DIM],
        )
    }

    /// `(N, C, CAR_DIM)` cars before an auto-reset (valid where `done`).
    #[getter]
    fn final_car_obs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = [
            self.inner.num_arenas(),
            self.inner.cars_per_arena(),
            layout::CAR_DIM,
        ];
        to_numpy(py, self.inner.final_car_obs(), &shape)
    }

    /// `(N, C, ACTION_DIM)` float32, the actions applied on the last step.
    #[getter]
    fn actions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let shape = [
            self.inner.num_arenas(),
            self.inner.cars_per_arena(),
            layout::ACTION_DIM,
        ];
        to_numpy(py, self.inner.actions(), &shape)
    }

    /// `(N,)` int8: scoring team this step (0 blue, 1 orange) or -1.
    #[getter]
    fn goal<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.per_arena(py, &self.inner.info().goal)
    }

    /// `(N,)` bool: episode ended by a goal this step.
    #[getter]
    fn terminated<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v: Vec<bool> = self
            .inner
            .info()
            .terminated
            .iter()
            .map(|&x| x != 0)
            .collect();
        self.per_arena(py, &v)
    }

    /// `(N,)` bool: episode hit `max_episode_ticks` this step.
    #[getter]
    fn truncated<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let v: Vec<bool> = self
            .inner
            .info()
            .truncated
            .iter()
            .map(|&x| x != 0)
            .collect();
        self.per_arena(py, &v)
    }

    /// `(N,)` uint32: ticks simulated this step.
    #[getter]
    fn ticks_simulated<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.per_arena(py, &self.inner.info().ticks_simulated)
    }

    /// `(N,)` uint64: ticks in the current (or just-ended) episode.
    #[getter]
    fn episode_ticks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.per_arena(py, &self.inner.info().episode_ticks)
    }

    /// `(N, C)` uint32: ball touches per car this step.
    #[getter]
    fn ball_touches<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.per_car(py, &self.inner.info().ball_touches)
    }

    /// `(N, C)` uint32: demos inflicted per car this step.
    #[getter]
    fn demos<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.per_car(py, &self.inner.info().demos)
    }

    /// `(N, C)` uint32: times each car was demolished this step.
    #[getter]
    fn demoed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.per_car(py, &self.inner.info().demoed)
    }

    /// `(N, C)` uint32: boost pads picked up per car this step.
    #[getter]
    fn boost_pickups<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.per_car(py, &self.inner.info().boost_pickups)
    }

    /// Dict of `ball`, `cars`, `pads` observation arrays.
    fn get_obs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("ball", self.ball_obs(py)?)?;
        d.set_item("cars", self.car_obs(py)?)?;
        d.set_item("pads", self.pad_obs(py)?)?;
        Ok(d)
    }

    /// Dict of every per-step info array.
    fn get_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("goal", self.goal(py)?)?;
        d.set_item("terminated", self.terminated(py)?)?;
        d.set_item("truncated", self.truncated(py)?)?;
        d.set_item("ticks_simulated", self.ticks_simulated(py)?)?;
        d.set_item("episode_ticks", self.episode_ticks(py)?)?;
        d.set_item("ball_touches", self.ball_touches(py)?)?;
        d.set_item("demos", self.demos(py)?)?;
        d.set_item("demoed", self.demoed(py)?)?;
        d.set_item("boost_pickups", self.boost_pickups(py)?)?;
        d.set_item("final_ball", self.final_ball_obs(py)?)?;
        d.set_item("final_cars", self.final_car_obs(py)?)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        let cfg = self.inner.config();
        format!(
            "BatchArena(num_arenas={}, blue_cars={}, orange_cars={}, game_mode={:?}, num_threads={}, tick_skip={})",
            self.inner.num_arenas(),
            cfg.blue_cars,
            cfg.orange_cars,
            cfg.arena.game_mode.name(),
            self.inner.num_threads(),
            cfg.tick_skip
        )
    }
}

/// Load RocketSim collision meshes from a folder with `soccar/`, `hoops/`
/// and `dropshot/` subfolders of `.cmf` files. Only the first call per
/// process has an effect.
#[pyfunction]
#[pyo3(signature = (collision_meshes_path = "./collision_meshes", silent = true))]
fn init(py: Python<'_>, collision_meshes_path: &str, silent: bool) -> PyResult<()> {
    py.detach(|| rocketsim_batch::init(collision_meshes_path, silent))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Whether collision meshes have been loaded.
#[pyfunction]
fn is_initialized() -> bool {
    rocketsim_batch::is_initialized()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BatchArena>()?;
    m.add_function(wrap_pyfunction!(init, m)?)?;
    m.add_function(wrap_pyfunction!(is_initialized, m)?)?;
    m.add("BALL_DIM", layout::BALL_DIM)?;
    m.add("CAR_DIM", layout::CAR_DIM)?;
    m.add("ACTION_DIM", layout::ACTION_DIM)?;
    m.add("NO_GOAL", rocketsim_batch::NO_GOAL)?;
    m.add("TICK_RATE", rocketsim_batch::rocketsim::consts::TICK_RATE)?;
    Ok(())
}
