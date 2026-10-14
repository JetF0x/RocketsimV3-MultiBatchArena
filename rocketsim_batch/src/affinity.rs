//! Picking and applying CPU pins for the batch's thread pool.

use core_affinity::CoreId;

/// Logical CPUs this process may run on, ascending.
fn allowed_cpus() -> Vec<usize> {
    let mut ids: Vec<usize> = core_affinity::get_core_ids()
        .unwrap_or_default()
        .into_iter()
        .map(|c| c.id)
        .collect();
    if ids.is_empty() {
        ids = (0..num_cpus::get()).collect();
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// First logical CPU of every physical core, read from sysfs.
#[cfg(target_os = "linux")]
fn first_sibling_per_core() -> Option<Vec<usize>> {
    let mut firsts = Vec::new();
    for entry in std::fs::read_dir("/sys/devices/system/cpu").ok()? {
        let name = entry.ok()?.file_name();
        let Some(idx) = name.to_str()?.strip_prefix("cpu") else {
            continue;
        };
        if idx.parse::<usize>().is_err() {
            continue;
        }
        let path = format!("/sys/devices/system/cpu/cpu{idx}/topology/thread_siblings_list");
        let Ok(list) = std::fs::read_to_string(path) else {
            continue;
        };
        // Formats: "3", "0,16", "0-1".
        let first = list.trim().split([',', '-']).next()?.parse().ok()?;
        firsts.push(first);
    }
    firsts.sort_unstable();
    firsts.dedup();
    (!firsts.is_empty()).then_some(firsts)
}

/// Windows numbers SMT siblings next to each other (0/1, 2/3, ...), so with
/// SMT on, the even CPUs are one per physical core.
#[cfg(not(target_os = "linux"))]
fn first_sibling_per_core() -> Option<Vec<usize>> {
    let (physical, logical) = (num_cpus::get_physical(), num_cpus::get());
    if cfg!(windows) && physical * 2 == logical {
        Some((0..logical).step_by(2).collect())
    } else {
        None
    }
}

/// One logical CPU per physical core (the first SMT sibling of each), limited
/// to the CPUs this process is allowed to use. Falls back to every allowed
/// logical CPU when the topology is unknown.
///
/// This is the default pin list for `BatchConfig::pin_threads`.
#[must_use]
pub fn physical_core_cpus() -> Vec<usize> {
    let allowed = allowed_cpus();
    match first_sibling_per_core() {
        Some(firsts) => {
            let cpus: Vec<usize> = firsts.into_iter().filter(|c| allowed.contains(c)).collect();
            if cpus.is_empty() { allowed } else { cpus }
        }
        None => allowed,
    }
}

/// Pins the calling thread to `cpu`. Logs and carries on if the OS refuses
/// (e.g. macOS, which has no hard affinity).
pub(crate) fn pin_current_thread(cpu: usize) {
    if !core_affinity::set_for_current(CoreId { id: cpu }) {
        log::warn!("rocketsim_batch: could not pin thread to logical CPU {cpu}");
    }
}
