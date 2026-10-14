# Vendored upstream

`vendor/rocketsim/` is the `rocketsim` crate from RocketSim v3:

- Repository: https://github.com/ZealanL/RocketSim
- Branch: `v3-rust`
- Commit: `6cf55d3` ("Bump version to 0.2.11"), 2026-10-13
- Crate version: `0.2.11` (also on crates.io as `rocketsim`)
- License: MIT (see `vendor/rocketsim/LICENSE`)

Only the `rocketsim/` crate was copied. The upstream `rocketsim_test` (benchmarks and replay metric, which need the v2 C++
bindings) and `rocketsim_vis` (visualizer) crates were left out.

## Local changes

- `Cargo.toml`: `readme` points at `UPSTREAM_README.md`, which is a copy of the upstream root `README.md`. Everything
  else is unchanged. The `*.workspace = true` fields resolve against this repo's root `Cargo.toml`, which mirrors
  upstream's `[workspace.package]` and `[workspace.dependencies]`.

Source files are unmodified. Record any future patch here.

## Re-syncing

```sh
git clone --branch v3-rust https://github.com/ZealanL/RocketSim /tmp/rs-v3
rsync -a --delete --exclude Cargo.toml --exclude UPSTREAM_README.md --exclude LICENSE \
    /tmp/rs-v3/rocketsim/ vendor/rocketsim/
cp /tmp/rs-v3/LICENSE vendor/rocketsim/LICENSE
cp /tmp/rs-v3/README.md vendor/rocketsim/UPSTREAM_README.md
diff /tmp/rs-v3/rocketsim/Cargo.toml vendor/rocketsim/Cargo.toml   # merge dependency changes by hand
diff /tmp/rs-v3/Cargo.toml Cargo.toml                              # sync [workspace.package] / deps
cargo test --workspace
```

Then update the commit and version above.
