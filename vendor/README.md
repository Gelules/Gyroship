# Vendored dependencies

## naga

`naga/` is naga 30.0.1, the shader translator wgpu uses (MIT OR Apache-2.0; the
licence files are in the folder), with one change, marked "Gyroship patch" in
`src/front/mod.rs`:

- `Typifier::grow` starts from the first unresolved expression. It used
  `iter().skip(n)`, which walked the `n` expressions already resolved on every
  call, so typing a function was quadratic in its expression count. The large
  translated pixel programs (some 15,000 lines) took seconds each to create.

Gyroship's shader cache stores naga modules (`crates/gpu/src/shader/cache.rs`) and
stamps its entries with a hash of this folder (`crates/gpu/build.rs`). Hence two
more rules:

- `naga-types`, which defines some of the stored types, is pinned (`=30.0.1` in
  `Cargo.toml`): a newer one changes this folder, so the stamp, before entries are read.
- Cargo reruns the stamp's build script only for files newer than its last run:
  after replacing this folder with files that keep older dates, touch them.

Both the repository's workspace and the generated game workspace use it through
`[patch.crates-io]`. Remove the folder and those patch sections once a naga
release used by wgpu has the same fix. Then replace `../../vendor/naga` in
`crates/gpu/build.rs` with exact pins of `naga` and `naga-types` in
`crates/gpu/Cargo.toml`, which the stamp hashes: without them, a naga upgrade would
keep serving entries made with the older naga.
