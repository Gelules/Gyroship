# Vendored dependencies

## naga

`naga/` is naga 30.0.1, the shader translator wgpu uses (MIT OR Apache-2.0; the
licence files are in the folder), with one change, marked "Gyroship patch" in
`src/front/mod.rs`:

- `Typifier::grow` starts from the first unresolved expression. It used
  `iter().skip(n)`, which walked the `n` expressions already resolved on every
  call, so typing a function was quadratic in its expression count. The large
  translated pixel programs (some 15,000 lines) took seconds each to create.

Both the repository's workspace and the generated game workspace use it through
`[patch.crates-io]`. Remove the folder and those patch sections once a naga
release used by wgpu has the same fix.
