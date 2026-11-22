#!/bin/sh
# Makes the files of a Gyroship release (D112; docs/building.md, "Making a release"), in Git Bash,
# from the repository's root, at the release's tag: the tools and the launcher built without this
# machine's paths, then, in artifacts/release/<version>:
#   gyroship-<version>-windows-x64.zip  the programs, at its top: gyroship-launcher.exe,
#                                       gyroship-tools.exe, SDL3.dll, LICENSE (Gyroship's) and
#                                       SDL3-LICENSE.txt (SDL's)
#   gyroship-<version>-source.zip       the repository at tag v<version>, in gyroship-<version>/
#   SHA256SUMS.txt                      both archives' SHA-256, as sha256sum writes them
# It publishes nothing: it prints the command that does.
# usage: sh tools/release.sh <version> <SDL3 folder>
#   <SDL3 folder>: SDL3.dll and LICENSE.txt, from the win32-x64 zip of an SDL 3 release.
set -eu
usage="usage: sh tools/release.sh <version> <SDL3 folder>"
version=${1:?$usage}
sdl=${2:?$usage}
tag="v$version"
fail() {
  echo "release: $*" >&2
  exit 1
}

[ -f crates/runtime/Cargo.toml ] || fail "run it from the repository's root"
[ -f "$sdl/SDL3.dll" ] && [ -f "$sdl/LICENSE.txt" ] || fail "$sdl holds no SDL3.dll and LICENSE.txt"
grep -qx "version = \"$version\"" crates/launcher/Cargo.toml || fail "crates/launcher/Cargo.toml does not say version = \"$version\": set it, commit, then tag $tag"
[ "$(git rev-parse -q --verify "refs/tags/$tag^{commit}" || true)" = "$(git rev-parse HEAD)" ] || fail "tag $tag is not this commit: commit, then tag it"
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "the working tree has changes: the programs would not be the tag's"

# Built in target folders of their own, so that the usual builds stay as they are, and without this
# machine's paths, which panic messages would otherwise carry (encoded flags: a path may hold spaces).
root=$(cygpath -w "$PWD")
home=$(cygpath -w "${CARGO_HOME:-$USERPROFILE/.cargo}")
CARGO_ENCODED_RUSTFLAGS=$(printf '%s\037%s' "--remap-path-prefix=$root=." "--remap-path-prefix=$home=cargo")
export CARGO_ENCODED_RUSTFLAGS
cargo build --release -j 4 --target-dir target/dist
(cd crates/launcher && cargo build --release -j 4 --target-dir target/dist)
launcher=crates/launcher/target/dist/release/gyroship-launcher.exe
tools=target/dist/release/gyroship-tools.exe
for exe in "$launcher" "$tools"; do
  for path in "$root" "$(cygpath -m "$PWD")" "$home" "$USERPROFILE"; do
    if grep -qaF "$path" "$exe"; then
      fail "$exe still holds $path"
    fi
  done
done

out="artifacts/release/$version"
programs="gyroship-$version-windows-x64.zip"
source="gyroship-$version-source.zip"
rm -rf "$out"
mkdir -p "$out/programs"
cp "$launcher" "$tools" "$sdl/SDL3.dll" LICENSE "$out/programs/"
cp "$sdl/LICENSE.txt" "$out/programs/SDL3-LICENSE.txt"
# Windows' own tar writes the zip, as the launcher's own tar reads it back.
"$(cygpath "$SYSTEMROOT")/System32/tar.exe" -a -c -f "$out/$programs" -C "$out/programs" gyroship-launcher.exe gyroship-tools.exe SDL3.dll LICENSE SDL3-LICENSE.txt
rm -rf "$out/programs"
git archive --format=zip --prefix="gyroship-$version/" -o "$out/$source" "$tag"
(
  cd "$out"
  for f in "$programs" "$source"; do
    printf '%s  %s\n' "$(sha256sum < "$f" | cut -d' ' -f1)" "$f"
  done > SHA256SUMS.txt
)
ls -l "$out"
cat "$out/SHA256SUMS.txt"
echo "Publish it, from the repository's root:"
echo "  gh release create $tag $out/$programs $out/$source $out/SHA256SUMS.txt --verify-tag --title \"Gyroship $version\" --notes \"What changed, for the players.\""
