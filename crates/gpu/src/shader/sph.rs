//! The shader program header (SPH, 32 words) of a graphics program: which
//! attributes a stage reads and writes, how a pixel shader interpolates its
//! inputs and which render targets it writes. Bit positions from NVIDIA's
//! public SPH documentation (open-gpu-doc `Shader-Program-Header`, with the
//! Turing+ 32-word layout as Mesa NAK's `sph.rs` uses it).

use crate::sass::Stage;

/// How a pixel shader input component is interpolated (2 bits each).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Interp {
    Unused,
    /// Flat: the provoking vertex's value.
    Constant,
    Perspective,
    /// Linear in screen space.
    ScreenLinear,
}

#[derive(Clone, Copy, Debug)]
pub struct Sph {
    pub stage: Stage,
    words: [u32; 32],
}

impl Sph {
    /// No header: compute programs (their QMD describes them).
    pub fn empty(stage: Stage) -> Sph {
        Sph { stage, words: [0; 32] }
    }

    pub fn new(stage: Stage, code: &[u8]) -> Option<Sph> {
        let mut words = [0u32; 32];
        for (w, b) in words.iter_mut().zip(code.get(..128)?.as_chunks::<4>().0) {
            *w = u32::from_le_bytes(*b);
        }
        Some(Sph { stage, words })
    }

    fn bits(&self, lo: usize, len: usize) -> u32 {
        (0..len).fold(0, |v, i| v | ((self.words[(lo + i) / 32] >> ((lo + i) % 32)) & 1) << i)
    }

    /// Vertex-like stages: the components (mask, bit 0 = x) read of each
    /// generic attribute vector (bits 192..320).
    pub fn inputs(&self) -> [u8; 32] {
        std::array::from_fn(|v| self.bits(192 + 4 * v, 4) as u8)
    }

    /// Vertex-like stages: the components written of each generic output
    /// vector (bits 432..560).
    pub fn outputs(&self) -> [u8; 32] {
        std::array::from_fn(|v| self.bits(432 + 4 * v, 4) as u8)
    }

    /// Pixel: the interpolation of each component of each generic input
    /// vector (bits 192..448, 2 bits per component).
    pub fn interpolation(&self) -> [[Interp; 4]; 32] {
        std::array::from_fn(|v| std::array::from_fn(|c| [Interp::Unused, Interp::Constant, Interp::Perspective, Interp::ScreenLinear][self.bits(192 + 8 * v + 2 * c, 2) as usize]))
    }

    /// Pixel: the components written of each of the 8 color targets (bits 576..608).
    pub fn targets(&self) -> [u8; 8] {
        std::array::from_fn(|t| self.bits(576 + 4 * t, 4) as u8)
    }

    /// Pixel: whether the shader writes the sample mask and the depth.
    pub fn writes_sample_mask(&self) -> bool {
        self.bits(608, 1) == 1
    }

    pub fn writes_depth(&self) -> bool {
        self.bits(609, 1) == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_read_where_the_documentation_puts_them() {
        let mut code = [0u8; 128];
        let mut put = |word: usize, v: u32| code[4 * word..4 * word + 4].copy_from_slice(&v.to_le_bytes());
        // Vertex: inputs x,y of vectors 0 and 1 (word 6), output vector 0 x,y (word 13 bits 16, 17).
        put(6, 0x33);
        put(13, 0x3_3000);
        let s = Sph::new(Stage::Vertex, &code).unwrap();
        assert_eq!((s.inputs()[0], s.inputs()[1], s.outputs()[0]), (3, 3, 3));
        // Pixel: vector 0 x,y perspective (word 6 = 0xa), target 0 red only, depth.
        let mut code = [0u8; 128];
        let mut put = |word: usize, v: u32| code[4 * word..4 * word + 4].copy_from_slice(&v.to_le_bytes());
        put(6, 0xa);
        put(18, 1);
        put(19, 2);
        let s = Sph::new(Stage::Pixel, &code).unwrap();
        assert_eq!(s.interpolation()[0], [Interp::Perspective, Interp::Perspective, Interp::Unused, Interp::Unused]);
        assert_eq!((s.targets()[0], s.writes_sample_mask(), s.writes_depth()), (1, false, true));
    }
}
