//! `--video FILE`: the presented images (from `--dump-from`, every
//! `--dump-every`) piped to ffmpeg as they come, H.264, to watch a replay.
//! Host output (game images: never committed), never journaled. Needs ffmpeg
//! in the PATH.

use std::io::Write;
use std::process::{Child, Command, Stdio};

pub struct Video {
    ffmpeg: Child,
    /// The size of the first image; images of another size are left out.
    pub size: (u32, u32),
    pub skipped: u64,
}

impl Video {
    /// ffmpeg reading `size` RGBA images at `fps` on its input, writing `path`.
    pub fn start(path: &std::path::Path, size: (u32, u32), fps: f64) -> Result<Video, String> {
        let (w, h) = size;
        let ffmpeg = Command::new("ffmpeg")
            .args(["-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", &format!("{w}x{h}"), "-framerate", &format!("{fps}"), "-i", "-"])
            .args(["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"])
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot start ffmpeg: {e}"))?;
        Ok(Video { ffmpeg, size, skipped: 0 })
    }

    /// One image, `size` RGBA rows; false once ffmpeg stopped reading.
    pub fn frame(&mut self, rgba: &[u8]) -> bool {
        self.ffmpeg.stdin.as_mut().is_some_and(|i| i.write_all(rgba).is_ok())
    }

    /// Closes ffmpeg's input and waits for it to write the file.
    pub fn finish(mut self) -> Result<(), String> {
        drop(self.ffmpeg.stdin.take());
        let status = self.ffmpeg.wait().map_err(|e| e.to_string())?;
        if status.success() { Ok(()) } else { Err(format!("ffmpeg ended with {status}")) }
    }
}
