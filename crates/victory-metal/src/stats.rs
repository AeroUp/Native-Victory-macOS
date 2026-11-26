//! What the backend allocates, for the frame log: the figures Metal's HUD shows but which that
//! HUD cannot split by kind (textures, heaps, upload buffers, and title memory wrapped without a
//! copy). Counters are relaxed atomics: they are read for a log line, not for control flow.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use victory_backend::resources::{TexDim, TextureDesc};

/// Host textures by size.
pub static TEXTURE_BYTES: AtomicU64 = AtomicU64::new(0);
pub static TEXTURE_COUNT: AtomicU64 = AtomicU64::new(0);
/// Descriptor-heap buffers.
pub static HEAP_BYTES: AtomicU64 = AtomicU64::new(0);
/// Buffers made to upload memory into a host texture.
pub static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);
/// Title memory wrapped by `newBufferWithBytesNoCopy`: counted by Metal as allocated, but it is
/// the title's own memory, not a copy.
pub static WRAPPED_BYTES: AtomicU64 = AtomicU64::new(0);
pub static WRAPPED_COUNT: AtomicU64 = AtomicU64::new(0);

/// What a host texture of `d` costs: level 0 over its slices, and at most a third more for the
/// mip chain.
pub fn texture_bytes(d: &TextureDesc) -> u64 {
    let depth = if d.dim == TexDim::D3 {
        d.depth_or_layers as u64
    } else {
        1
    };
    let layers = if d.dim == TexDim::D2Array {
        d.depth_or_layers as u64
    } else {
        1
    };
    let (block, per_block) = d.format.block();
    let blocks = (d.width as u64).div_ceil(block as u64)
        * (d.height as u64).div_ceil(block as u64)
        * depth
        * layers;
    let level = blocks * per_block as u64;
    if d.mips > 1 {
        level + level / 3
    } else {
        level
    }
}

pub fn texture_created(d: &TextureDesc) {
    TEXTURE_BYTES.fetch_add(texture_bytes(d), Relaxed);
    TEXTURE_COUNT.fetch_add(1, Relaxed);
}

pub fn texture_destroyed(bytes: u64) {
    TEXTURE_BYTES.fetch_sub(bytes, Relaxed);
    TEXTURE_COUNT.fetch_sub(1, Relaxed);
}

/// `(textures, bytes, heaps, uploads, wrapped)` in bytes, for the frame log.
pub fn snapshot() -> (u64, u64, u64, u64, u64) {
    (
        TEXTURE_COUNT.load(Relaxed),
        TEXTURE_BYTES.load(Relaxed),
        HEAP_BYTES.load(Relaxed),
        UPLOAD_BYTES.load(Relaxed),
        WRAPPED_BYTES.load(Relaxed),
    )
}
