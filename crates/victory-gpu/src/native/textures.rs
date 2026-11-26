//! Title textures on the native host: the console's tiled layouts (`victory_formats::texture`)
//! detiled from title memory into backend textures.
//!
//! A title texture lives in title memory in the console's swizzled layout; the backend's texture is
//! Apple's own. Each texture the host samples is detiled into a linear staging buffer and uploaded,
//! mip by mip and slice by slice, when the memory's content changed since the last time (a hash of
//! the memory, taken when work is submitted). Render targets and depth targets are the GPU's: they
//! start empty and are not read from memory.
//!
//! shortcut: the detile runs on the CPU (`Layout::detile`, the code the D3D12 host uses too) and
//! a thread hashes the memory of every sampled texture round after round (see `Watch`); a compute detile that reads the
//! pool buffer directly and a per-page change test (what `watch.rs` gives the D3D12 host) are the
//! next step. A change detiles the whole texture.

use super::descriptors::format;
use super::title::Title;
use std::collections::HashMap;
use std::rc::Rc;
use victory_backend::resources::*;
use victory_formats::texture::{Dimension, Layout};

/// What realized host textures cost, in bytes of title memory they mirror (a close estimate of
/// their host size) and how many there are. Logged with the frame counters so a leak or an
/// unbounded world-streaming set shows up before the machine swaps.
pub static TEXTURE_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TEXTURE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TARGET_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PENDING_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TEXTURE_EVICTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Realized textures whose title memory is no longer mapped, and their bytes.
pub static UNMAPPED_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static UNMAPPED_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Textures whose memory was entirely zero at their first upload (content not streamed in yet).
pub static BLANK_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static BLANK_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Textures bound to a placeholder because their content had not arrived (VICTORY_TEXTURE_LAZY).
pub static LAZY_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `D3D12_RESOURCE_FLAGS`.
const RENDER_TARGET: u32 = 1;
const DEPTH_STENCIL: u32 = 2;
const UNORDERED_ACCESS: u32 = 4;

/// A title texture as the title made it: its description and where its memory is.
#[derive(Clone)]
pub struct TextureInfo {
    pub dimension: Dimension,
    /// DXGI format number, the console's (before `pc_format`).
    pub format: u32,
    pub width: u32,
    pub height: u32,
    /// Array length, or depth for 3D.
    pub depth_or_layers: u32,
    pub flags: u32,
    /// The layout of plane 0 (depth, for depth and stencil).
    pub layout: Layout,
    pub memory: *const u8,
    pub size: usize,
}

impl TextureInfo {
    pub fn render_target(&self) -> bool {
        self.flags & (RENDER_TARGET | DEPTH_STENCIL) != 0
    }

    /// The backend's description of it; `None` for formats or shapes the host does not know.
    pub fn desc(&self) -> Option<TextureDesc> {
        let depth = self.flags & DEPTH_STENCIL != 0;
        let mut format = format(self.format)?;
        // A typeless 32-bit float is a depth target when the texture is one.
        if depth && format == Format::R32Float {
            format = Format::D32Float;
        }
        if depth && format == Format::R16Unorm {
            format = Format::D16Unorm;
        }
        let (dim, d) = match self.dimension {
            Dimension::Tex3d => (TexDim::D3, self.depth_or_layers as usize),
            _ if self.depth_or_layers > 1 => (TexDim::D2Array, self.depth_or_layers as usize),
            _ => (TexDim::D2, 1),
        };
        Some(TextureDesc {
            dim,
            format,
            width: self.width as usize,
            height: self.height.max(1) as usize,
            depth_or_layers: d,
            mips: self.layout.mips.len().max(1),
            render_target: self.render_target(),
            unordered_access: self.flags & UNORDERED_ACCESS != 0,
        })
    }
}

/// Reads `len` bytes of title memory (zeros where it cannot be read: a page the title never mapped).
pub fn read_memory(at: *const u8, len: usize) -> Vec<u8> {
    const CHUNK: usize = 1 << 20;
    let mut out = vec![0u8; len];
    for (i, chunk) in out.chunks_mut(CHUNK).enumerate() {
        // Only an address, checked by the kernel when read.
        let _ = super::safe_read(at.wrapping_add(i * CHUNK), chunk);
    }
    out
}

/// Title texels as the host format holds them: DXGI `B4G4R4A4_UNORM` (115) has blue in the low
/// nibble and alpha in the top one, Metal's `ABGR4Unorm` alpha low and red top. The host's own
/// writes and reads agree with each other, so only bytes from title memory are converted.
pub fn to_host(format: u32, texels: &mut [u8]) {
    if format == 115 {
        for t in texels.as_chunks_mut::<2>().0 {
            *t = u16::from_le_bytes(*t).rotate_left(4).to_le_bytes();
        }
    }
}

/// Detiles every mip of every slice of `tiled` (the texture's whole memory) into `tex`.
pub fn upload<B: ResourceBackend>(
    backend: &B,
    tex: &B::Texture,
    info: &TextureInfo,
    tiled: &[u8],
) -> Result<(), String> {
    let layout = &info.layout;
    let slices = if info.dimension == Dimension::Tex3d {
        1
    } else {
        layout.slices
    };
    let n = layout.desc.element_bytes as usize;
    for mip in 0..layout.mips.len() as u32 {
        let m = &layout.mips[mip as usize];
        let row = m.width as usize * n;
        let slice_pitch = row * m.height as usize;
        let depth = layout.subresource_depth(mip).max(1) as usize;
        let mut linear = vec![0u8; slice_pitch * depth];
        for slice in 0..slices {
            layout
                .detile(tiled, mip, slice, &mut linear, row, slice_pitch)
                .map_err(|e| format!("detile mip {mip} slice {slice}: {e:?}"))?;
            to_host(info.format, &mut linear);
            backend.upload_texture(tex, mip as usize, slice as usize, &linear, row, slice_pitch);
        }
    }
    Ok(())
}

/// Textures whose memory the scanner thread watches, and those it found changed.
///
/// Hashing every sampled texture's memory on every submission (150 ms at 170 MiB, under the
/// title lock the title's other threads wait on) made a title that streams textures slow to a
/// crawl as they accumulated. A thread of its own hashes them in rounds, off the lock, and a
/// submission only looks at the textures it marked.
/// shortcut: a change is seen one round later (a round is as long as the hashing takes, about
/// 100 ms for 150 MiB), so a texture written just before it is sampled may show its old content
/// for a few frames; a per-page write test would make that exact.
#[derive(Default)]
struct Watch {
    /// Title texture key to (address, bytes, whether it is a render target: its first sight is
    /// its baseline, not a change).
    entries: HashMap<usize, (usize, usize, bool)>,
    dirty: std::collections::HashSet<usize>,
}

static WATCH: std::sync::Mutex<Option<Watch>> = std::sync::Mutex::new(None);

fn watch<R>(f: impl FnOnce(&mut Watch) -> R) -> R {
    f(WATCH.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

fn start_scanner() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("victory texture scan".into()).spawn(|| {
            let mut seen: HashMap<usize, blake3::Hash> = HashMap::new();
            let mut buf = vec![0u8; 1 << 20];
            let mut rounds = 0u64;
            let mut since = std::time::Instant::now();
            let (mut bytes, mut entries) = (0u64, 0usize);
            // The whole watched set is gigabytes once the world is loaded (16 000 textures), and
            // hashing all of it every round took seconds of memory bandwidth away from the title.
            // A quarter of it per round keeps the same latency at a fraction of the cost.
            let mut cursor = 0usize;
            loop {
                let list: Vec<(usize, (usize, usize, bool))> =
                    watch(|w| w.entries.iter().map(|(&k, &v)| (k, v)).collect());
                if !list.is_empty() {
                    let live: std::collections::HashSet<usize> =
                        list.iter().map(|&(k, _)| k).collect();
                    seen.retain(|k, _| live.contains(k));
                }
                if !list.is_empty() {
                    let batch = (list.len() / 4).max(1);
                    for i in 0..batch {
                        let (key, (at, size, target)) = list[(cursor + i) % list.len()];
                        let mut h = blake3::Hasher::new();
                        for off in (0..size).step_by(buf.len()) {
                            let n = buf.len().min(size - off);
                            buf[..n].fill(0);
                            let _ = super::safe_read((at + off) as *const u8, &mut buf[..n]);
                            h.update(&buf[..n]);
                        }
                        bytes += size as u64;
                        entries += 1;
                        let h = h.finalize();
                        let before = seen.insert(key, h);
                        if before != Some(h) && !(target && before.is_none()) {
                            watch(|w| w.dirty.insert(key));
                        }
                    }
                    cursor = (cursor + batch) % list.len();
                }
                rounds += 1;
                if since.elapsed().as_secs() >= 5 {
                    // The round's cost is the title's texture bandwidth: at 16 000 textures this
                    // is gigabytes per round, so the figure matters for the frame rate.
                    tracing::info!(
                        rounds,
                        entries,
                        gb = bytes as f64 / (1u64 << 30) as f64,
                        seconds = format_args!("{:.1}", since.elapsed().as_secs_f64()),
                        "texture scanner: hashed part of the watched set"
                    );
                    rounds = 0;
                    bytes = 0;
                    entries = 0;
                    since = std::time::Instant::now();
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        });
    });
}

/// A texture the host keeps for a title texture.
pub struct Tracked<B: ResourceBackend> {
    pub texture: Rc<B::Texture>,
    info: TextureInfo,
    /// What the host texture costs; 0 while its content has not arrived (a shared placeholder).
    bytes: u64,
    /// Bound to the shared placeholder, not a texture of its own.
    blank: bool,
    /// The hash of the memory last uploaded; `None` for targets and before the first upload.
    last: Option<blake3::Hash>,
    /// Uploads of its memory so far.
    pub uploads: u64,
    /// Submissions left in which its memory is probed at submission rather than left to the
    /// scanner (0: cold), and the hash of what the last probe read.
    hot: u32,
    probe: Option<blake3::Hash>,
}

/// Submissions a texture whose memory changed (or that was uploaded blank) stays hot without a
/// further change.
const HOT: u32 = 600;
/// Maximum hot textures probed synchronously on a single submission. Boot movies and UI glyph
/// caches have ~12 hot textures (all checked every submission); when world streaming realizes
/// thousands of blank textures at once, this bounds per-submission probing to O(1) while the
/// background scanner thread catches the rest off the host lock.
///
/// shortcut: a title write to a hot texture can take up to `ceil(hot / MAX_SYNC_HOT)`
/// submissions to be noticed synchronously (unless the scanner catches it first); raise this if
/// more than 32 textures need same-submission updates (e.g. many simultaneous streaming UI atlases).
const MAX_SYNC_HOT: usize = 32;
/// Pages of a large hot texture's memory a probe reads, spread over it.
const PROBE_PAGES: usize = 8;
/// Bytes sampled per page when probing a large hot texture.
const PROBE_SAMPLE: usize = 256;
/// Hot textures up to this size that changed more than once are probed whole (a glyph cache gets
/// a few glyphs at a time). Textures written once after a blank upload (streaming) are sampled:
/// probing all of them whole looked about 15% slower in one run.
const PROBE_WHOLE: usize = 4 << 20;

/// Hash of the bytes a probe of `size` bytes at `at` reads: all of them if `whole`, else
/// `PROBE_PAGES` samples of `PROBE_SAMPLE` bytes spread across the texture, reusing `scratch`.
///
/// shortcut: when sampled, a write that misses every sampled window goes unseen until the
/// scanner's next round; a texture the title rewrites whole (a movie's planes) is caught at once.
fn probe_hash(at: usize, size: usize, whole: bool, scratch: &mut Vec<u8>) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    if whole || size <= PROBE_PAGES * PROBE_SAMPLE {
        const CHUNK: usize = 1 << 20;
        let cap = size.min(CHUNK);
        if scratch.len() < cap {
            scratch.resize(cap, 0);
        }
        for off in (0..size).step_by(CHUNK) {
            let n = CHUNK.min(size - off);
            scratch[..n].fill(0);
            let _ = super::safe_read((at + off) as *const u8, &mut scratch[..n]);
            hasher.update(&scratch[..n]);
        }
        return hasher.finalize();
    }
    const PAGE: usize = 4096;
    let pages = size.div_ceil(PAGE);
    if scratch.len() < PROBE_SAMPLE {
        scratch.resize(PROBE_SAMPLE, 0);
    }
    for k in 0..PROBE_PAGES {
        let off = k * pages / PROBE_PAGES * PAGE;
        let n = PROBE_SAMPLE.min(size - off);
        scratch[..n].fill(0);
        let _ = super::safe_read((at + off) as *const u8, &mut scratch[..n]);
        hasher.update(&scratch[..n]);
    }
    hasher.finalize()
}

/// The key of a texture a view names by address and description (no resource object): a number no
/// object address has (the top bit is set).
pub fn placed_key(address: u64, description: &[u32; 14]) -> usize {
    let mut h = blake3::Hasher::new();
    h.update(&address.to_le_bytes());
    for (i, w) in description.iter().enumerate() {
        // Words 1 and 13 are padding (stack bytes in the title's records).
        h.update(&(if i == 1 || i == 13 { 0 } else { *w }).to_le_bytes());
    }
    let b = h.finalize();
    (u64::from_le_bytes(b.as_bytes()[..8].try_into().expect("8 bytes")) | 1 << 63) as usize
}

impl<B: PresentBackend> Title<B> {
    /// A title texture exists. Its host texture is made when a view, a render target or a clear
    /// first names it (most of a title's textures are never used), and filled from title memory
    /// when work next goes to the GPU.
    pub fn texture_created(&mut self, key: usize, info: TextureInfo) {
        if self.pending.insert(key, info).is_none() {
            PENDING_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Makes the host texture of title texture `key` if it has none yet and registers it for
    /// views. `false` when there is none (not a title texture, or one the host cannot represent).
    pub fn realize(&mut self, key: usize) -> bool {
        if self.textures.contains_key(&key) {
            return true;
        }
        let Some(info) = self.pending.get(&key).cloned() else {
            return false;
        };
        // A texture whose memory is in neither a pool nor this process's own address space is a
        // descriptor the title left behind: reading it yields zeros and a full-size host texture of
        // zeros is the largest waste the host has. It stays pending, so a later view tries again.
        // Some resources (the swapchain's 4K images) are allocated by the runtime in host memory
        // and have no title address at all: those read fine here and are kept.
        if self
            .host
            .memory()
            .resolve(info.memory as u64, info.size as u64)
            .is_none()
        {
            let mut probe = [0u8; 16];
            if !super::safe_read(info.memory, &mut probe) {
                UNMAPPED_BYTES.fetch_add(info.size as u64, std::sync::atomic::Ordering::Relaxed);
                UNMAPPED_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
                    tracing::info!(
                        key = format_args!("{key:#x}"),
                        at = format_args!("{:#x}", info.memory as usize),
                        len = info.size,
                        w = info.width,
                        h = info.height,
                        layers = info.depth_or_layers,
                        format = info.format,
                        flags = info.flags,
                        "a title texture's memory is not in a mapped pool"
                    );
                }
                return false;
            }
        }
        self.pending.remove(&key);
        PENDING_COUNT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        let Some(desc) = info.desc() else {
            // Loud once per format: views of it get no texture and their draws sample nothing.
            static SEEN: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if !seen.contains(&info.format) {
                seen.push(info.format);
                tracing::warn!(format = info.format, w = info.width, h = info.height, flags = info.flags, "a title texture format the host does not represent (first of this format)");
            }
            return false;
        };
        // A world load realizes thousands of textures whose content has not been streamed in yet
        // (all zeros in title memory). A full-size host texture of zeros costs the same as content
        // and shows the same (black), so with VICTORY_TEXTURE_LAZY they start on one small
        // placeholder per format and shape, and get their own texture when content arrives.
        let lazy = !desc.render_target && std::env::var_os("VICTORY_TEXTURE_LAZY").is_some();
        let (tex, blank) = if lazy {
            match self.placeholder_for(&desc) {
                Some(p) => (p, true),
                None => return false,
            }
        } else {
            match self.host.texture(&desc) {
                Ok(t) => (t, false),
                Err(e) => {
                    tracing::warn!(error = %e, ?desc, "no host texture");
                    return false;
                }
            }
        };
        if blank {
            LAZY_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // (The unmapped check above ran before the texture was made: an unmapped title texture
        // never gets here, so there is no second copy to drop.)
        self.host.register_texture(key, &tex);
        // Render targets too: what the title (CPU, DMA) writes into their memory is uploaded when
        // it changes, as it is not otherwise seen (a GPU write never changes title memory).
        watch(|w| w.entries.insert(key, (info.memory as usize, info.size, info.render_target())));
        start_scanner();
        // Debugging (VICTORY_MAGENTA=1): render targets start magenta, so a frame that stays so
        // was never drawn into, and a black one was drawn black.
        if info.render_target() && std::env::var_os("VICTORY_MAGENTA").is_some() {
            let m = &info.layout.mips[0];
            let n = info.layout.desc.element_bytes as usize;
            let row = m.width as usize * n;
            let pixel = [0xffu8, 0, 0xff, 0xff];
            let linear: Vec<u8> = (0..row * m.height as usize).map(|i| pixel[i % 4]).collect();
            self.host.backend.upload_texture(&tex, 0, 0, &linear, row, row * m.height as usize);
        }
        if !info.render_target() {
            self.unloaded.insert(key);
        }
        // The large ones, once each: what a texture budget has to think about.
        if info.size >= (8 << 20) {
            static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 20 {
                tracing::info!(
                    key = format_args!("{key:#x}"),
                    w = info.width,
                    h = info.height,
                    layers = info.depth_or_layers,
                    mips = info.layout.mips.len(),
                    format = info.format,
                    flags = info.flags,
                    mb = info.size as f64 / (1u64 << 20) as f64,
                    "a large title texture"
                );
            }
        }
        let bytes = if blank { 0 } else { info.size as u64 };
        if bytes > 0 {
            TEXTURE_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
            TEXTURE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if info.render_target() {
            TARGET_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        }
        self.textures.insert(
            key,
            Tracked {
                texture: tex,
                info,
                bytes,
                blank,
                last: None,
                uploads: 0,
                hot: 0,
                probe: None,
            },
        );
        true
    }

    /// The 1-texel (1 block for compressed formats) texture a title texture without content is
    /// bound to: one per format and shape, so a view of it has the shape the title's view asks
    /// for and Metal accepts the view.
    fn placeholder_for(&mut self, desc: &TextureDesc) -> Option<Rc<B::Texture>> {
        if let Some((_, _, t)) = self
            .placeholders
            .iter()
            .find(|(f, d, _)| *f == desc.format && *d == desc.dim)
        {
            return Some(t.clone());
        }
        let block = desc.format.block().0 as usize;
        let small = TextureDesc {
            dim: desc.dim,
            format: desc.format,
            width: block,
            height: block,
            depth_or_layers: 1,
            mips: 1,
            render_target: false,
            unordered_access: false,
        };
        let t = self.host.texture(&small).ok()?;
        self.placeholders.push((desc.format, desc.dim, t.clone()));
        Some(t)
    }

    /// The format of the host texture of title texture `key`.
    pub fn format_of(&self, key: usize) -> Option<Format> {
        self.textures.get(&key).and_then(|t| t.info.desc()).map(|d| d.format)
    }

    /// The host texture of title texture `key`.
    pub fn texture_of(&mut self, key: usize) -> Option<Rc<B::Texture>> {
        self.realize(key);
        self.textures.get(&key).map(|t| t.texture.clone())
    }

    /// How many times title texture `key`'s memory was detiled into its host texture.
    pub fn uploads_of(&self, key: usize) -> u64 {
        self.textures.get(&key).map_or(0, |t| t.uploads)
    }

    pub fn texture_released(&mut self, key: usize) {
        if let Some(t) = self.textures.remove(&key) {
            if t.bytes > 0 {
                TEXTURE_BYTES.fetch_sub(t.bytes, std::sync::atomic::Ordering::Relaxed);
                TEXTURE_COUNT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
            if t.info.render_target() {
                TARGET_BYTES.fetch_sub(t.bytes, std::sync::atomic::Ordering::Relaxed);
            }
        }
        if self.pending.remove(&key).is_some() {
            PENDING_COUNT.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.unloaded.remove(&key);
        self.hot_keys.retain(|&k| k != key);
        watch(|w| {
            w.entries.remove(&key);
            w.dirty.remove(&key);
        });
    }

    /// Brings the textures the host samples up to date with title memory: those whose memory
    /// changed since their last upload are detiled again.
    pub fn sync_textures(&mut self) {
        let keys: Vec<usize> = self
            .textures
            .iter()
            .filter(|(_, t)| !t.info.render_target())
            .map(|(&k, _)| k)
            .collect();
        self.sync_keys(keys);
    }

    /// What a submission does: the textures not uploaded yet, and those the scanner or hot probe
    /// found changed since.
    pub fn sync_dirty(&mut self) {
        let mut dirty = watch(|w| std::mem::take(&mut w.dirty));
        // Hot textures (a movie's planes, a glyph cache) are probed synchronously in O(1) bounded
        // batches so world streaming (thousands of blank textures) never stalls submissions.
        let mut budget = self.hot_keys.len().min(MAX_SYNC_HOT);
        while budget > 0 && !self.hot_keys.is_empty() {
            budget -= 1;
            let idx = self.hot_cursor % self.hot_keys.len();
            let key = self.hot_keys[idx];
            let Some(t) = self.textures.get_mut(&key) else {
                self.hot_keys.swap_remove(idx);
                continue;
            };
            if t.hot == 0 {
                self.hot_keys.swap_remove(idx);
                continue;
            }
            let whole = t.uploads >= 2 && t.info.size <= PROBE_WHOLE;
            let now = probe_hash(t.info.memory as usize, t.info.size, whole, &mut self.probe_buf);
            let before = t.probe.replace(now);
            if before != Some(now) {
                dirty.insert(key);
                self.hot_cursor = idx + 1;
            } else {
                t.hot -= 1;
                if t.hot == 0 {
                    t.probe = None;
                    self.hot_keys.swap_remove(idx);
                } else {
                    self.hot_cursor = idx + 1;
                }
            }
        }
        let unloaded = std::mem::take(&mut self.unloaded);
        if unloaded.is_empty() && dirty.is_empty() {
            return;
        }
        let mut todo: Vec<usize> = unloaded.into_iter().collect();
        for key in dirty {
            if !todo.contains(&key) {
                todo.push(key);
            }
        }
        self.sync_keys(todo);
    }

    fn sync_keys(&mut self, keys: Vec<usize>) {
        // A world load realizes thousands of textures at once; detiling and uploading all of them
        // in one submission holds the frame (and its fence wait) for seconds, which reads as a
        // stuck GPU. What does not fit waits for the next submission.
        const SYNC_MS: u128 = 8;
        const SYNC_BYTES: u64 = 48 << 20;
        let Title {
            host,
            textures,
            hot_keys,
            stats,
            ..
        } = self;
        let started = std::time::Instant::now();
        let mut done = 0usize;
        let mut bytes = 0u64;
        for (n, key) in keys.iter().enumerate() {
            if done > 0 && (bytes >= SYNC_BYTES || started.elapsed().as_millis() >= SYNC_MS) {
                let rest = &keys[n..];
                watch(|w| w.dirty.extend(rest.iter().copied()));
                tracing::info!(
                    done,
                    rest = rest.len(),
                    bytes,
                    ms = started.elapsed().as_millis(),
                    "texture upload budget spent: the rest waits for the next submission"
                );
                return;
            }
            let Some(t) = textures.get_mut(key) else {
                continue;
            };
            bytes += t.info.size as u64;
            done += 1;
            let tiled = read_memory(t.info.memory, t.info.size);
            let hash = blake3::hash(&tiled);
            if t.last == Some(hash) {
                continue;
            }
            let blank = tiled.iter().all(|&b| b == 0);
            if blank && t.uploads == 0 {
                BLANK_BYTES.fetch_add(t.info.size as u64, std::sync::atomic::Ordering::Relaxed);
                BLANK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if t.blank {
                // Still blank: the placeholder already samples what the memory holds (zeros).
                if !blank {
                    // Content arrived: its own texture, and every heap entry that names the key
                    // points at it from now on.
                    let Some(desc) = t.info.desc() else {
                        t.last = Some(hash);
                        continue;
                    };
                    match host.texture(&desc) {
                        Ok(real) => {
                            t.texture = real.clone();
                            t.blank = false;
                            t.bytes = t.info.size as u64;
                            TEXTURE_BYTES.fetch_add(t.bytes, std::sync::atomic::Ordering::Relaxed);
                            TEXTURE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            host.register_texture(*key, &real);
                            host.rewrite_key(*key);
                        }
                        Err(e) => {
                            tracing::warn!(texture = *key, error = %e, "no host texture for content that arrived");
                            t.last = Some(hash);
                            continue;
                        }
                    }
                } else {
                    t.last = Some(hash);
                    continue;
                }
            }
            // Written again after its first upload, or blank at it (written later): hot.
            if !t.info.render_target() && (t.uploads > 0 || blank) {
                if t.hot == 0 {
                    hot_keys.push(*key);
                }
                t.hot = HOT;
            }
            match upload(&host.backend, &t.texture, &t.info, &tiled) {
                Ok(()) => {
                    t.last = Some(hash);
                    t.uploads += 1;
                    stats.detiles += 1;
                }
                Err(e) => {
                    t.last = Some(hash);
                    stats.undetilable += 1;
                    tracing::warn!(texture = key, error = %e, "a title texture was not detiled");
                }
            }
        }
    }
}

/// The textures of a title host, by title resource key.
pub type Textures<B> = HashMap<usize, Tracked<B>>;

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn probe_hash_detects_writes_in_sampled_and_whole_modes() {
        let mut mem = vec![0u8; 16 * 4096];
        let mut scratch = Vec::new();
        let h0_sampled = probe_hash(mem.as_ptr() as usize, mem.len(), false, &mut scratch);
        let h0_whole = probe_hash(mem.as_ptr() as usize, mem.len(), true, &mut scratch);
        // Flip a byte in the first sampled window.
        mem[42] = 0xAB;
        let h1_sampled = probe_hash(mem.as_ptr() as usize, mem.len(), false, &mut scratch);
        let h1_whole = probe_hash(mem.as_ptr() as usize, mem.len(), true, &mut scratch);
        assert_ne!(h0_sampled, h1_sampled);
        assert_ne!(h0_whole, h1_whole);
        // Flip a byte outside sampled windows: whole mode catches it.
        mem[42] = 0;
        mem[1000] = 0xCD;
        assert_ne!(h0_whole, probe_hash(mem.as_ptr() as usize, mem.len(), true, &mut scratch));
    }
}
