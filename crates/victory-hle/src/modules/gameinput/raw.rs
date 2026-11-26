//! Raw Input (`VICTORY_RAW_INPUT`, on by default): a key is held from its
//! make message to its break, whatever polling would have seen in between,
//! and auto-repeats change nothing; the mouse reports its own counts, so the
//! cursor never has to be put back in the middle. A thread of ours owns a
//! message-only window that receives both (`RIDEV_INPUTSINK`); the poll
//! thread reads what it gathered ([`Shared`]).
//!
//! A remote desktop (TeamViewer) injects its input: it moves the mouse to
//! absolute positions, and may send a held key as a press and a release per
//! repeat. Absolute positions become movement here; a release of an injected
//! key waits for the key's grace (`VICTORY_KEY_GRACE`), so a press that
//! follows it in time keeps the key held.
#![cfg_attr(not(windows), allow(dead_code))]

/// Bits of a raw mouse report's `usFlags`.
const MOUSE_MOVE_ABSOLUTE: u16 = 0x01;
const MOUSE_VIRTUAL_DESKTOP: u16 = 0x02;
/// The span of absolute coordinates (0 to 65535 over the screen).
const ABSOLUTE_SPAN: f32 = 65535.0;
/// An absolute jump past this (a quarter of the screen) is the cursor coming
/// back elsewhere, out of the remote window and in again: no movement.
const ABSOLUTE_JUMP: i32 = 16384;

/// How close to an edge of the window (a share of its width or height) the
/// cursor is put back in the middle, while the mouse moves to absolute
/// positions: held in the window, it would stop at the edge.
const EDGE_MARGIN: f32 = 0.15;

/// Whether the cursor at `at` should go back to the middle of the window's
/// client area `rect` (left, top, right, bottom, on the screen): only with
/// `absolute` positions (a remote desktop), and near an edge.
pub(super) fn needs_recentering(
    absolute: bool,
    at: (i32, i32),
    rect: (i32, i32, i32, i32),
) -> bool {
    let (left, top, right, bottom) = rect;
    if !absolute || right <= left || bottom <= top {
        return false;
    }
    let mx = ((right - left) as f32 * EDGE_MARGIN) as i32;
    let my = ((bottom - top) as f32 * EDGE_MARGIN) as i32;
    at.0 < left + mx || at.0 >= right - mx || at.1 < top + my || at.1 >= bottom - my
}

/// The middle of `rect` (left, top, right, bottom).
pub(super) fn middle(rect: (i32, i32, i32, i32)) -> (i32, i32) {
    (
        rect.0 + (rect.2 - rect.0) / 2,
        rect.1 + (rect.3 - rect.1) / 2,
    )
}

/// Keys held, from their make and break messages.
#[derive(Debug, Default)]
pub(super) struct KeyTracker {
    /// How long a break of an injected key waits, in microseconds.
    grace: u64,
    keys: Vec<Held>,
}

#[derive(Clone, Copy, Debug)]
struct Held {
    scan_code: u32,
    virtual_key: u8,
    /// When its break takes effect (microseconds); none while held.
    release_at: Option<u64>,
    /// A poll has reported it: a key let go before any poll saw it is still
    /// reported once, so a short tap is never lost.
    seen: bool,
}

impl KeyTracker {
    pub fn new(grace: u64) -> KeyTracker {
        KeyTracker {
            grace,
            keys: Vec::new(),
        }
    }

    /// A make (`down`) or break of the key at `scan_code`, at `now`
    /// (microseconds); `injected` when a program (a remote desktop) sent it
    /// rather than a keyboard.
    pub fn key(&mut self, scan_code: u32, virtual_key: u8, down: bool, injected: bool, now: u64) {
        let index = self.keys.iter().position(|k| k.scan_code == scan_code);
        match (down, index) {
            // An auto-repeat, or a press again within the grace.
            (true, Some(i)) => self.keys[i].release_at = None,
            (true, None) => self.keys.push(Held {
                scan_code,
                virtual_key,
                release_at: None,
                seen: false,
            }),
            (false, Some(i)) => {
                let key = &mut self.keys[i];
                let at = if injected { now + self.grace } else { now };
                key.release_at = Some(key.release_at.map_or(at, |was| was.min(at)));
                if key.seen && key.release_at.is_some_and(|at| at <= now) {
                    self.keys.remove(i);
                }
            }
            (false, None) => {}
        }
    }

    /// The keys held at `now` (scan code, virtual key), in the order pressed.
    pub fn held(&mut self, now: u64) -> Vec<(u32, u8)> {
        let held = |k: &Held| k.release_at.is_none_or(|at| now < at);
        let out = self
            .keys
            .iter()
            .filter(|k| !k.seen || held(k))
            .map(|k| (k.scan_code, k.virtual_key))
            .collect();
        self.keys.retain(held);
        for key in &mut self.keys {
            key.seen = true;
        }
        out
    }

    /// Lets go of every key (focus lost: a break may never come).
    pub fn clear(&mut self) {
        self.keys.clear();
    }
}

/// The mouse's movement and buttons, from its reports.
#[derive(Debug, Default)]
pub(super) struct Motion {
    dx: f32,
    dy: f32,
    /// The last absolute position, while the reports are absolute.
    absolute: Option<(i32, i32)>,
    /// Put back in the middle since the last absolute report: still
    /// absolute, though no reference.
    recentering: bool,
    left: bool,
    right: bool,
    middle: bool,
}

impl Motion {
    /// A report: `flags` its `usFlags`, `x`, `y` its `lLastX`, `lLastY`;
    /// `size` is the width and height in pixels absolute coordinates span
    /// ([`absolute_size`]).
    pub fn moved(&mut self, flags: u16, x: i32, y: i32, size: (f32, f32)) {
        if flags & MOUSE_MOVE_ABSOLUTE == 0 {
            self.absolute = None;
            self.recentering = false;
            self.dx += x as f32;
            self.dy += y as f32;
            return;
        }
        self.recentering = false;
        if let Some((was_x, was_y)) = self.absolute.replace((x, y)) {
            let (dx, dy) = (x - was_x, y - was_y);
            if dx.abs() < ABSOLUTE_JUMP && dy.abs() < ABSOLUTE_JUMP {
                self.dx += dx as f32 * size.0 / ABSOLUTE_SPAN;
                self.dy += dy as f32 * size.1 / ABSOLUTE_SPAN;
            }
        }
    }

    /// Whether the last report gave an absolute position.
    pub fn absolute(&self) -> bool {
        self.absolute.is_some() || self.recentering
    }

    /// The cursor was put back in the middle: the next absolute position is
    /// the new reference, the jump to it no movement.
    pub fn recentered(&mut self) {
        self.absolute = None;
        self.recentering = true;
    }

    /// A report's button changes (`usButtonFlags`).
    pub fn buttons(&mut self, flags: u16) {
        for (down, up, button) in [
            (0x01, 0x02, &mut self.left),
            (0x04, 0x08, &mut self.right),
            (0x10, 0x20, &mut self.middle),
        ] {
            if flags & down != 0 {
                *button = true;
            }
            if flags & up != 0 {
                *button = false;
            }
        }
    }

    /// The movement since the last call and the buttons held.
    pub fn take(&mut self) -> super::keyboard::Mouse {
        let (dx, dy) = (std::mem::take(&mut self.dx), std::mem::take(&mut self.dy));
        super::keyboard::Mouse {
            dx,
            dy,
            left: self.left,
            right: self.right,
            middle: self.middle,
        }
    }

    /// Forgets the movement and lets go of the buttons.
    pub fn clear(&mut self) {
        *self = Motion {
            absolute: self.absolute,
            recentering: self.recentering,
            ..Motion::default()
        };
    }
}

/// Whether absolute coordinates of a report with `flags` span the virtual
/// desktop (every screen) rather than the main screen.
pub(super) fn spans_virtual_desktop(flags: u16) -> bool {
    flags & MOUSE_VIRTUAL_DESKTOP != 0
}

/// What the input thread gathered, read by the poll thread.
#[derive(Debug, Default)]
pub(super) struct Gathered {
    pub keys: KeyTracker,
    pub mouse: Motion,
}

#[cfg(windows)]
pub(super) use win::{Shared, start};

#[cfg(windows)]
mod win {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Mutex, OnceLock};

    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
    };
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{MAPVK_VK_TO_VSC_EX, MapVirtualKeyW};
    use windows_sys::Win32::UI::Input::{
        GetRawInputData, HRAWINPUT, RAWINPUT, RAWINPUTDEVICE, RAWINPUTHEADER, RID_INPUT,
        RIDEV_INPUTSINK, RIM_TYPEKEYBOARD, RIM_TYPEMOUSE, RegisterRawInputDevices,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DefWindowProcW, DispatchMessageW, GetMessageW, GetSystemMetrics,
        HWND_MESSAGE, MSG, RI_KEY_BREAK, RI_KEY_E0, RI_KEY_E1, RegisterClassW, SM_CXSCREEN,
        SM_CXVIRTUALSCREEN, SM_CYSCREEN, SM_CYVIRTUALSCREEN, WM_INPUT, WNDCLASSW,
    };

    use super::{Gathered, spans_virtual_desktop};
    use crate::modules::gameinput::lock;

    /// The input thread's findings, for the poll thread.
    pub(in crate::modules::gameinput) struct Shared {
        gathered: Mutex<Gathered>,
    }

    impl Shared {
        /// Keys held at `now` (scan code, virtual key).
        pub fn keys(&self, now: u64) -> Vec<(u32, u8)> {
            lock(&self.gathered).keys.held(now)
        }
        pub fn forget_keys(&self) {
            lock(&self.gathered).keys.clear();
        }
        /// The mouse since the last call.
        pub fn mouse(&self) -> crate::modules::gameinput::keyboard::Mouse {
            lock(&self.gathered).mouse.take()
        }
        pub fn forget_mouse(&self) {
            lock(&self.gathered).mouse.clear();
        }
        /// The mouse moves to absolute positions (a remote desktop).
        pub fn absolute(&self) -> bool {
            lock(&self.gathered).mouse.absolute()
        }
        /// Puts the cursor back (`put`), the jump it makes no movement;
        /// no report comes in between.
        pub fn recenter(&self, put: impl FnOnce()) {
            let mut gathered = lock(&self.gathered);
            put();
            gathered.mouse.recentered();
        }
    }

    static SHARED: OnceLock<Option<&'static Shared>> = OnceLock::new();

    /// Starts the input thread (once); none when Raw Input is unavailable.
    pub fn start(grace: u64) -> Option<&'static Shared> {
        *SHARED.get_or_init(|| {
            let shared: &'static Shared = Box::leak(Box::new(Shared {
                gathered: Mutex::new(Gathered {
                    keys: super::KeyTracker::new(grace),
                    ..Gathered::default()
                }),
            }));
            let (ready, wait) = std::sync::mpsc::channel();
            let spawned = std::thread::Builder::new()
                .name("victory raw input".into())
                .spawn(move || run(shared, ready));
            let ok = spawned.is_ok() && wait.recv().unwrap_or(false);
            if ok {
                tracing::info!("raw input: keyboard and mouse");
            } else {
                tracing::warn!("raw input unavailable: keys and mouse are polled");
            }
            ok.then_some(shared)
        })
    }

    /// The input thread's state, for its window procedure.
    static GATHER: OnceLock<&'static Shared> = OnceLock::new();
    static SAW_INJECTED: AtomicBool = AtomicBool::new(false);
    static SAW_ABSOLUTE: AtomicBool = AtomicBool::new(false);

    fn run(shared: &'static Shared, ready: std::sync::mpsc::Sender<bool>) {
        let _ = GATHER.set(shared);
        let class: Vec<u16> = "VictoryRawInput\0".encode_utf16().collect();
        // SAFETY: a class and a message-only window of our own, on this
        // thread; the registration names that window.
        let window = unsafe {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
            let instance = GetModuleHandleW(std::ptr::null());
            let wc = WNDCLASSW {
                lpfnWndProc: Some(window_proc),
                hInstance: instance,
                lpszClassName: class.as_ptr(),
                ..Default::default()
            };
            RegisterClassW(&wc);
            CreateWindowExW(
                0,
                class.as_ptr(),
                std::ptr::null(),
                0,
                0,
                0,
                0,
                0,
                HWND_MESSAGE,
                std::ptr::null_mut(),
                instance,
                std::ptr::null(),
            )
        };
        if window.is_null() {
            let _ = ready.send(false);
            return;
        }
        // Generic desktop: mouse (2) and keyboard (6), whichever window is
        // in front; the poll thread keeps them only while ours is.
        let devices = [2u16, 6].map(|usage| RAWINPUTDEVICE {
            usUsagePage: 1,
            usUsage: usage,
            dwFlags: RIDEV_INPUTSINK,
            hwndTarget: window,
        });
        // SAFETY: two filled-in registrations.
        let registered = unsafe {
            RegisterRawInputDevices(
                devices.as_ptr(),
                devices.len() as u32,
                size_of::<RAWINPUTDEVICE>() as u32,
            )
        } != 0;
        let _ = ready.send(registered);
        if !registered {
            return;
        }
        let mut msg = MSG::default();
        // SAFETY: this thread's messages.
        while unsafe { GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) } > 0 {
            // SAFETY: a message just received.
            unsafe { DispatchMessageW(&msg) };
        }
    }

    extern "system" fn window_proc(
        hwnd: HWND,
        message: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        if message == WM_INPUT {
            gather(lparam as HRAWINPUT);
        }
        // SAFETY: the default handling (WM_INPUT's cleanup among it).
        unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
    }

    fn gather(handle: HRAWINPUT) {
        let Some(shared) = GATHER.get() else {
            return;
        };
        let mut input = RAWINPUT::default();
        let mut size = size_of::<RAWINPUT>() as u32;
        // SAFETY: room for a keyboard or mouse report (a larger HID one fails).
        let read = unsafe {
            GetRawInputData(
                handle,
                RID_INPUT,
                (&raw mut input).cast(),
                &mut size,
                size_of::<RAWINPUTHEADER>() as u32,
            )
        };
        if read == u32::MAX || read == 0 {
            return;
        }
        // Input sent by a program (SendInput) comes from no device.
        let injected = input.header.hDevice.is_null();
        let now = crate::modules::gameinput::now();
        match input.header.dwType {
            RIM_TYPEKEYBOARD => {
                // SAFETY: a keyboard report.
                let k = unsafe { input.data.keyboard };
                let Some((scan_code, virtual_key)) = key_of(k.MakeCode, k.Flags, k.VKey) else {
                    return;
                };
                if injected && !SAW_INJECTED.swap(true, Ordering::Relaxed) {
                    tracing::info!(
                        "raw input: injected keys (a remote desktop); a release waits for VICTORY_KEY_GRACE"
                    );
                }
                let down = u32::from(k.Flags) & RI_KEY_BREAK == 0;
                lock(&shared.gathered)
                    .keys
                    .key(scan_code, virtual_key, down, injected, now);
            }
            RIM_TYPEMOUSE => {
                // SAFETY: a mouse report.
                let m = unsafe { input.data.mouse };
                // SAFETY: the button half of the union, as every report has.
                let buttons = unsafe { m.Anonymous.Anonymous.usButtonFlags };
                let size = if m.usFlags & super::MOUSE_MOVE_ABSOLUTE != 0 {
                    if !SAW_ABSOLUTE.swap(true, Ordering::Relaxed) {
                        tracing::info!(
                            injected,
                            "raw input: the mouse moves to absolute positions (a remote desktop)"
                        );
                    }
                    let (w, h) = if spans_virtual_desktop(m.usFlags) {
                        (SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN)
                    } else {
                        (SM_CXSCREEN, SM_CYSCREEN)
                    };
                    // SAFETY: plain queries.
                    unsafe { (GetSystemMetrics(w) as f32, GetSystemMetrics(h) as f32) }
                } else {
                    (0.0, 0.0)
                };
                let mut gathered = lock(&shared.gathered);
                gathered.mouse.moved(m.usFlags, m.lLastX, m.lLastY, size);
                gathered.mouse.buttons(buttons);
            }
            _ => {}
        }
    }

    /// A report's key (scan code, 0xE0xx extended; virtual key, left and
    /// right modifiers apart); none for the fake keys some sequences add.
    fn key_of(make: u16, flags: u16, vkey: u16) -> Option<(u32, u8)> {
        let flags = u32::from(flags);
        // 0xFF: part of an escape sequence; the shifts with E0: the fake
        // ones the keyboard adds around the arrows with Num Lock.
        if vkey == 0xFF || vkey == 0 || (flags & RI_KEY_E0 != 0 && matches!(make, 0x2A | 0x36)) {
            return None;
        }
        let vk = vkey as u8;
        let mut scan = u32::from(make);
        if scan == 0 {
            // SAFETY: plain query.
            scan = unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) };
        } else if flags & RI_KEY_E0 != 0 {
            scan |= 0xE000;
        } else if flags & RI_KEY_E1 != 0 {
            scan |= 0xE100;
        }
        let e0 = scan & 0xFF00 == 0xE000;
        let vk = match vk {
            0x10 if scan == 0x36 => 0xA1, // right Shift
            0x10 => 0xA0,
            0x11 if e0 => 0xA3, // right Ctrl
            0x11 => 0xA2,
            0x12 if e0 => 0xA5, // right Alt (AltGr)
            0x12 => 0xA4,
            other => other,
        };
        Some((scan, vk))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 0x11;
    const MS: u64 = 1_000;

    fn held(keys: &mut KeyTracker, now: u64) -> bool {
        keys.held(now).iter().any(|&(scan, _)| scan == W)
    }

    #[test]
    fn a_key_is_held_from_its_press_to_its_release_whatever_repeats() {
        let mut keys = KeyTracker::new(60 * MS);
        keys.key(W, b'Z', true, false, 0);
        // Polls every 8 ms for a second, with auto-repeats (after 250 ms,
        // every 33 ms) between them: held at every poll.
        let mut repeat = 250 * MS;
        for poll in 1..=125u64 {
            let now = poll * 8 * MS;
            while repeat <= now {
                keys.key(W, b'Z', true, false, repeat);
                repeat += 33 * MS;
            }
            assert!(held(&mut keys, now), "poll at {} ms", now / MS);
        }
        // The release from a keyboard counts at once.
        keys.key(W, b'Z', false, false, 1_001 * MS);
        assert!(!held(&mut keys, 1_002 * MS));
        assert!(keys.held(2_000 * MS).is_empty());
    }

    #[test]
    fn an_injected_key_sent_as_presses_and_releases_stays_held() {
        // A remote desktop sending the held key as a press and a release
        // every 33 ms.
        let mut keys = KeyTracker::new(60 * MS);
        let mut next = 0;
        for poll in 1..=125u64 {
            let now = poll * 8 * MS;
            while next <= now {
                keys.key(W, b'Z', true, true, next);
                keys.key(W, b'Z', false, true, next + MS);
                next += 33 * MS;
            }
            assert!(held(&mut keys, now), "poll at {} ms", now / MS);
        }
        // Once they stop, the key is let go after the grace.
        let last = next - 33 * MS + MS;
        assert!(held(&mut keys, last + 50 * MS));
        assert!(!held(&mut keys, last + 61 * MS));
        // Without a grace it flickers, as polling saw it.
        let mut bare = KeyTracker::new(0);
        bare.key(W, b'Z', true, true, 0);
        assert!(held(&mut bare, 0));
        bare.key(W, b'Z', false, true, MS);
        assert!(!held(&mut bare, 8 * MS));
    }

    #[test]
    fn a_tap_between_two_polls_is_seen_once() {
        let mut keys = KeyTracker::new(60 * MS);
        assert!(!held(&mut keys, 0));
        keys.key(W, b'Z', true, false, 2 * MS);
        keys.key(W, b'Z', false, false, 5 * MS);
        assert!(held(&mut keys, 8 * MS));
        assert!(!held(&mut keys, 16 * MS));
        // A release with no press, and focus lost while held.
        keys.key(W, b'Z', false, false, 20 * MS);
        assert!(!held(&mut keys, 24 * MS));
        keys.key(W, b'Z', true, false, 30 * MS);
        keys.clear();
        assert!(!held(&mut keys, 32 * MS));
    }

    #[test]
    fn keys_are_reported_in_the_order_pressed() {
        let mut keys = KeyTracker::new(0);
        keys.key(0x1E, b'Q', true, false, 0);
        keys.key(W, b'Z', true, false, MS);
        keys.key(0x1E, b'Q', true, false, 2 * MS);
        assert_eq!(keys.held(8 * MS), vec![(0x1E, b'Q'), (W, b'Z')]);
    }

    #[test]
    fn the_mouse_moves_by_counts_or_from_absolute_positions() {
        let mut m = Motion::default();
        m.moved(0, 3, -2, (0.0, 0.0));
        m.moved(0, 1, 0, (0.0, 0.0));
        let got = m.take();
        assert_eq!((got.dx, got.dy), (4.0, -2.0));
        assert_eq!(m.take().dx, 0.0);
        // Absolute (a remote desktop): a 1920 x 1080 screen spans 0-65535.
        let screen = (1920.0, 1080.0);
        m.moved(MOUSE_MOVE_ABSOLUTE, 32768, 32768, screen);
        assert_eq!(m.take().dx, 0.0, "the first position is no movement");
        m.moved(MOUSE_MOVE_ABSOLUTE, 32768 + 341, 32768 - 607, screen);
        let got = m.take();
        assert!(
            (got.dx - 10.0).abs() < 0.05 && (got.dy + 10.0).abs() < 0.05,
            "{got:?}"
        );
        // A jump back into the remote window elsewhere is no movement.
        m.moved(MOUSE_MOVE_ABSOLUTE, 1000, 1000, screen);
        assert_eq!(m.take().dx, 0.0);
        // Buttons follow their down and up flags.
        m.buttons(0x01 | 0x10);
        let got = m.take();
        assert!(got.left && got.middle && !got.right);
        m.buttons(0x02 | 0x04);
        let got = m.take();
        assert!(!got.left && got.middle && got.right);
        m.clear();
        let got = m.take();
        assert!(!got.left && !got.middle && !got.right);
    }

    #[test]
    fn absolute_positions_go_back_to_the_middle_near_an_edge() {
        // A window's client area from (100, 100) to (1100, 900).
        let rect = (100, 100, 1100, 900);
        assert_eq!(middle(rect), (600, 500));
        // A real mouse (relative counts) is never put back.
        assert!(!needs_recentering(false, (101, 500), rect));
        // Absolute: put back within 15 % of an edge, not in the middle.
        assert!(!needs_recentering(true, (600, 500), rect));
        assert!(!needs_recentering(true, (250, 220), rect));
        assert!(needs_recentering(true, (249, 500), rect));
        assert!(needs_recentering(true, (951, 500), rect));
        assert!(needs_recentering(true, (600, 219), rect));
        assert!(needs_recentering(true, (600, 781), rect));
        // The jump back to the middle is no movement; moving on is.
        let screen = (1920.0, 1080.0);
        let mut m = Motion::default();
        m.moved(MOUSE_MOVE_ABSOLUTE, 50_000, 30_000, screen);
        m.moved(MOUSE_MOVE_ABSOLUTE, 52_000, 30_000, screen);
        assert!(m.take().dx > 50.0);
        m.recentered();
        assert!(m.absolute(), "still absolute while put back");
        m.moved(MOUSE_MOVE_ABSOLUTE, 32_768, 32_768, screen);
        let got = m.take();
        assert_eq!((got.dx, got.dy), (0.0, 0.0));
        m.moved(MOUSE_MOVE_ABSOLUTE, 32_768 + 341, 32_768, screen);
        assert!((m.take().dx - 10.0).abs() < 0.05);
        // Relative counts again: no longer absolute.
        m.moved(0, 2, 0, screen);
        assert!(!m.absolute());
    }
}
