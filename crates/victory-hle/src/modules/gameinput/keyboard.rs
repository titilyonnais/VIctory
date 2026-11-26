//! The keyboard's (and mouse's) stand-in gamepad. By default keys count by
//! where they sit, not by what they print: scan codes, so the QWERTY "WASD"
//! block is ZQSD on an AZERTY keyboard (`VICTORY_KEYBOARD=vk` brings back
//! the older table by virtual key, `off` no stand-in at all). The mouse,
//! while one of our windows has focus, turns the right stick and pulls the
//! triggers (`VICTORY_MOUSE`, `VICTORY_MOUSE_SENS`, `VICTORY_MOUSE_DEADZONE`,
//! `VICTORY_MOUSE_INVERT`). `touches-clavier.md` lists every key.

use std::sync::LazyLock;

use super::{
    A, B, KeyState, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT, RIGHT_SHOULDER,
    RIGHT_STICK, UP, VIEW, X, Y,
};

/// Scan codes (set 1, 0xE0xx when extended) by their QWERTY names; the
/// comment gives the AZERTY label of the same key where it differs.
pub(super) mod key {
    pub const ESC: u32 = 0x01;
    pub const BACKSPACE: u32 = 0x0E;
    pub const TAB: u32 = 0x0F;
    pub const Q: u32 = 0x10; // AZERTY A
    pub const W: u32 = 0x11; // AZERTY Z
    pub const E: u32 = 0x12;
    pub const R: u32 = 0x13;
    pub const I: u32 = 0x17;
    pub const ENTER: u32 = 0x1C;
    pub const LCTRL: u32 = 0x1D;
    pub const A: u32 = 0x1E; // AZERTY Q
    pub const S: u32 = 0x1F;
    pub const D: u32 = 0x20;
    pub const F: u32 = 0x21;
    pub const J: u32 = 0x24;
    pub const K: u32 = 0x25;
    pub const L: u32 = 0x26;
    pub const LSHIFT: u32 = 0x2A;
    pub const Z: u32 = 0x2C; // AZERTY W
    pub const X: u32 = 0x2D;
    pub const C: u32 = 0x2E;
    pub const V: u32 = 0x2F;
    pub const SPACE: u32 = 0x39;
    pub const NUM8: u32 = 0x48;
    pub const NUM4: u32 = 0x4B;
    pub const NUM5: u32 = 0x4C;
    pub const NUM6: u32 = 0x4D;
    pub const NUM2: u32 = 0x50;
    pub const NUM_ENTER: u32 = 0xE01C;
    pub const UP: u32 = 0xE048;
    pub const LEFT: u32 = 0xE04B;
    pub const RIGHT: u32 = 0xE04D;
    pub const DOWN: u32 = 0xE050;
}

/// Buttons by key position (scan code, button), after GTA V's PC keys
/// where a console button has one: Shift runs (A held), Space jumps (X),
/// F gets in and out (Y), R reloads (B), Tab or Q the weapon wheel (LB),
/// E cover (RB), Ctrl crouches (L3), C looks back (R3), V the camera (View).
const POSITION_BUTTONS: [(u32, u32); 18] = [
    (key::ESC, MENU),
    (key::V, VIEW),
    (key::LSHIFT, A),
    (key::ENTER, A),
    (key::NUM_ENTER, A),
    (key::R, B),
    (key::BACKSPACE, B),
    (key::SPACE, X),
    (key::F, Y),
    (key::UP, UP),
    (key::DOWN, super::DOWN),
    (key::LEFT, LEFT),
    (key::RIGHT, RIGHT),
    (key::TAB, LEFT_SHOULDER),
    (key::Q, LEFT_SHOULDER),
    (key::E, RIGHT_SHOULDER),
    (key::LCTRL, LEFT_STICK),
    (key::C, RIGHT_STICK),
];

/// How keys become pad input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Layout {
    /// By position (scan code): the default.
    Position,
    /// The older table by virtual key (`FALLBACK_BUTTONS`), so by label.
    Vk,
    /// No stand-in input from the keyboard or mouse.
    Off,
}

/// The mouse since the last poll: movement in pixels, buttons held. All
/// zero while the mouse does not steer (no focus, or turned off).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Mouse {
    pub dx: i32,
    pub dy: i32,
    pub left: bool,
    pub right: bool,
    pub middle: bool,
}

/// Settings, from the environment ([`settings`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Settings {
    pub layout: Layout,
    /// The mouse steers the camera (and its cursor is held in the window).
    pub mouse: bool,
    /// Stick deflection per unit of mouse speed (1: full at
    /// [`FULL_SPEED`] pixels a second).
    pub sensitivity: f32,
    /// Deflection of the slowest movement, past the title's dead zone.
    pub dead_zone: f32,
    /// Mouse forward looks down.
    pub invert: bool,
}

/// Mouse speed, in pixels a second, that pushes the stick all the way at
/// sensitivity 1.
pub(super) const FULL_SPEED: f32 = 2000.0;
/// Smoothing time of the mouse stick, in seconds: a 125 Hz mouse moves the
/// cursor every other poll.
const SMOOTHING: f32 = 0.03;

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            layout: Layout::Position,
            mouse: true,
            sensitivity: 1.0,
            dead_zone: 0.2,
            invert: false,
        }
    }
}

impl Settings {
    /// The older behaviour: keys by virtual key, no mouse.
    #[cfg(test)]
    pub fn vk() -> Settings {
        Settings {
            layout: Layout::Vk,
            mouse: false,
            ..Settings::default()
        }
    }

    /// Settings from variables (`get` reads one, `None` when unset).
    pub fn from_vars(get: impl Fn(&str) -> Option<String>) -> Settings {
        let default = Settings::default();
        let off = |v: &str| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "non");
        let number = |name: &str, min: f32, max: f32, default: f32| {
            get(name)
                .and_then(|v| v.trim().replace(',', ".").parse::<f32>().ok())
                .filter(|v| v.is_finite())
                .map_or(default, |v| v.clamp(min, max))
        };
        let layout = match get("VICTORY_KEYBOARD").map(|v| v.trim().to_ascii_lowercase()) {
            Some(v) if v == "vk" => Layout::Vk,
            Some(v) if off(&v) => Layout::Off,
            _ => Layout::Position,
        };
        // The mouse comes with the new layout; with the older one only when
        // asked for.
        let mouse = match get("VICTORY_MOUSE") {
            Some(v) => !off(&v),
            None => layout == Layout::Position,
        } && layout != Layout::Off;
        Settings {
            layout,
            mouse,
            sensitivity: number("VICTORY_MOUSE_SENS", 0.01, 100.0, default.sensitivity),
            dead_zone: number("VICTORY_MOUSE_DEADZONE", 0.0, 0.9, default.dead_zone),
            invert: get("VICTORY_MOUSE_INVERT").is_some_and(|v| !off(&v)),
        }
    }
}

/// The settings of this run, read once.
pub(super) fn settings() -> &'static Settings {
    static SETTINGS: LazyLock<Settings> = LazyLock::new(|| {
        let settings = Settings::from_vars(|name| std::env::var(name).ok());
        tracing::info!(?settings, "keyboard and mouse");
        settings
    });
    &SETTINGS
}

/// Mouse movement as a stick: speed, smoothed, past a dead zone.
#[derive(Clone, Copy, Debug)]
pub(super) struct MouseStick {
    sensitivity: f32,
    dead_zone: f32,
    invert: bool,
    x: f32,
    y: f32,
    /// The last update, in microseconds.
    last: Option<u64>,
}

impl MouseStick {
    pub fn new(settings: &Settings) -> MouseStick {
        MouseStick {
            sensitivity: settings.sensitivity,
            dead_zone: settings.dead_zone,
            invert: settings.invert,
            x: 0.0,
            y: 0.0,
            last: None,
        }
    }

    /// The stick for movement `dx`, `dy` (pixels, down positive) at `now`
    /// (microseconds): -1 to 1, up positive.
    pub fn update(&mut self, dx: i32, dy: i32, now: u64) -> (f32, f32) {
        let dt = self
            .last
            .map_or(0.008, |last| now.saturating_sub(last) as f32 / 1e6)
            .clamp(0.001, 0.1);
        self.last = Some(now);
        let scale = self.sensitivity / (FULL_SPEED * dt);
        let target_x = dx as f32 * scale;
        let target_y = if self.invert { dy } else { -dy } as f32 * scale;
        let k = (dt / SMOOTHING).min(1.0);
        self.x += (target_x - self.x) * k;
        self.y += (target_y - self.y) * k;
        let length = self.x.hypot(self.y);
        if length < 0.01 {
            (self.x, self.y) = (0.0, 0.0);
            return (0.0, 0.0);
        }
        let out = (self.dead_zone + (1.0 - self.dead_zone) * length).min(1.0);
        (self.x / length * out, self.y / length * out)
    }
}

/// The stand-in pad for `keys` held, the `mouse` and its `stick`.
pub(super) fn pad(layout: Layout, keys: &[KeyState], mouse: &Mouse, stick: (f32, f32)) -> PadState {
    let from_keys = match layout {
        Layout::Off => return PadState::default(),
        Layout::Vk => super::fallback(keys),
        Layout::Position => by_position(keys),
    };
    let trigger = |held: bool| if held { 1.0 } else { 0.0 };
    let from_mouse = PadState {
        buttons: if mouse.middle { RIGHT_STICK } else { 0 },
        left_trigger: trigger(mouse.right),
        right_trigger: trigger(mouse.left),
        right_x: stick.0,
        right_y: stick.1,
        ..PadState::default()
    };
    super::merged(from_keys, from_mouse)
}

fn by_position(keys: &[KeyState]) -> PadState {
    let held = |code: u32| keys.iter().any(|k| k.scan_code == code);
    let on = |codes: &[u32]| {
        if codes.iter().any(|&c| held(c)) {
            1.0
        } else {
            0.0
        }
    };
    PadState {
        buttons: POSITION_BUTTONS
            .iter()
            .filter(|&&(code, _)| held(code))
            .fold(0, |all, &(_, bit)| all | bit),
        left_trigger: on(&[key::Z]),
        right_trigger: on(&[key::X]),
        left_x: on(&[key::D]) - on(&[key::A]),
        left_y: on(&[key::W]) - on(&[key::S]),
        right_x: on(&[key::L, key::NUM6]) - on(&[key::J, key::NUM4]),
        right_y: on(&[key::I, key::NUM8]) - on(&[key::K, key::NUM5, key::NUM2]),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{DOWN, merged};
    use super::*;

    /// A key as an AZERTY keyboard reports it: its label's virtual key, the
    /// scan code of where it sits.
    fn azerty(label: u8, scan_code: u32) -> KeyState {
        KeyState {
            scan_code,
            virtual_key: label,
            ..Default::default()
        }
    }

    fn still() -> (Mouse, (f32, f32)) {
        (Mouse::default(), (0.0, 0.0))
    }

    #[test]
    fn azerty_zqsd_moves_like_qwerty_wasd() {
        let (mouse, stick) = still();
        let go = |keys: &[KeyState]| pad(Layout::Position, keys, &mouse, stick);
        // AZERTY Z, Q, S, D sit where QWERTY has W, A, S, D.
        assert_eq!(go(&[azerty(b'Z', key::W)]).left_y, 1.0);
        assert_eq!(go(&[azerty(b'S', key::S)]).left_y, -1.0);
        assert_eq!(go(&[azerty(b'Q', key::A)]).left_x, -1.0);
        assert_eq!(go(&[azerty(b'D', key::D)]).left_x, 1.0);
        // AZERTY W (QWERTY Z) is the left trigger, not forward.
        let w = go(&[azerty(b'W', key::Z)]);
        assert_eq!((w.left_y, w.left_trigger), (0.0, 1.0));
        // AZERTY A (QWERTY Q) is the left shoulder, not a stick.
        let a = go(&[azerty(b'A', key::Q)]);
        assert_eq!((a.buttons, a.left_x), (LEFT_SHOULDER, 0.0));
        // Diagonals and opposite keys.
        let both = go(&[azerty(b'Z', key::W), azerty(b'D', key::D)]);
        assert_eq!((both.left_x, both.left_y), (1.0, 1.0));
        let none = go(&[azerty(b'Q', key::A), azerty(b'D', key::D)]);
        assert_eq!(none.left_x, 0.0);
    }

    #[test]
    fn keys_press_their_buttons() {
        let (mouse, stick) = still();
        let go = |code: u32| pad(Layout::Position, &[azerty(0, code)], &mouse, stick);
        for (code, button) in [
            (key::LSHIFT, A),
            (key::ENTER, A),
            (key::SPACE, X),
            (key::F, Y),
            (key::R, B),
            (key::BACKSPACE, B),
            (key::ESC, MENU),
            (key::V, VIEW),
            (key::TAB, LEFT_SHOULDER),
            (key::E, RIGHT_SHOULDER),
            (key::LCTRL, LEFT_STICK),
            (key::C, RIGHT_STICK),
            (key::UP, UP),
            (key::DOWN, DOWN),
            (key::LEFT, LEFT),
            (key::RIGHT, RIGHT),
        ] {
            assert_eq!(go(code).buttons, button, "scan code {code:#x}");
        }
        assert_eq!(go(key::X).right_trigger, 1.0);
        // The camera by keys: I/J/K/L and the keypad.
        assert_eq!(go(key::L).right_x, 1.0);
        assert_eq!(go(key::NUM4).right_x, -1.0);
        assert_eq!(go(key::NUM8).right_y, 1.0);
        assert_eq!(go(key::NUM5).right_y, -1.0);
        // Keypad 8 is no arrow: the arrows are the cross.
        assert_eq!(go(key::NUM8).buttons, 0);
        // F8 (the mouse on and off) is no button.
    }

    #[test]
    fn the_vk_layout_keeps_the_older_table() {
        let (mouse, stick) = still();
        // By label: an AZERTY Z is the left trigger again.
        let z = pad(Layout::Vk, &[azerty(b'Z', key::W)], &mouse, stick);
        assert_eq!((z.left_trigger, z.left_y), (1.0, 0.0));
        let w = pad(Layout::Vk, &[azerty(b'W', key::Z)], &mouse, stick);
        assert_eq!(w.left_y, 1.0);
        assert_eq!(
            pad(Layout::Off, &[azerty(b'W', key::W)], &mouse, stick),
            PadState::default()
        );
    }

    #[test]
    fn the_mouse_aims_shoots_and_turns_the_camera() {
        let mouse = Mouse {
            left: true,
            right: true,
            middle: true,
            ..Mouse::default()
        };
        let p = pad(Layout::Position, &[], &mouse, (0.5, -0.25));
        assert_eq!(
            p,
            PadState {
                buttons: RIGHT_STICK,
                left_trigger: 1.0,
                right_trigger: 1.0,
                right_x: 0.5,
                right_y: -0.25,
                ..PadState::default()
            }
        );
        // A camera key wins over a slower mouse.
        let p = pad(
            Layout::Position,
            &[azerty(b'J', key::J)],
            &Mouse::default(),
            (0.5, 0.0),
        );
        assert_eq!(p.right_x, -1.0);
    }

    #[test]
    fn keyboard_and_a_host_pad_merge() {
        let host = PadState {
            buttons: B,
            left_x: -0.3,
            left_y: 0.4,
            right_x: 0.2,
            left_trigger: 0.5,
            ..PadState::default()
        };
        let keys = pad(
            Layout::Position,
            &[azerty(b'Z', key::W), azerty(0, key::SPACE)],
            &Mouse::default(),
            (0.0, 0.0),
        );
        let both = merged(host, keys);
        assert_eq!(both.buttons, B | X);
        // Of each axis the one pushed further.
        assert_eq!((both.left_x, both.left_y, both.right_x), (-0.3, 1.0, 0.2));
        assert_eq!(both.left_trigger, 0.5);
        // At rest the keyboard changes nothing.
        let rest = pad(Layout::Position, &[], &Mouse::default(), (0.0, 0.0));
        assert_eq!(merged(host, rest), host);
    }

    /// The stick after `polls` polls 8 ms apart, each moving `dx`, `dy`.
    fn steady(settings: &Settings, dx: i32, dy: i32, polls: u64) -> (f32, f32) {
        let mut stick = MouseStick::new(settings);
        let mut out = (0.0, 0.0);
        for i in 0..polls {
            out = stick.update(dx, dy, 1_000 + i * 8_000);
        }
        out
    }

    #[test]
    fn mouse_sensitivity_scales_the_stick() {
        let linear = Settings {
            dead_zone: 0.0,
            ..Settings::default()
        };
        // 4 pixels each 8 ms: 500 pixels a second, a quarter of full speed.
        let (x, y) = steady(&linear, 4, 0, 50);
        assert!((x - 0.25).abs() < 0.01 && y == 0.0, "{x} {y}");
        let double = Settings {
            sensitivity: 2.0,
            ..linear
        };
        let (x2, _) = steady(&double, 4, 0, 50);
        assert!((x2 - 0.5).abs() < 0.01, "{x2}");
        // Mouse forward (up the screen) looks up, unless inverted.
        let (_, up) = steady(&linear, 0, -4, 50);
        assert!(up > 0.2, "{up}");
        let inverted = Settings {
            invert: true,
            ..linear
        };
        let (_, down) = steady(&inverted, 0, -4, 50);
        assert!(down < -0.2, "{down}");
        // Fast movement stops at full deflection, in its direction (down
        // the screen is down on the stick).
        let (x, y) = steady(&linear, 100, 100, 50);
        assert!((x.hypot(y) - 1.0).abs() < 1e-4 && x > 0.0 && (x + y).abs() < 1e-4);
        // The dead zone: the slowest movement already clears it.
        let (slow, _) = steady(&Settings::default(), 1, 0, 50);
        assert!((0.2..=0.26).contains(&slow), "{slow}");
        // At rest it comes back to the middle, exactly.
        let mut stick = MouseStick::new(&Settings::default());
        stick.update(50, 0, 0);
        let mut rest = (1.0, 1.0);
        for i in 1..40 {
            rest = stick.update(0, 0, i * 8_000);
        }
        assert_eq!(rest, (0.0, 0.0));
    }

    #[test]
    fn settings_come_from_the_environment() {
        let vars = |list: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                list.iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, v)| v.to_string())
            }
        };
        assert_eq!(Settings::from_vars(vars(&[])), Settings::default());
        let old = Settings::from_vars(vars(&[("VICTORY_KEYBOARD", "vk")]));
        assert_eq!((old.layout, old.mouse), (Layout::Vk, false));
        let old_mouse =
            Settings::from_vars(vars(&[("VICTORY_KEYBOARD", "VK"), ("VICTORY_MOUSE", "1")]));
        assert_eq!((old_mouse.layout, old_mouse.mouse), (Layout::Vk, true));
        let off = Settings::from_vars(vars(&[("VICTORY_KEYBOARD", "off"), ("VICTORY_MOUSE", "1")]));
        assert_eq!((off.layout, off.mouse), (Layout::Off, false));
        let tuned = Settings::from_vars(vars(&[
            ("VICTORY_MOUSE_SENS", "1,5"),
            ("VICTORY_MOUSE_DEADZONE", "7"),
            ("VICTORY_MOUSE_INVERT", "1"),
            ("VICTORY_MOUSE", "0"),
        ]));
        assert_eq!(
            tuned,
            Settings {
                mouse: false,
                sensitivity: 1.5,
                dead_zone: 0.9,
                invert: true,
                ..Settings::default()
            }
        );
        let bad = Settings::from_vars(vars(&[("VICTORY_MOUSE_SENS", "beaucoup")]));
        assert_eq!(bad.sensitivity, 1.0);
    }
}
