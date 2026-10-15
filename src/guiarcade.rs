//! Runs built-in arcade games through the graphical frame protocol.

use std::time::{Duration, Instant};

use ratatui::crossterm::event::{KeyCode, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};

use crate::{
    arcade::Arcade,
    console_input::ConsoleInput,
    guiframe::{Cell, Input, Snapshot},
};

/// Drives an already opened game, retiring its graphical identity on every exit.
///
/// # Errors
/// Returns an error on interruption, disconnection, invalid grids or failed input/publication.
pub(crate) fn run_with(
    arcade: &mut Arcade,
    console: &ConsoleInput,
    component: &str,
    id: &str,
    mut read: impl FnMut(&mut bool) -> Result<Option<Vec<u8>>, String>,
    mut publish: impl FnMut(&str, Option<&Snapshot>) -> Result<(), String>,
) -> Result<Option<String>, String> {
    let mut size = (80, 16);
    let mut last = Instant::now();
    arcade.sound.set(false);
    let result = (|| loop {
        let tick = Instant::now();
        if console.interrupted() || console.eof() {
            break Err("Graphical arcade interrupted or disconnected".into());
        }
        arcade.step(u64::try_from(last.elapsed().as_millis()).unwrap_or(u64::MAX));
        last = Instant::now();
        let frame = snapshot(arcade, component, id, size);
        if !frame.is_valid() {
            break Err("Arcade returned an invalid graphical grid".into());
        }
        publish(id, Some(&frame))?;
        for request in console.frame_inputs(id, &mut read)? {
            let outcome = apply_input(arcade, request.input, &mut size);
            if let crate::arcade::Outcome::Close(line) = outcome {
                return Ok(line);
            }
        }
        // Use the same twenty-frame cadence as WASM frames.
        if let Some(remaining) = Duration::from_millis(50).checked_sub(tick.elapsed()) {
            std::thread::sleep(remaining);
        }
    })();
    let line = arcade
        .close()
        .or_else(|| result.as_ref().ok().cloned().flatten());
    let retired = publish(id, None);
    result.and(retired.map(|()| line))
}

pub(crate) fn apply_input(
    arcade: &mut Arcade,
    input: Input,
    size: &mut (u16, u16),
) -> crate::arcade::Outcome {
    use crate::arcade::Outcome;
    match input {
        Input::Close => Outcome::Close(arcade.close()),
        Input::Resize { w, h } => {
            if w >= crate::arcade::MIN_W && h > crate::arcade::MIN_H {
                *size = (w, h);
            }
            Outcome::Stay
        }
        Input::Key { code, text } => {
            let Ok(mut key) = crate::uiremote::parse_key(&code) else {
                return Outcome::Stay;
            };
            if let Some(text) = text
                && !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
            {
                key.code = KeyCode::Char(text);
                if text.is_ascii_uppercase() {
                    key.modifiers |= KeyModifiers::SHIFT;
                }
            }
            // Terminal bells and transcript veiling have no graphical adapter yet.
            if matches!(key.code, KeyCode::Char('b' | 't')) {
                return Outcome::Stay;
            }
            arcade.handle_key(key)
        }
        Input::Text { text } => {
            if !crate::guiframe::valid_text(&text) {
                return Outcome::Stay;
            }
            for character in text.chars() {
                let outcome = apply_input(
                    arcade,
                    Input::Key {
                        code: if character == ' ' {
                            "space".into()
                        } else {
                            character.to_lowercase().collect()
                        },
                        text: Some(character),
                    },
                    size,
                );
                if matches!(outcome, Outcome::Close(_)) {
                    return outcome;
                }
            }
            Outcome::Stay
        }
        Input::Mouse { kind, x, y } => {
            if x >= size.0 || y >= size.1.saturating_sub(1) {
                return Outcome::Stay;
            }
            let kind = match kind.as_str() {
                "down" => MouseEventKind::Down(MouseButton::Left),
                "up" => MouseEventKind::Up(MouseButton::Left),
                "drag" => MouseEventKind::Drag(MouseButton::Left),
                "move" => MouseEventKind::Moved,
                "scroll_up" => MouseEventKind::ScrollUp,
                _ => MouseEventKind::ScrollDown,
            };
            arcade.handle_mouse(
                MouseEvent {
                    kind,
                    column: x,
                    row: y,
                    modifiers: KeyModifiers::NONE,
                },
                size.0,
                size.1 - 1,
            )
        }
    }
}

pub(crate) fn snapshot(arcade: &Arcade, component: &str, id: &str, (w, h): (u16, u16)) -> Snapshot {
    let area_h = h.saturating_sub(1);
    let mut cells: Vec<_> = arcade
        .glyphs(w, area_h)
        .into_iter()
        .filter(|glyph| glyph.x < w && glyph.y < area_h)
        .map(|glyph| Cell {
            x: glyph.x,
            y: glyph.y,
            ch: glyph.ch,
            fg: [glyph.color.0, glyph.color.1, glyph.color.2],
            bg: None,
            bold: false,
        })
        .collect();
    let mut line = |text: &str, x: u16, y: u16, bold| {
        cells.extend(
            text.chars()
                .take(usize::from(w.saturating_sub(x)))
                .enumerate()
                .map(|(index, ch)| Cell {
                    x: x + u16::try_from(index).unwrap_or_default(),
                    y,
                    ch,
                    fg: [190, 202, 220],
                    bg: Some([0, 0, 0]),
                    bold,
                }),
        );
    };
    if let Some(banner) = arcade.banner(w, area_h) {
        let length = u16::try_from(banner.chars().count()).unwrap_or(w).min(w);
        line(&banner, w.saturating_sub(length) / 2, area_h / 2, true);
    }
    let footer = arcade
        .footer()
        .replace(" · b sound", "")
        .replace(" · b mute", "")
        .replace(" · t opaque", "")
        .replace(" · t veiled", "");
    line(&format!("Esc/q quit · silent · {footer}"), 0, area_h, false);
    Snapshot {
        id: id.into(),
        component: component.into(),
        w,
        h,
        cells,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphical_arcade_resizes_closes_and_preserves_queued_prompts_and_parked_games() {
        for command in Arcade::COMMANDS {
            let mut arcade = Arcade::new();
            assert!(arcade.open(command, true, 1));
            let console = ConsoleInput::default();
            let mut frames = Vec::new();
            let mut reads = 0;
            run_with(&mut arcade, &console, "arcade", "frame-one", |eof| {
                reads += 1;
                if reads == 1 {
                    Ok(Some(br#"{"type":"frame_input","id":"frame-one","event":"resize","w":120,"h":40}
{"type":"frame_input","id":"frame-one","event":"key","code":"space","text":" "}
{"type":"frame_input","id":"frame-one","event":"key","code":"shift+up","text":null}
{"type":"frame_input","id":"stale","event":"close"}
next prompt
"#.to_vec()))
                } else if reads == 2 {
                    Ok(Some(br#"{"type":"frame_input","id":"frame-one","event":"close"}
"#.to_vec()))
                } else { *eof = true; Ok(None) }
            }, |id, snapshot| { assert_eq!(id, "frame-one"); frames.push(snapshot.cloned()); Ok(()) }).unwrap();
            assert_eq!(
                (frames[1].as_ref().unwrap().w, frames[1].as_ref().unwrap().h),
                (120, 40)
            );
            assert!(frames.last().unwrap().is_none());
            assert!(!arcade.is_open());
            assert_eq!(
                console.prompt(|_| unreachable!()).unwrap().as_deref(),
                Some("next prompt\n")
            );
            if command != "/matrix" {
                assert!(arcade.has_parked(command));
            }
        }
    }

    #[test]
    fn graphical_arcade_disconnection_and_output_failure_retire_the_frame() {
        let mut arcade = Arcade::new();
        arcade.open("/pelota", true, 1);
        let console = ConsoleInput::default();
        let mut retired = false;
        assert!(
            run_with(
                &mut arcade,
                &console,
                "arcade",
                "frame-one",
                |eof| {
                    *eof = true;
                    Ok(None)
                },
                |_, snapshot| {
                    retired |= snapshot.is_none();
                    Ok(())
                }
            )
            .is_err()
        );
        assert!(retired);
        arcade.open("/pelota", false, 2);
        retired = false;
        assert!(
            run_with(
                &mut arcade,
                &ConsoleInput::default(),
                "arcade",
                "frame-two",
                |_| Ok(None),
                |_, snapshot| {
                    if snapshot.is_some() {
                        Err("Renderer disconnected".into())
                    } else {
                        retired = true;
                        Ok(())
                    }
                }
            )
            .is_err()
        );
        assert!(retired);
        assert!(!arcade.is_open());
    }
}
