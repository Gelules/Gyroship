//! The store demo's lines in the full game's main menu (layer-demo, item 4 of D91), ported from
//! edit_demo.py. On the practice layer's main menu (QUIT GAME's line there already), each of the
//! demo's three lines is spawned in the full game and its box shown; their labels, numbers and
//! clicks stay the demo's. CAMPAIGN's explicit Up goes to the lowest of them instead of the top
//! bar, and the menu's boxes grow by a line each.

use super::quit::{grow_menu, indent_of};
use super::{Layer, Recipe, ascii, json, one};

const ABOUT: &str = "Add the eShop demo's lines to the main menu (fresh copy: METEO saves a new game)";
/// The keys of a component's reference, in the file's order: the explicit Up takes the target's.
const REFERENCE: [&str; 5] = ["debug_description", "vtable", "guid", "entity_guid", "path_hash"];

/// The demo layer: the practice layer with its main menu edited.
pub fn layer(r: &Recipe, base: &Layer) -> Result<Layer, String> {
    let name = r.one("title.level")?;
    let menu = base.entries.get(name).ok_or("the practice layer has no main menu")?;
    let menu = edit_demo(&ascii(menu.clone(), "the practice layer's main menu")?, r).map_err(|e| format!("the main menu: {e}"))?;
    let mut layer = base.clone();
    layer.about = ABOUT.into();
    layer.entries.insert(name.to_string(), menu.into_bytes());
    Ok(layer)
}

/// The decoded main menu level with the demo's lines shown. Text edits only: every other byte
/// stays; each anchor found exactly once in its line's entity (from the previous entity's name at
/// the same indentation to its own name).
pub fn edit_demo(text: &str, r: &Recipe) -> Result<String, String> {
    let (names, filter) = (r.all("demo.lines")?, r.one("title.filter")?);
    let per_line: f64 = r.one("demo.height")?.parse().map_err(|e| format!("mods.txt: demo.height: {e}"))?;
    let mut lines: Vec<String> = text.split('\n').map(String::from).collect();
    let everywhere = 0..lines.len();
    // (line, true) gets the filter's value, (line, false) is the collapsed box, removed.
    let mut edits: Vec<(usize, bool)> = Vec::new();
    for line in names {
        let name = one(&lines, everywhere.clone(), |l| l.trim() == format!("\"name\": \"{line}\","), &format!("{line}'s name"))?;
        let head = format!("{}\"name\": \"", indent_of(&lines[name]));
        let first = (0..name).filter(|&i| lines[i].starts_with(&head)).max().map_or(0, |i| i + 1);
        let i_box = one(&lines, first..name, |l| l.trim() == "\"collapsed\": true,", &format!("{line}'s collapsed box"))?;
        let i_filter = one(&lines, first..name, |l| l.trim() == format!("\"cvar_name\": \"{filter}\","), &format!("{line}'s filter"))?;
        if i_box >= i_filter {
            return Err(format!("{line}: its parts are not in the order this tool knows"));
        }
        edits.extend([(i_filter, true), (i_box, false)]);
    }

    // CAMPAIGN's explicit Up: the five lines of its target's reference take those of the lowest
    // demo line's button widget (same keys, same order; indentation kept).
    let target = names.last().ok_or("mods.txt: demo.lines is empty")?;
    let (up_field, up_old, widget) = (r.one("demo.up_field")?, r.one("demo.up_old")?, r.one("demo.widget")?);
    let i_up = one(&lines, everywhere.clone(), |l| l.trim() == format!("\"field_name\": \"{up_field}\""), "CAMPAIGN's explicit Up")?;
    let i_old = one(&lines, i_up.saturating_sub(20)..i_up, |l| l.trim().starts_with(&format!("\"debug_description\": \"{up_old}")), "the explicit Up's target")?;
    let wanted = format!("\"debug_description\": \"{target}{widget}\",");
    let i_ref = everywhere.clone().find(|&i| lines[i].trim() == wanted).ok_or_else(|| format!("no button widget of {target}"))?;
    for (k, key) in REFERENCE.iter().enumerate() {
        let head = format!("\"{key}\": ");
        let (old, new) = (&lines[i_old + k], &lines[i_ref + k]);
        if !(old.trim().starts_with(&head) && new.trim().starts_with(&head)) {
            return Err(format!("the explicit Up's reference and {target}'s do not have the keys this tool knows"));
        }
        lines[i_old + k] = format!("{}{}", indent_of(old), new.trim());
    }

    grow_menu(&mut lines, r, per_line * names.len() as f64, Some(4))?;
    // The lines' edits, from the last to the first, so the indexes above stay right.
    edits.sort();
    for &(i, value) in edits.iter().rev() {
        if value {
            let indent = indent_of(&lines[i]).to_string();
            lines.splice(i..i, [format!("{indent}\"value\": {{"), format!("{indent}\t\"bool\": false"), format!("{indent}}},")]);
        } else {
            lines.remove(i);
        }
    }
    let out = lines.join("\n");
    json::no_key_twice(&out)?;
    Ok(out)
}
