//! The practice layer, `layer` (M6 steps 7 and 7b): QUIT GAME in the pause menu and in the title
//! menu, and the mission timer, ported from the scripts that first built it. Three entries of
//! data.zip are rebuilt from the player's own: the menus' string table (one string more, our 14
//! texts), the pause menu prefab (a third choice in its QUIT confirmations), the main menu level
//! (a sixth line: the store demo's purchase line made QUIT GAME). The settings file gets the
//! timer. The title line's click needs the "quit" hook of templates/hooks.txt.

use super::json::{self, render};
use super::{Archive, Layer, Recipe, ascii, crc32c, import_key, is_guid, is_native, once, one, uuid5};
use gyroship_practice::texts::{self, Text};
use serde_json::{Value, json};

/// Our string's id, and its GUID, made once and kept: the levels refer to the string by it.
const ID: &str = "GYROSHIP_QUIT_GAME";
pub const GUID: &str = "83fbc756-fe56-4f74-87ae-29f6bec0e440";
/// The namespace of our overrides' GUIDs (UUID v5 of "<instance>/<field>"): a build writes the
/// same bytes each time.
const NAMESPACE: &str = "5f1d6a0e-8c2b-4f37-9a41-3e7b2c9d0f18";
const ABOUT: &str = "Add QUIT GAME and the mission timer (the other mods add them too)";

/// The practice layer, from the dump's archive and its settings file.
pub fn layer(r: &Recipe, zip: &mut Archive, cvar: &str) -> Result<Layer, String> {
    let texts: Vec<&str> = (0..texts::REGIONS).map(|region| texts::text(Text::QuitGame, region)).collect();
    let (strings_name, strings) = zip.named(r.one("quit.strings")?)?;
    let strings = String::from_utf8(strings).map_err(|e| format!("the menus' string table is not UTF-8: {e}"))?;
    let strings = add_string(&strings, r.one("quit.template")?, &texts).map_err(|e| format!("the menus' string table: {e}"))?;
    let (pause_name, pause) = zip.named(r.one("quit.pause")?)?;
    let pause = edit_prefab(&ascii(pause, "the pause menu")?, r).map_err(|e| format!("the pause menu: {e}"))?;
    let (title_name, title) = zip.named(r.one("title.level")?)?;
    let height: f64 = r.one("title.height")?.parse().map_err(|e| format!("mods.txt: title.height: {e}"))?;
    let title = edit_title(&ascii(title, "the main menu")?, r, height).map_err(|e| format!("the main menu: {e}"))?;
    let cvar = with_cvar(cvar, r.one("timer.cvar")?)?;
    let entries = [(strings_name, strings.into_bytes()), (pause_name, pause.into_bytes()), (title_name, title.into_bytes())];
    Ok(Layer { about: ABOUT.into(), cvar, entries: entries.into_iter().collect() })
}

/// The lines of `block` made of `tabs` tabs, `head`, a value `ok` takes and `tail`, the whole
/// line: exactly `count` of them, each value replaced by `new()`'s, in turn.
#[allow(clippy::too_many_arguments)]
fn sub_lines(block: &str, tabs: usize, head: &str, tail: &str, ok: impl Fn(&str) -> bool, mut new: impl FnMut() -> String, count: usize, what: &str) -> Result<String, String> {
    let prefix = format!("{}{head}", "\t".repeat(tabs));
    let mut n = 0;
    let mut lines = Vec::new();
    for line in block.split('\n') {
        match line.strip_prefix(&prefix).and_then(|rest| rest.strip_suffix(tail)) {
            Some(value) if ok(value) => {
                n += 1;
                lines.push(format!("{prefix}{}{tail}", new()));
            }
            _ => lines.push(line.to_string()),
        }
    }
    if n != count {
        return Err(format!("{n} lines of {what} in the template, {count} expected"));
    }
    Ok(lines.join("\n"))
}

/// A whole JSON string, quotes and escapes included.
fn is_string_literal(s: &str) -> bool {
    let Some(inner) = s.strip_prefix('"').and_then(|s| s.strip_suffix('"')) else { return false };
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return false,
            '\\' if chars.next().is_none() => return false,
            _ => {}
        }
    }
    true
}

/// The quoted GUID-like words of `text` (36 of 0-9, a-f and dashes), left to right.
fn quoted_guids(text: &str) -> Vec<&str> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 38 <= b.len() {
        if b[i] == b'"' && b[i + 37] == b'"' && b[i + 1..i + 37].iter().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c) || *c == b'-') {
            out.push(&text[i + 1..i + 37]);
            i += 38;
        } else {
            i += 1;
        }
    }
    out
}

/// Adds our string to the decoded menus' string table (`.locs_strings`): the text of an existing
/// translated string, `template`, in which only "$guid", "guid", "$name", "id", "version_hash" and
/// the 14 "text" values change. It goes after the last string, and its GUID at the end of the
/// header's "strings" and of the image's "exports". Every other byte stays.
pub fn add_string(text: &str, template: &str, texts: &[&str]) -> Result<String, String> {
    let tabs = |n: usize| "\t".repeat(n);
    if texts.len() != texts::REGIONS {
        return Err(format!("{} texts, {} expected", texts.len(), texts::REGIONS));
    }
    if let Some(i) = texts.iter().position(|t| t.is_empty() || t.chars().any(|c| (c as u32) < 32)) {
        return Err(format!("text {i} is empty or holds a control character"));
    }
    if text.contains(&format!("\"id\": \"{ID}\"")) {
        return Err(format!("the id {ID} is already in the table"));
    }
    if text.contains(GUID) {
        return Err(format!("the GUID {GUID} is already in the table"));
    }
    // The template's object, as text: from its "{" line to its "}" (exported objects sit at 4 tabs).
    let found = text.find(&format!("\n{}\"id\": \"{template}\",\n", tabs(5)));
    let (Some(at), 1) = (found, text.matches(&format!("\"id\": \"{template}\"")).count()) else {
        return Err(format!("the template {template} is not in the table exactly once"));
    };
    let start = text[..at].rfind(&format!("\n{}{{\n", tabs(4))).ok_or("no start of the template's object")? + 1;
    let close = format!("\n{}}}", tabs(4));
    let end = at + text[at..].find(&close).ok_or("no end of the template's object")? + close.len();
    let block = &text[start..end];
    let obj = json::parse(block)?;
    if obj.get("type").and_then(Value::as_str) != Some("k_locs_translation_type_translate") {
        return Err(format!("the template's type is {}, not translate", obj.get("type").unwrap_or(&Value::Null)));
    }
    // The regions, in our order (texts.rs): US English has no language of its own in the table.
    let slots: Vec<Value> = obj.get("translations").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default().iter().map(|t| t.get("language").cloned().unwrap_or(Value::Null)).collect();
    let regions: Vec<Value> = (0..texts::REGIONS)
        .map(|r| match (r, texts::tag(r).split_once('-')) {
            (0, _) | (_, None) => Value::Null,
            (_, Some((language, country))) => json!({"country": format!("k_country_{}", country.to_lowercase()), "language": format!("k_language_{language}")}),
        })
        .collect();
    if slots != regions {
        return Err(format!("the template has {} translations, or not in the expected region order", slots.len()));
    }
    if obj.get("$guid") != obj.get("guid") {
        return Err("the template's $guid and guid differ".into());
    }
    let guid_chars = |v: &str| v.len() == 36 && v.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c) || c == b'-');
    let plain = |v: &str| !v.contains(['"', '\\']);
    let block = sub_lines(block, 5, "\"$guid\": \"", "\",", guid_chars, || GUID.into(), 1, "$guid")?;
    let block = sub_lines(&block, 5, "\"guid\": \"", "\",", guid_chars, || GUID.into(), 1, "guid")?;
    let block = sub_lines(&block, 5, "\"$name\": \"", "\",", plain, || ID.to_lowercase().chars().take(19).collect(), 1, "$name")?;
    let block = sub_lines(&block, 5, "\"id\": \"", "\",", plain, || ID.into(), 1, "id")?;
    let digits = |v: &str| !v.is_empty() && v.bytes().all(|c| c.is_ascii_digit());
    let block = sub_lines(&block, 5, "\"version_hash\": ", ",", digits, || crc32c(texts[0].as_bytes()).to_string(), 1, "version_hash")?;
    let mut next = texts.iter();
    let block = sub_lines(&block, 7, "\"text\": ", "", is_string_literal, || json::string(next.next().expect("14 texts"), false), texts::REGIONS, "text")?;

    // 1. The header's "strings": the first array of the file closed at 5 tabs.
    let head_end = text.find(&format!("\n{}],\n", tabs(5))).ok_or("no header")?;
    if !text.starts_with("{\n\t\"segments\": [\n\t\t{\n\t\t\t\"exported_objects\": [\n\t\t\t\t{\n\t\t\t\t\t\"strings\": [\n") {
        return Err("the file does not start with the header's \"strings\"".into());
    }
    let listed = quoted_guids(&text[..head_end]);
    // 2. The end of exported_objects: the only array closed at 3 tabs. 3. The image's "exports":
    // the last array of the file.
    let objects_end = format!("\n{}}}\n{}],\n", tabs(4), tabs(3));
    let exports_at = text.rfind("\n\t\"exports\": [\n").ok_or("no exports")?;
    let tail = "\n\t]\n}";
    if text.matches(&objects_end).count() != 1 || !text.ends_with(tail) || text[exports_at..text.len() - tail.len()].contains("\n\t]") {
        return Err("\"exported_objects\" or \"exports\" is not where this tool expects it".into());
    }
    if quoted_guids(&text[exports_at..]).get(1..) != Some(&listed[..]) {
        return Err("\"exports\" is not the header followed by the header's \"strings\"".into());
    }
    let objects_at = text.find(&objects_end).expect("found once");
    let closing = format!("\n{}}}", tabs(4));
    Ok([
        &text[..head_end],
        &format!(",\n{}\"{GUID}\"", tabs(6)),
        &text[head_end..objects_at],
        &format!("\n{}}},\n", tabs(4)),
        &block,
        &text[objects_at + closing.len()..text.len() - tail.len()],
        &format!(",\n{}\"{GUID}\"", tabs(2)),
        tail,
    ]
    .concat())
}

/// A string member of an object, when it is one.
fn as_str(v: Option<&Value>) -> Option<&str> {
    v.and_then(Value::as_str)
}

/// The objects of a JSON tree, with their depth (the number of keys and indexes from the root):
/// an object at depth d is written d tabs in.
fn walk<'a>(v: &'a Value, depth: usize, out: &mut Vec<(usize, &'a Value)>) {
    let children: Box<dyn Iterator<Item = &Value>> = match v {
        Value::Object(map) => {
            out.push((depth, v));
            Box::new(map.values())
        }
        Value::Array(items) => Box::new(items.iter()),
        _ => return,
    };
    for c in children.filter(|c| c.is_object() || c.is_array()) {
        walk(c, depth + 1, out);
    }
}

/// QUIT GAME in the pause menu's QUIT confirmation: edits the decoded pause prefab. For each
/// confirmation instance, four overrides go at the end of its prefab_overrides, as the game's own
/// three-button confirmations carry them: the third button shown, its label (our string), its
/// sound, and its signal to the pause menu's script, whose handler quits. Our string's GUID enters
/// the level's imports at its sorted place. Text insertions only; every anchor found once.
pub fn edit_prefab(text: &str, r: &Recipe) -> Result<String, String> {
    let (confirm, menu_script, menu_suffix) = (r.one("quit.confirm_level")?, r.one("quit.menu_script")?, r.one("quit.menu_script_suffix")?);
    let (override_vtable, signal_vtable, sound, event) = (r.one("quit.override_vtable")?, r.one("quit.signal_vtable")?, r.one("quit.sound_level")?, r.one("quit.event")?);
    let (button1, pause_script, fields) = (r.one("quit.button1_signal")?, r.one("quit.pause_script")?, r.all("quit.fields")?);
    if fields.len() != 4 {
        return Err("mods.txt: quit.fields wants the four field names".into());
    }
    let doc = json::parse(text)?;
    let imports: Vec<&str> = doc.get("imports").and_then(Value::as_array).ok_or("no imports")?.iter().filter_map(Value::as_str).collect();
    if !imports.contains(&sound) {
        return Err(format!("the sound level {sound} is not imported by this level"));
    }
    let mut nodes = Vec::new();
    walk(&doc, 0, &mut nodes);
    let components = |n: &'_ Value| n.get("components").and_then(Value::as_array).cloned().unwrap_or_default();
    let instances_it = |c: &Value| as_str(c.get("level")) == Some(confirm);
    let mut inserts: Vec<(usize, String)> = Vec::new();
    let mut pause_ref: Option<Value> = None;
    for name in r.all("quit.instances")? {
        let found: Vec<&(usize, &Value)> = nodes.iter().filter(|(_, n)| as_str(n.get("name")) == Some(name) && components(n).iter().any(instances_it)).collect();
        let [&(path_len, entity)] = found[..] else {
            return Err(format!("{} entities named {name} that instance the confirmation level", found.len()));
        };
        let prefabs: Vec<Value> = components(entity).into_iter().filter(instances_it).collect();
        let [prefab] = &prefabs[..] else {
            return Err(format!("{name}: {} prefab components", prefabs.len()));
        };
        let overrides = prefab.get("prefab_overrides").and_then(Value::as_array).ok_or_else(|| format!("{name}: no prefab_overrides"))?;
        // The overrides' depth: ... "components", i, "prefab_overrides", k.
        let depth = path_len + 4;
        let field = |o: &Value| as_str(o.get("field_name")).map(String::from);
        let taken: Vec<String> = overrides.iter().filter_map(field).filter(|f| fields.contains(f)).collect();
        if !taken.is_empty() {
            return Err(format!("{name} already overrides {taken:?}: edited before, or not the entry we know"));
        }
        // The siblings' reference to the instance's confirmation script, copied as it is.
        let refs: Vec<&Value> = overrides.iter().filter_map(|o| o.get("component")).filter(|c| as_str(c.get("guid")) == Some(menu_script)).collect();
        if refs.is_empty() || refs.iter().any(|c| *c != refs[0]) {
            return Err(format!("{name}: no single form of the confirmation script's reference"));
        }
        if as_str(refs[0].get("debug_description")) != Some(&format!("{name}{menu_suffix}")) {
            return Err(format!("{name}: unexpected reference {:?}", refs[0].get("debug_description")));
        }
        // The pause menu's script, as button 1's own signal addresses it.
        let b1: Vec<&Value> = overrides.iter().filter(|o| field(o).as_deref() == Some(button1)).collect();
        let targets = b1.first().and_then(|o| o.pointer("/value_storage/handle/targets")).and_then(Value::as_array);
        let (1, Some([target])) = (b1.len(), targets.map(Vec::as_slice)) else {
            return Err(format!("{name}: {button1} is not one override with one target"));
        };
        let target_component = target.get("component").ok_or_else(|| format!("{name}: button 1's target has no component"))?;
        if as_str(target_component.get("debug_description")) != Some(pause_script) || target.get("args").is_some() {
            return Err(format!("{name}: button 1 does not signal the pause menu's script plainly"));
        }
        if pause_ref.as_ref().is_some_and(|p| p != target_component) {
            return Err("the instances address two different pause menu scripts".into());
        }
        pause_ref = Some(target_component.clone());
        let new = |f: &str, storage: Value| json!({"value_storage": storage, "vtable": override_vtable, "guid": uuid5(NAMESPACE, &format!("{name}/{f}")), "component": refs[0], "field_name": f});
        let added = [
            new(&fields[0], json!({"bool_value": true, "type": "k_vm_type_bool"})),
            new(&fields[1], json!({"handle": GUID, "type": "k_vm_type_ref"})),
            new(&fields[2], json!({"handle": sound, "type": "k_vm_type_ref"})),
            new(&fields[3], json!({"handle": {"targets": [{"component": target_component, "event_name": event}], "vtable": signal_vtable}, "type": "k_vm_type_ref"})),
        ];
        for o in &added {
            let g = as_str(o.get("guid")).expect("a GUID");
            if text.contains(g) {
                return Err(format!("the generated GUID {g} already occurs in the file"));
            }
        }
        // The anchor: the instance's last override as the file writes it, then the array's end.
        // Finding it once also proves `render` writes the file's own layout.
        let last = render(overrides.last().ok_or_else(|| format!("{name}: no override"))?, depth, true);
        let at = once(text, &format!("{last}\n{}]", "\t".repeat(depth - 1)), &format!("the end of {name}'s prefab_overrides"))?;
        inserts.push((at + last.len(), added.iter().map(|o| format!(",\n{}", render(o, depth, true))).collect()));
    }

    // The imports: engine types first, then every other GUID sorted.
    if !imports.contains(&GUID) {
        let first = imports.iter().position(|g| !is_native(g)).ok_or("no import but engine types")?;
        let rest = &imports[first..];
        if rest.iter().any(|g| is_native(g)) || !rest.windows(2).all(|w| import_key(w[0]) <= import_key(w[1])) {
            return Err("the imports are not in the order this tool knows (engine types first, the others sorted)".into());
        }
        match rest.iter().find(|g| import_key(g) > import_key(GUID)) {
            Some(next) => inserts.push((once(text, &format!("\n\t\t\"{next}\""), &format!("the import line of {next}"))? + 1, format!("\t\t\"{GUID}\",\n"))),
            None => {
                let last = format!("\n\t\t\"{}\"", rest[rest.len() - 1]);
                inserts.push((once(text, &format!("{last}\n\t],\n\t\"exports\": ["), "the last import line")? + last.len(), format!(",\n\t\t\"{GUID}\"")));
            }
        }
    }
    let mut out = text.to_string();
    inserts.sort();
    for (at, piece) in inserts.iter().rev() {
        out.insert_str(*at, piece);
    }
    let edited = json::parse(&out)?;
    if !edited.get("imports").and_then(Value::as_array).is_some_and(|i| i.iter().any(|g| g.as_str() == Some(GUID))) {
        return Err("the string's GUID is missing from the result's imports".into());
    }
    Ok(out)
}

/// The leading tabs of a line.
pub(super) fn indent_of(line: &str) -> &str {
    &line[..line.len() - line.trim_start_matches('\t').len()]
}

/// The line `i` of a height, `"height": <number>,`, with `grow` added to it, as Python's repr
/// writes the sum (`round_to`: rounded to that many decimals first).
fn taller(lines: &mut [String], i_name: usize, heights: impl IntoIterator<Item = usize>, grow: f64, round_to: Option<i32>, what: &str) -> Result<(), String> {
    let i = heights.into_iter().find(|&i| lines[i].trim().starts_with("\"height\": "));
    let parsed = i.and_then(|i| {
        let line = &lines[i];
        let rest = line.trim_start_matches('\t');
        let tabs = line.len() - rest.len();
        let value = rest.strip_prefix("\"height\": ")?;
        let (number, comma) = value.strip_suffix(',').map_or((value, ""), |n| (n, ","));
        (tabs > 0 && !number.is_empty() && number.bytes().all(|b| b.is_ascii_digit() || b == b'.')).then(|| (i, tabs, number.parse::<f64>().ok(), comma))
    });
    let Some((i, tabs, Some(old), comma)) = parsed else {
        return Err(format!("{what}: no height of its own next to its name (line {})", i_name + 1));
    };
    if (i.min(i_name) + 1..i.max(i_name)).any(|k| lines[k].contains("\"name\": \"")) {
        return Err(format!("{what}: no height of its own next to its name (line {})", i_name + 1));
    }
    let new = match round_to {
        Some(d) => format!("{:.*}", d as usize, old + grow).parse::<f64>().expect("a number"),
        None => old + grow,
    };
    lines[i] = format!("{}\"height\": {}{comma}", "\t".repeat(tabs), json::float(new));
    Ok(())
}

/// The lines' container, the frame drawn behind the lines, the dark panel and the glow behind the
/// menu, taller by `grow`: they hang from the bottom of the screen, so they grow upwards. The
/// container's height follows its name; the three others' come before theirs.
pub fn grow_menu(lines: &mut [String], r: &Recipe, grow: f64, round_to: Option<i32>) -> Result<(), String> {
    let everywhere = 0..lines.len();
    let container = r.one("title.container")?;
    let i_container = one(lines, everywhere.clone(), |l| l.trim() == format!("\"name\": \"{container}\","), "the lines' container")?;
    taller(lines, i_container, i_container..lines.len(), grow, round_to, "the lines' container")?;
    for part in r.all("title.behind")? {
        let i_part = one(lines, everywhere.clone(), |l| l.trim() == format!("\"name\": \"{part}\","), part)?;
        taller(lines, i_part, (0..i_part).rev(), grow, round_to, part)?;
    }
    Ok(())
}

/// The level's imports, lines `i_imports + 1..i_end`, with `new` GUIDs added at their sorted
/// places (engine types first, the others sorted).
pub fn add_imports(lines: &mut Vec<String>, text: &str, new: &[&str]) -> Result<(), String> {
    let i_imports = one(lines, 0..lines.len(), |l| l == "\t\"imports\": [", "the imports")?;
    let i_end = (i_imports..lines.len()).find(|&i| lines[i] == "\t],").ok_or("the imports do not end")?;
    let imports: Vec<String> = lines[i_imports + 1..i_end].iter().map(|l| l.trim().trim_end_matches(',').trim_matches('"').to_string()).collect();
    let parsed: Vec<String> = json::parse(text)?.get("imports").and_then(Value::as_array).ok_or("no imports")?.iter().filter_map(|g| g.as_str().map(String::from)).collect();
    if imports.iter().any(|g| !is_guid(g)) || imports != parsed {
        return Err("the imports are not one GUID per line".into());
    }
    let first = imports.iter().position(|g| !is_native(g)).ok_or("no import but engine types")?;
    let mut rest: Vec<String> = imports[first..].to_vec();
    if rest.iter().any(|g| is_native(g)) || !rest.windows(2).all(|w| import_key(&w[0]) <= import_key(&w[1])) {
        return Err("the imports are not in the order this tool knows (engine types first, the others sorted)".into());
    }
    let mut added: Vec<&str> = new.iter().copied().filter(|g| !imports.iter().any(|i| i == g)).collect();
    added.sort_by_key(|g| import_key(g));
    added.dedup();
    for g in added {
        let at = rest.iter().position(|r| import_key(r) > import_key(g)).unwrap_or(rest.len());
        rest.insert(at, g.to_string());
    }
    let all: Vec<String> = imports[..first].iter().cloned().chain(rest).collect();
    let n = all.len();
    lines.splice(i_imports + 1..i_end, all.iter().enumerate().map(|(k, g)| format!("\t\t\"{g}\"{}", if k + 1 < n { "," } else { "" })));
    Ok(())
}

/// QUIT GAME in the title menu: edits the decoded main menu level. Under TRAINING the level holds
/// a line only the store demo shows (its box collapsed, a filter spawning it in the demo alone);
/// it becomes QUIT GAME: spawned in the full game, its box shown, our string as its label, number
/// 06, its click an event of the demo's script whose handler the "quit" hook takes. Our string
/// and the number 06 enter the imports; with `grow`, the menu's boxes make room for the line.
pub fn edit_title(text: &str, r: &Recipe, grow: f64) -> Result<String, String> {
    let (old_text, num_05, num_06) = (r.one("title.old_text")?, r.one("title.num_05")?, r.one("title.num_06")?);
    let (before, line, filter) = (r.one("title.before")?, r.one("title.line")?, r.one("title.filter")?);
    let (old_event, new_event) = (r.one("title.old_event")?, r.one("title.new_event")?);
    let mut lines: Vec<String> = text.split('\n').map(String::from).collect();
    let everywhere = 0..lines.len();
    let is = |l: &str, want: String| l.trim() == want;
    // The demo line's entity: its parts come before its name, after the previous line's name.
    let first = one(&lines, everywhere.clone(), |l| is(l, format!("\"name\": \"{before}\",")), "the name of the line before")? + 1;
    let name = one(&lines, everywhere.clone(), |l| is(l, format!("\"name\": \"{line}\",")), "the demo line's name")?;
    if first >= name {
        return Err(format!("the demo line does not follow {before}"));
    }
    let i_text = one(&lines, first..name, |l| is(l, format!("\"handle\": \"{old_text}\",")), "the label")?;
    let i_num = one(&lines, first..name, |l| is(l, format!("\"handle\": \"{num_05}\",")), "the number")?;
    let i_event = one(&lines, first..name, |l| is(l, format!("\"event_name\": \"{old_event}\"")), "the click's event")?;
    let i_box = one(&lines, first..name, |l| l.trim() == "\"collapsed\": true,", "the collapsed box")?;
    let i_filter = one(&lines, first..name, |l| is(l, format!("\"cvar_name\": \"{filter}\",")), "the filter")?;
    if !(i_text < i_num && i_num < i_event && i_event < i_box && i_box < i_filter) {
        return Err("the demo line's parts are not in the order this tool knows".into());
    }
    // The imports come after the entities: their new lines leave the demo line's indexes right.
    if one(&lines, everywhere, |l| l == "\t\"imports\": [", "the imports")? < i_filter {
        return Err("the imports come before the demo line: not the entry this tool knows".into());
    }
    add_imports(&mut lines, text, &[GUID, num_06])?;
    if grow != 0.0 {
        grow_menu(&mut lines, r, grow, None)?;
    }
    // The demo line, from its last part to its first, so the indexes above stay right.
    let indent = indent_of(&lines[i_filter]).to_string();
    lines.splice(i_filter..i_filter, [format!("{indent}\"value\": {{"), format!("{indent}\t\"bool\": false"), format!("{indent}}},")]);
    lines.remove(i_box);
    lines[i_event] = lines[i_event].replace(old_event, new_event);
    lines[i_num] = lines[i_num].replace(num_05, num_06);
    lines[i_text] = lines[i_text].replace(old_text, GUID);
    let out = lines.join("\n");
    json::no_key_twice(&out)?;
    let imports: Vec<Value> = json::parse(&out)?.get("imports").and_then(Value::as_array).cloned().unwrap_or_default();
    let has = |g: &str| imports.iter().any(|i| i.as_str() == Some(g));
    let unique: std::collections::HashSet<&Value> = imports.iter().collect();
    if !has(GUID) || !has(num_06) || unique.len() != imports.len() {
        return Err("the result's imports are wrong".into());
    }
    Ok(out)
}

/// The dump's build settings (a JSON object) with `key` set: written as the layer has it, one
/// key per line after a tab, LF line ends.
pub fn with_cvar(text: &str, key: &str) -> Result<String, String> {
    let v = json::parse(text).map_err(|e| format!("veclipse_build.cvar: {e}"))?;
    let obj = v.as_object().ok_or("veclipse_build.cvar is not a JSON object")?;
    if obj.contains_key(key) {
        return Err(format!("veclipse_build.cvar already sets {key}: not the dump this tool knows"));
    }
    let lines: Vec<String> = obj.iter().map(|(k, v)| format!("\t{}: {}", json::string(k, true), json::render(v, 0, true))).chain([format!("\t{}: true", json::string(key, true))]).collect();
    Ok(format!("{{\n{}\n}}\n", lines.join(",\n")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic table of two strings (no game data), in the file's layout: the string added
    /// after the last, its GUID listed twice, every other byte kept; taken out again, the table is
    /// back. Refused: a second run, a template whose regions are out of order.
    #[test]
    fn a_string_is_added_to_a_table() {
        let regions: Vec<Value> = (0..texts::REGIONS)
            .map(|r| {
                let tag = texts::tag(r);
                let (l, c) = tag.split_once('-').unwrap();
                if r == 0 { json!({"version": 1, "text": format!("t{r}")}) } else { json!({"version": 1, "language": {"country": format!("k_country_{}", c.to_lowercase()), "language": format!("k_language_{l}")}, "text": format!("t\"{r}")}) }
            })
            .collect();
        let string = |guid: &str, id: &str| json!({"version_hash": 7, "$guid": guid, "$name": id.to_lowercase(), "vtable": "v", "guid": guid, "id": id, "type": "k_locs_translation_type_translate", "translations": regions});
        let g = ["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002", "00000000-0000-0000-0000-000000000003"];
        let doc = json!({"segments": [{"exported_objects": [{"strings": [g[1], g[2]], "$guid": g[0], "path": "p"}, string(g[1], "TEMPLATE"), string(g[2], "OTHER")], "name": "default"}], "version": "vjson-1", "exports": g});
        let before = render(&doc, 0, false);
        let texts: Vec<&str> = (0..texts::REGIONS).map(|r| texts::text(Text::QuitGame, r)).collect();
        let after = add_string(&before, "TEMPLATE", &texts).unwrap();
        assert_eq!(after, render(&json::parse(&after).unwrap(), 0, false), "written as the file writes itself");
        let mut parsed = json::parse(&after).unwrap();
        let objects = parsed.pointer_mut("/segments/0/exported_objects").unwrap().as_array_mut().unwrap();
        let added = objects.pop().unwrap();
        assert_eq!((added["id"].as_str(), added["$guid"].as_str(), added["guid"].as_str(), added["$name"].as_str()), (Some(ID), Some(GUID), Some(GUID), Some("gyroship_quit_game")));
        assert_eq!(added["version_hash"], json!(crc32c(b"QUIT GAME")));
        assert_eq!(added["translations"].as_array().unwrap().iter().map(|t| t["text"].as_str().unwrap()).collect::<Vec<_>>(), texts);
        assert_eq!(objects[0]["strings"].as_array_mut().unwrap().pop(), Some(json!(GUID)));
        parsed["exports"].as_array_mut().unwrap().pop();
        assert_eq!(parsed, doc);
        assert!(add_string(&after, "TEMPLATE", &texts).unwrap_err().contains("already in the table"));
        let mut swapped = doc.clone();
        swapped.pointer_mut("/segments/0/exported_objects/1/translations").unwrap().as_array_mut().unwrap().swap(2, 3);
        assert!(add_string(&render(&swapped, 0, false), "TEMPLATE", &texts).unwrap_err().contains("region order"));
    }

    /// The build settings: the key added after the dump's, in the layer's layout.
    #[test]
    fn the_timer_is_set_in_the_build_settings() {
        assert_eq!(with_cvar("{\r\n\t\"a.b\": true\r\n}\r\n", "c.d").unwrap(), "{\n\t\"a.b\": true,\n\t\"c.d\": true\n}\n");
        assert!(with_cvar("{\"c.d\": false}", "c.d").unwrap_err().contains("already"));
        assert!(with_cvar("[]", "c.d").is_err());
    }

    #[test]
    fn string_literals_and_guid_words_are_told() {
        assert!(is_string_literal("\"a\\\"b\"") && is_string_literal("\"\"") && !is_string_literal("\"a\"b\"") && !is_string_literal("\"a\\\""));
        assert_eq!(quoted_guids("x \"0981fca4-2198-11ee-98de-d800c63bd261\", \"no\""), ["0981fca4-2198-11ee-98de-d800c63bd261"]);
    }
}
