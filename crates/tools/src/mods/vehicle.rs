//! Forced vehicle (layer-arwing, layer-landmaster, layer-bluemarine), ported from the scripts of
//! Data/practice/vehicle (build.py, closure.py, swap.py, package.py). In each layer, every
//! mission whose own vehicle is another spawns the layer's: its root level's player spawner names
//! that vehicle's prefab level (`swap`), and its package list loads what the prefab needs
//! (`package`): the files found by following the prefab's imports through the package list of a
//! mission where the vehicle is the mission's own (`closure`).

use super::json;
use super::quit::indent_of;
use super::{Archive, Layer, Recipe, ascii, import_key, is_guid, is_native};
use serde_json::{Value, json};
use std::collections::{BTreeSet, HashMap};

/// A package list's six arrays, one value per file, in dependency order.
const ARRAYS: [&str; 6] = ["list", "guids", "masks", "sizes", "align_log2s", "parents"];
/// A file's mask bit for a soft reference: listed, not loaded.
const SOFT: i64 = 8;

/// What a file of the archive says of itself: its exports and imports when it is JSON; a file
/// that starts as JSON but does not read as such; anything else (raw data, binary, absent).
pub enum Desc {
    Json { exports: Vec<String>, imports: Vec<String> },
    Unparsed,
    Other,
}

/// The description of the file at `path`.
fn describe(zip: &mut Archive, path: &str) -> Result<Desc, String> {
    if path.ends_with(".raw") {
        return Ok(Desc::Other);
    }
    let Some(bytes) = zip.whole_if(path, b'{')? else { return Ok(Desc::Other) };
    if bytes.iter().find(|b| !b" \t\n\r\x0b\x0c".contains(b)) != Some(&b'{') {
        return Ok(Desc::Other);
    }
    let Ok(doc) = serde_json::from_slice::<Value>(&bytes) else { return Ok(Desc::Unparsed) };
    let strings = |key: &str| doc.get(key).and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect()).unwrap_or_default();
    Ok(Desc::Json { exports: strings("exports"), imports: strings("imports") })
}

/// A decoded package list's arrays.
pub struct List {
    list: Vec<String>,
    guids: Vec<String>,
    masks: Vec<i64>,
    sizes: Vec<i64>,
    aligns: Vec<i64>,
    parents: Vec<i64>,
}

/// The arrays of a decoded package list, read as JSON values.
fn arrays(doc: &Value) -> Result<Vec<&Vec<Value>>, String> {
    let obj = doc.pointer("/segments/0/exported_objects/0").ok_or("no package list in it")?;
    let found: Vec<&Vec<Value>> = ARRAYS.iter().map(|a| obj.get(a).and_then(Value::as_array).ok_or_else(|| format!("no array {a}"))).collect::<Result<_, _>>()?;
    if found.iter().any(|a| a.len() != found[0].len()) {
        return Err("the six arrays do not have the same length".into());
    }
    Ok(found)
}

fn read_list(text: &str) -> Result<List, String> {
    let doc = json::parse(text)?;
    let a = arrays(&doc)?;
    let strings = |v: &Vec<Value>| v.iter().map(|x| x.as_str().map(String::from).ok_or("not a string")).collect::<Result<Vec<_>, _>>();
    let ints = |v: &Vec<Value>| v.iter().map(|x| x.as_i64().ok_or("not an integer")).collect::<Result<Vec<_>, _>>();
    Ok(List { list: strings(a[0])?, guids: strings(a[1])?, masks: ints(a[2])?, sizes: ints(a[3])?, aligns: ints(a[4])?, parents: ints(a[5])? })
}

/// Where the tail of raw data starts: after the last file that is not raw data.
fn tail_of(list: &[impl AsRef<str>]) -> Option<usize> {
    list.iter().rposition(|p| !p.as_ref().ends_with(".raw")).map(|i| i + 1)
}

/// A file of a vehicle's closure, as its home mission's package list holds it; `tail` when it
/// is in that list's tail of raw data.
pub struct Row {
    path: String,
    guid: String,
    mask: i64,
    size: i64,
    align: i64,
    parent: String,
    tail: bool,
}

/// What the file `root` needs in a package list, by its own imports: an import is a GUID, a
/// file's own (the list's guids) or one of the objects a file exports; a file's children (the
/// parent links) come with it; imports written "?N:GUID" are soft references, not followed.
pub fn closure(list: &List, root: &str, cache: &HashMap<String, Desc>) -> Result<Vec<Row>, String> {
    let root = format!("/{}", root.trim_start_matches('/'));
    let r = list.list.iter().position(|p| *p == root).ok_or_else(|| format!("{root} is not in the package list"))?;
    let mut by_guid: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, g) in list.guids.iter().enumerate() {
        by_guid.entry(g).or_default().push(i);
    }
    for (i, p) in list.list.iter().enumerate() {
        if let Some(Desc::Json { exports, .. }) = cache.get(p) {
            for g in exports {
                by_guid.entry(g).or_default().push(i);
            }
        }
    }
    let mut children = vec![Vec::new(); list.list.len()];
    for (i, &p) in list.parents.iter().enumerate() {
        if p >= 0 {
            children[p as usize].push(i);
        }
    }
    let (mut seen, mut todo) = (BTreeSet::new(), vec![r]);
    while let Some(i) = todo.pop() {
        if !seen.insert(i) {
            continue;
        }
        todo.extend(&children[i]);
        match cache.get(&list.list[i]) {
            Some(Desc::Json { imports, .. }) => todo.extend(imports.iter().filter(|g| !g.starts_with('?')).filter_map(|g| by_guid.get(g.as_str())).flatten()),
            Some(Desc::Unparsed | Desc::Other) => {}
            None => return Err(format!("{} is not described", list.list[i])),
        }
    }
    let tail = tail_of(&list.list).unwrap_or(0);
    Ok(seen
        .into_iter()
        .map(|i| Row {
            path: list.list[i].clone(),
            guid: list.guids[i].clone(),
            mask: list.masks[i],
            size: list.sizes[i],
            align: list.aligns[i],
            parent: if list.parents[i] >= 0 { list.list[list.parents[i] as usize].clone() } else { String::new() },
            tail: i >= tail,
        })
        .collect())
}

/// A GUID import, as a level lists it: an optional soft-reference prefix ("?N:"), the GUID.
fn import_line(line: &str) -> Option<(&str, &str, bool)> {
    let inner = line.strip_prefix("\t\t\"")?;
    let (inner, comma) = inner.strip_suffix("\",").map(|s| (s, true)).or_else(|| inner.strip_suffix('"').map(|s| (s, false)))?;
    let (prefix, guid) = match inner.strip_prefix('?').and_then(|s| s.split_once(':')) {
        Some((n, guid)) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => (&inner[..n.len() + 2], guid),
        Some(_) => return None,
        None => ("", inner),
    };
    is_guid(guid).then_some((prefix, guid, comma))
}

/// A mission's root level with its player spawner naming the prefab level `new` (one of
/// `prefabs`), which becomes a plain import of the level at its sorted place (the old one stays:
/// levels carry imports nothing refers to). Text edits only; the spawner line found once.
pub fn swap(text: &str, key: &str, new: &str, prefabs: &[&str]) -> Result<String, String> {
    if text.contains('\r') {
        return Err("the input has CR line ends; the dump's entry has LF only".into());
    }
    let mut lines: Vec<String> = text.split('\n').map(String::from).collect();
    let hits: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].contains(key)).collect();
    let [hit] = hits[..] else { return Err(format!("{} lines name {key}, expected exactly one", hits.len())) };
    let line = &lines[hit];
    let rest = line.trim_start_matches('\t');
    let tabs = line.len() - rest.len();
    let value = rest.strip_prefix(&format!("\"{key}\": \"")).filter(|_| tabs > 0);
    let parsed = value.and_then(|v| v.strip_suffix("\",").map(|g| (g, ",")).or_else(|| v.strip_suffix('"').map(|g| (g, ""))));
    let Some((old, comma)) = parsed.filter(|(g, _)| is_guid(g) && prefabs.contains(g)) else {
        return Err(format!("the spawner line is not one this tool knows: {:?}", line.trim()));
    };
    if old == new {
        return Err("the level already spawns that vehicle".into());
    }
    let old = old.to_string();
    lines[hit] = format!("{}\"{key}\": \"{new}\"{comma}", "\t".repeat(tabs));

    // The imports: engine types first, then every other GUID sorted; some carry a "?N:" prefix.
    let starts: Vec<usize> = (0..lines.len()).filter(|&i| lines[i] == "\t\"imports\": [").collect();
    let [i_imports] = starts[..] else { return Err(format!("{} imports lists, expected exactly one", starts.len())) };
    let i_end = (i_imports..lines.len()).find(|&i| lines[i] == "\t]," || lines[i] == "\t]").ok_or("the imports do not end")?;
    let imports: Vec<(String, String)> = lines[i_imports + 1..i_end].iter().map(|l| import_line(l).map(|(p, g, _)| (p.to_string(), g.to_string()))).collect::<Option<_>>().ok_or("the imports are not one GUID per line")?;
    let parsed: Vec<String> = json::parse(text)?.get("imports").and_then(Value::as_array).ok_or("no imports")?.iter().filter_map(|g| g.as_str().map(String::from)).collect();
    if imports.iter().map(|(p, g)| format!("{p}{g}")).collect::<Vec<_>>() != parsed {
        return Err("the imports read here are not those the parser reads".into());
    }
    let start = imports.iter().position(|(_, g)| !is_native(g)).ok_or("no import but engine types")?;
    let mut rest: Vec<(String, String)> = imports[start..].to_vec();
    if rest.iter().any(|(_, g)| is_native(g)) || !rest.windows(2).all(|w| import_key(&w[0].1) <= import_key(&w[1].1)) {
        return Err("the imports are not in the order this tool knows (engine types first, the others sorted)".into());
    }
    if !rest.iter().any(|(_, g)| *g == old) {
        return Err("the old prefab is not among the level's imports".into());
    }
    if rest.iter().any(|(_, g)| g == new) {
        for (p, g) in &mut rest {
            if g == new {
                p.clear();
            }
        }
    } else {
        let at = rest.iter().position(|(_, g)| import_key(g) > import_key(new)).unwrap_or(rest.len());
        rest.insert(at, (String::new(), new.to_string()));
    }
    let all: Vec<(String, String)> = imports[..start].iter().cloned().chain(rest).collect();
    let n = all.len();
    lines.splice(i_imports + 1..i_end, all.iter().enumerate().map(|(k, (p, g))| format!("\t\t\"{p}{g}\"{}", if k + 1 < n { "," } else { "" })));
    let out = lines.join("\n");
    json::no_key_twice(&out)?;
    let edited: Vec<Value> = json::parse(&out)?.get("imports").and_then(Value::as_array).cloned().unwrap_or_default();
    let unique: std::collections::HashSet<&Value> = edited.iter().collect();
    if !edited.iter().any(|g| g.as_str() == Some(new)) || unique.len() != edited.len() {
        return Err("the result's imports are wrong".into());
    }
    Ok(out)
}

/// A value as `json.dumps` writes it on one line.
fn compact(v: &Value) -> String {
    match v {
        Value::String(s) => json::string(s, true),
        Value::Number(n) => json::number(n),
        other => json::render(other, 0, true),
    }
}

/// A file of the new list: one of the old list's, by its index, or one of the closure's.
enum Item<'a> {
    Old(usize),
    New(&'a Row),
}

/// The path of a file of the new list.
fn path_of<'a>(item: &Item<'a>, list: &[&'a str]) -> &'a str {
    match item {
        Item::Old(x) => list[*x],
        Item::New(r) => r.path.as_str(),
    }
}

/// A mission's package list made to load the closure: the soft flag cleared on the closure files
/// it holds; the files it lacks inserted just before the root level, in their order in the home
/// mission, those of the home's tail at the end; the parents renumbered; the segment's size grown
/// by a generous bound. Every inserted file comes after what it imports (as far as `cache` tells).
/// The six arrays are written again, one element per line; every other byte stays. Returns the
/// list and a line of what was done.
pub fn package(text: &str, rows: &[Row], cache: &HashMap<String, Desc>) -> Result<(String, String), String> {
    if text.contains('\r') {
        return Err("the input has CR line ends; the dump's entry has LF only".into());
    }
    let mut lines: Vec<String> = text.split('\n').map(String::from).collect();
    let doc = json::parse(text)?;
    let old = arrays(&doc)?;
    let n = old[0].len();
    if rows.iter().any(|r| r.mask & SOFT != 0) {
        return Err("the closure holds soft-flagged files: it was not made in the vehicle's own mission".into());
    }
    // Where each array is in the text: its first element's line, checked element by element.
    let mut first = [0usize; 6];
    for (a, name) in ARRAYS.iter().enumerate() {
        let head = format!("\"{name}\": [");
        let starts: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].trim() == head).collect();
        let [at] = starts[..] else { return Err(format!("{} arrays named {name}, expected exactly one", starts.len())) };
        first[a] = at + 1;
        for (k, v) in old[a].iter().enumerate() {
            let line = lines.get(first[a] + k).ok_or_else(|| format!("{name} ends early"))?;
            if serde_json::from_str::<Value>(line.trim().trim_end_matches(',')).ok().as_ref() != Some(v) {
                return Err(format!("line {} is not element {k} of {name}", first[a] + k + 1));
            }
        }
        if !matches!(lines.get(first[a] + n).map(|l| l.trim()), Some("]" | "],")) {
            return Err(format!("{name} does not end where expected"));
        }
    }
    let list: Vec<&str> = old[0].iter().map(|v| v.as_str().unwrap_or_default()).collect();
    let index: HashMap<&str, usize> = list.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    let mut masks: Vec<i64> = old[2].iter().map(|v| v.as_i64().unwrap_or_default()).collect();
    let (mut cleared, mut other_bits) = (0, 0);
    for r in rows {
        if let Some(&i) = index.get(r.path.as_str()) {
            other_bits += usize::from((masks[i] & !SOFT) != r.mask);
            if masks[i] & SOFT != 0 {
                masks[i] &= !SOFT;
                cleared += 1;
            }
        }
    }

    // The new order: the files before the root level, the inserted ones, the root level, the
    // tail, the inserted ones of the tail.
    let tail_at = tail_of(&list).ok_or("the list is raw data only")?;
    let root = tail_at - 1;
    if !list[root].ends_with(".level") {
        return Err(format!("the last file before the tail is not a level: {}", list[root]));
    }
    let main_rows: Vec<&Row> = rows.iter().filter(|r| !index.contains_key(r.path.as_str()) && !r.tail).collect();
    let tail_rows: Vec<&Row> = rows.iter().filter(|r| !index.contains_key(r.path.as_str()) && r.tail).collect();
    let order: Vec<Item> = (0..root).map(Item::Old).chain(main_rows.iter().map(|r| Item::New(r))).chain([Item::Old(root)]).chain((tail_at..n).map(Item::Old)).chain(tail_rows.iter().map(|r| Item::New(r))).collect();
    let total = order.len();
    let new_index: HashMap<&str, usize> = order.iter().enumerate().map(|(k, item)| (path_of(item, &list), k)).collect();
    if new_index.len() != total {
        return Err("a path would be twice in the list".into());
    }
    let mut new: [Vec<Value>; 6] = Default::default();
    for item in &order {
        match item {
            Item::Old(x) => {
                for a in [0, 1, 3, 4] {
                    new[a].push(old[a][*x].clone());
                }
                new[2].push(json!(masks[*x]));
                let p = old[5][*x].as_i64().unwrap_or(-1);
                new[5].push(json!(if p >= 0 { new_index[list[p as usize]] as i64 } else { -1 }));
            }
            Item::New(r) => {
                if !r.parent.is_empty() && !new_index.contains_key(r.parent.as_str()) {
                    return Err(format!("{} has a parent outside the list and the closure: {}", r.path, r.parent));
                }
                new[0].push(json!(r.path));
                new[1].push(json!(r.guid));
                new[2].push(json!(r.mask));
                new[3].push(json!(r.size));
                new[4].push(json!(r.align));
                new[5].push(json!(if r.parent.is_empty() { -1 } else { new_index[r.parent.as_str()] as i64 }));
            }
        }
    }

    // Every inserted file after what it imports, and the root level after the prefab.
    let (mut checked, mut unresolved) = (0, 0);
    // The first file of the new list that holds each GUID, as its own or as an export.
    let mut by_guid: HashMap<&str, usize> = HashMap::new();
    for (k, g) in new[1].iter().enumerate() {
        let g = g.as_str().unwrap_or_default();
        by_guid.entry(g).and_modify(|m| *m = (*m).min(k)).or_insert(k);
    }
    for (k, p) in new[0].iter().enumerate() {
        if let Some(Desc::Json { exports, .. }) = p.as_str().and_then(|p| cache.get(p)) {
            for g in exports {
                by_guid.entry(g.as_str()).and_modify(|m| *m = (*m).min(k)).or_insert(k);
            }
        }
    }
    for r in main_rows.iter().chain(&tail_rows) {
        let Desc::Json { imports, .. } = cache.get(&r.path).ok_or_else(|| format!("{} is not described", r.path))? else { continue };
        let k = new_index[r.path.as_str()];
        for g in imports.iter().filter(|g| !g.starts_with('?')) {
            match by_guid.get(g.as_str()) {
                None => unresolved += 1,
                Some(&m) => {
                    checked += 1;
                    if m >= k {
                        return Err(format!("{} (#{k}) would come before a file it imports (#{m} {})", r.path, path_of(&order[m], &list)));
                    }
                }
            }
        }
    }
    if rows.iter().filter(|r| !r.tail).any(|r| new_index[r.path.as_str()] >= new_index[list[root]]) {
        return Err("a closure file would come after the root level".into());
    }

    // The text: each array written again, from the last of the file to the first.
    let added = main_rows.len() + tail_rows.len();
    let mut by_place: Vec<usize> = (0..6).collect();
    by_place.sort_by_key(|&a| std::cmp::Reverse(first[a]));
    for a in by_place {
        let indent = indent_of(&lines[first[a]]).to_string();
        let written: Vec<String> = new[a].iter().enumerate().map(|(k, v)| format!("{indent}{}{}", compact(v), if k + 1 < total { "," } else { "" })).collect();
        lines.splice(first[a]..first[a] + n, written);
    }
    if added > 0 {
        let size_line = |l: &str| {
            let rest = l.trim_start_matches('\t');
            let digits = rest.strip_prefix("\"size\": ")?.strip_suffix(',')?;
            (rest.len() < l.len() && !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then(|| digits.parse::<u64>().ok()).flatten()
        };
        let sizes: Vec<usize> = (0..lines.len().min(60)).filter(|&i| size_line(&lines[i]).is_some()).collect();
        let [at] = sizes[..] else { return Err(format!("{} segment sizes in the head of the file, expected exactly one", sizes.len())) };
        let growth = (main_rows.iter().chain(&tail_rows).map(|r| r.path.len() as u64 + 1 + 64).sum::<u64>() + 4095) / 4096 * 4096;
        let value = size_line(&lines[at]).expect("a size") + growth;
        lines[at] = format!("{}\"size\": {value},", indent_of(&lines[at]));
    }

    let out = lines.join("\n");
    let edited_doc = json::parse(&out)?;
    let edited = arrays(&edited_doc)?;
    if (0..6).any(|a| *edited[a] != new[a]) {
        return Err("the result's arrays are not those meant".into());
    }
    // Every file the list had keeps its values and its parent.
    let at = |v: &Vec<Value>, k: usize| v[k].as_i64().unwrap_or_default();
    for (i, p) in list.iter().enumerate() {
        let k = new_index[p];
        let mask = at(edited[2], k);
        if [1, 3, 4].iter().any(|&a| edited[a][k] != old[a][i]) || (mask != at(old[2], i) && mask != at(old[2], i) & !SOFT) {
            return Err(format!("file {i} changed"));
        }
        let parent_now = usize::try_from(at(edited[5], k)).ok().and_then(|q| edited[0][q].as_str());
        let parent_was = usize::try_from(at(old[5], i)).ok().map(|q| list[q]);
        if parent_now != parent_was {
            return Err(format!("file {i} lost its parent"));
        }
    }
    if rows.iter().any(|r| at(edited[2], new_index[r.path.as_str()]) & SOFT != 0) {
        return Err("closure files are still soft-flagged".into());
    }
    if added == 0 && text.split('\n').zip(&lines).filter(|(a, b)| a != b).count() != cleared {
        return Err("more lines changed than masks cleared".into());
    }
    let mib = main_rows.iter().chain(&tail_rows).map(|r| r.size).sum::<i64>() as f64 / (1 << 20) as f64;
    let done = format!(
        "closure {} files: soft flag cleared on {cleared}; inserted {} before the root level (#{root}) and {} in the tail ({mib:.1} MiB by their sizes); {other_bits} held with other mask bits than in the source mission; order checked on {checked} imports of the inserted files ({unresolved} import no file of the list)",
        rows.len(),
        main_rows.len(),
        tail_rows.len()
    );
    Ok((out, done))
}

/// The three vehicle layers, each on the practice layer.
pub fn layers(r: &Recipe, zip: &mut Archive, base: &Layer) -> Result<Vec<(String, Layer)>, String> {
    let (level_path, package_path, prefab_path, spawner) = (r.one("vehicle.level_path")?, r.one("vehicle.package_path")?, r.one("vehicle.prefab_path")?, r.one("vehicle.spawner")?);
    // A mission's root level, its package list, its own vehicle.
    let mission = |m: &str| -> Result<(String, String, String), String> {
        let [folder, level, own] = r.all(&format!("mission.{m}"))? else { return Err(format!("mods.txt: mission.{m} wants a folder, a level and a vehicle")) };
        let level = level_path.replace("{folder}", folder).replace("{level}", level);
        Ok((level.clone(), package_path.replace("{level_path}", &level), own.clone()))
    };
    let missions: Vec<(String, (String, String, String))> = r.all("vehicle.missions")?.iter().map(|m| Ok((m.clone(), mission(m)?))).collect::<Result<_, String>>()?;
    let vehicles: Vec<(String, String, String)> = r
        .all("vehicle.list")?
        .iter()
        .map(|v| match r.all(&format!("vehicle.{v}"))? {
            [guid, home] => Ok((v.clone(), guid.clone(), home.clone())),
            _ => Err(format!("mods.txt: vehicle.{v} wants a prefab GUID and a home mission")),
        })
        .collect::<Result<_, String>>()?;
    let prefabs: Vec<&str> = vehicles.iter().map(|(_, g, _)| g.as_str()).collect();

    // The home missions' package lists, and the description of every file they list.
    let mut cache: HashMap<String, Desc> = HashMap::new();
    let mut homes: HashMap<String, List> = HashMap::new();
    for (_, _, home) in &vehicles {
        let (_, package, _) = mission(home)?;
        let (_, bytes) = zip.at(&package)?;
        let list = read_list(&ascii(bytes, &format!("{home}'s package list"))?).map_err(|e| format!("{home}'s package list: {e}"))?;
        for p in &list.list {
            if !cache.contains_key(p) {
                let d = describe(zip, p)?;
                cache.insert(p.clone(), d);
            }
        }
        homes.insert(home.clone(), list);
    }
    let count = |want: fn(&Desc) -> bool| cache.values().filter(|d| want(d)).count();
    println!("  described {} files of the home missions: {} JSON, {} that start as JSON but do not read as such", cache.len(), count(|d| matches!(d, Desc::Json { .. })), count(|d| matches!(d, Desc::Unparsed)));
    let mut out = Vec::new();
    for (v, guid, home) in &vehicles {
        let rows = closure(&homes[home], &prefab_path.replace("{vehicle}", v), &cache).map_err(|e| format!("{v}'s closure: {e}"))?;
        let mut layer = base.clone();
        layer.about = format!("Force the {} as the vehicle", match v.as_str() {
            "arwing" => "Arwing",
            "landmaster" => "Landmaster",
            "bluemarine" => "Blue-Marine",
            other => other,
        });
        for (m, (level, list_path, own)) in &missions {
            if own == v {
                continue;
            }
            let (level_name, bytes) = zip.at(level)?;
            let swapped = swap(&ascii(bytes, &format!("{m}'s level"))?, spawner, guid, &prefabs).map_err(|e| format!("{v}, {m}'s level: {e}"))?;
            let (package_name, bytes) = zip.at(list_path)?;
            let (listed, done) = package(&ascii(bytes, &format!("{m}'s package list"))?, &rows, &cache).map_err(|e| format!("{v}, {m}'s package list: {e}"))?;
            println!("  {v}, {m}: level {level_name}, package list {package_name}; {done}");
            layer.entries.insert(level_name, swapped.into_bytes());
            layer.entries.insert(package_name, listed.into_bytes());
        }
        out.push((format!("layer-{v}"), layer));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_lines_are_read_with_their_soft_prefix() {
        assert_eq!(import_line("\t\t\"?12:0981fca4-2198-11ee-98de-d800c63bd261\","), Some(("?12:", "0981fca4-2198-11ee-98de-d800c63bd261", true)));
        assert_eq!(import_line("\t\t\"0981fca4-2198-11ee-98de-d800c63bd261\""), Some(("", "0981fca4-2198-11ee-98de-d800c63bd261", false)));
        assert_eq!(import_line("\t\t\"?x:0981fca4-2198-11ee-98de-d800c63bd261\","), None);
        assert_eq!(import_line("\t\"0981fca4-2198-11ee-98de-d800c63bd261\","), None);
    }

    /// A synthetic package list (no game data): the closure file it lacks inserted before its
    /// root level, the soft flag cleared on the one it holds, the parents renumbered.
    #[test]
    fn a_package_list_loads_the_closure() {
        let doc = json!({"segments": [{"size": 4096, "exported_objects": [{"list": ["/a.level", "/b.prefab", "/root.level", "/t.raw"], "guids": ["g-a", "g-b", "g-r", "g-t"], "masks": [1, 9, 1, 1], "sizes": [10, 20, 30, 40], "align_log2s": [4, 4, 4, 4], "parents": [-1, 0, -1, 2]}]}]});
        let text = json::render(&doc, 0, true);
        let row = |path: &str, guid: &str, parent: &str, tail: bool| Row { path: path.into(), guid: guid.into(), mask: 1, size: 5, align: 4, parent: parent.into(), tail };
        let rows = [row("/b.prefab", "g-b", "/a.level", false), row("/v.prefab", "g-v", "/b.prefab", false), row("/v.raw", "g-w", "", true)];
        let mut cache = HashMap::new();
        cache.insert("/v.prefab".to_string(), Desc::Json { exports: vec![], imports: vec!["g-b".into(), "?1:g-x".into()] });
        cache.insert("/v.raw".to_string(), Desc::Other);
        let (out, done) = package(&text, &rows, &cache).unwrap();
        let edited = json::parse(&out).unwrap();
        let list = edited.pointer("/segments/0/exported_objects/0").unwrap();
        assert_eq!(list["list"], json!(["/a.level", "/b.prefab", "/v.prefab", "/root.level", "/t.raw", "/v.raw"]));
        assert_eq!(list["masks"], json!([1, 1, 1, 1, 1, 1]));
        assert_eq!(list["parents"], json!([-1, 0, 1, -1, 3, -1]));
        assert_eq!(edited["segments"][0]["size"], json!(8192));
        assert!(done.contains("soft flag cleared on 1; inserted 1 before the root level (#2) and 1 in the tail"), "{done}");
        assert!(done.contains("order checked on 1 imports"), "{done}");
        // Out of order: the inserted file imports the root level's GUID.
        cache.insert("/v.prefab".to_string(), Desc::Json { exports: vec![], imports: vec!["g-r".into()] });
        assert!(package(&text, &rows, &cache).unwrap_err().contains("would come before a file it imports"));
    }
}
