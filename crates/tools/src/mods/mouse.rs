//! The tutorial's mouse question within a minute or two of flight (layer-mouse), ported from
//! mklayer.py. On the practice layer, the tutorial's root level starts with its sequence 1 (the
//! first two objects of its sequence list swapped), and sequence 1's lessons before the mouse
//! introduction no longer require their objectives, so that their gates let the player through.

use super::{Archive, Layer, Recipe};

const ABOUT: &str = "Reach the tutorial's mouse question quickly (new save, Joy-Con with mouse mode)";

/// The mouse layer: the practice layer and the tutorial's two entries.
pub fn layer(r: &Recipe, zip: &mut Archive, base: &Layer) -> Result<Layer, String> {
    let text = |bytes: Vec<u8>, what: &str| String::from_utf8(bytes).map_err(|e| format!("{what} is not UTF-8: {e}"));
    let (root_name, root) = zip.named(r.one("mouse.root")?)?;
    let root = swap_first_two(&text(root, "the tutorial's root level")?, r.one("mouse.list")?, r.all("mouse.first")?).map_err(|e| format!("the tutorial's root level: {e}"))?;
    let (sequence_name, sequence) = zip.named(r.one("mouse.sequence")?)?;
    let gates: usize = r.one("mouse.gates")?.parse().map_err(|e| format!("mods.txt: mouse.gates: {e}"))?;
    let sequence = open_gates(&text(sequence, "the tutorial's sequence 1")?, r.one("mouse.gate")?, gates).map_err(|e| format!("the tutorial's sequence 1: {e}"))?;
    let mut layer = base.clone();
    layer.about = ABOUT.into();
    layer.entries.insert(root_name, root.into_bytes());
    layer.entries.insert(sequence_name, sequence.into_bytes());
    Ok(layer)
}

/// The list `key`'s first two objects swapped, found as the script found them: braces counted from
/// the list's opening bracket, until four objects; the first two must name `names`, in order.
pub fn swap_first_two(t: &str, key: &str, names: &[String]) -> Result<String, String> {
    let [first, second] = names else { return Err("mods.txt: mouse.first wants two names".into()) };
    let at = t.find(&format!("\"{key}\": [")).ok_or_else(|| format!("no list {key}"))?;
    let b = t.as_bytes();
    let mut k = at + t[at..].find('[').expect("the list's bracket");
    let (mut objects, mut depth, mut start) = (Vec::new(), 0usize, 0usize);
    while objects.len() < 4 {
        k += 1;
        match b.get(k) {
            None => return Err(format!("the list {key} has fewer than four objects")),
            Some(b'{') => {
                if depth == 0 {
                    start = k;
                }
                depth += 1;
            }
            Some(b'}') => {
                depth = depth.checked_sub(1).ok_or_else(|| format!("the list {key} closes more than it opens"))?;
                if depth == 0 {
                    objects.push((start, k + 1));
                }
            }
            _ => {}
        }
    }
    let ((a0, b0), (a1, b1)) = (objects[0], objects[1]);
    if !t[a0..b0].contains(first.as_str()) || !t[a1..b1].contains(second.as_str()) {
        return Err(format!("the list {key} does not start with {first} then {second}"));
    }
    Ok([&t[..a0], &t[a1..b1], &t[b0..a1], &t[a0..b0], &t[b1..]].concat())
}

/// Every `"<gate>": true` made false: exactly `count` of them.
pub fn open_gates(t: &str, gate: &str, count: usize) -> Result<String, String> {
    let shut = format!("\"{gate}\": true");
    match t.matches(&shut).count() {
        n if n == count => Ok(t.replace(&shut, &format!("\"{gate}\": false"))),
        n => Err(format!("{n} lessons require their objectives, {count} expected")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_two_objects_of_a_list_trade_places() {
        let t = "{\"sequences\": [{\"p\": \"s0\", \"x\": {}}, {\"p\": \"s1\"}, {\"p\": \"s2\"}, {\"p\": \"s3\"}], \"z\": 1}";
        let names = ["s0".to_string(), "s1".to_string()];
        assert_eq!(swap_first_two(t, "sequences", &names).unwrap(), "{\"sequences\": [{\"p\": \"s1\"}, {\"p\": \"s0\", \"x\": {}}, {\"p\": \"s2\"}, {\"p\": \"s3\"}], \"z\": 1}");
        assert!(swap_first_two(t, "sequences", &["s1".to_string(), "s0".to_string()]).is_err());
        assert!(swap_first_two("\"sequences\": [{}, {}]", "sequences", &names).unwrap_err().contains("fewer than four"));
        assert!(swap_first_two("{\"sequences\": [{}, {}]}", "sequences", &names).unwrap_err().contains("closes more than it opens"));
        assert_eq!(open_gates("\"g\": true, \"g\": true", "g", 2).unwrap(), "\"g\": false, \"g\": false");
        assert!(open_gates("\"g\": true", "g", 2).unwrap_err().contains("1 lessons"));
    }
}
