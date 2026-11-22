//! JSON as the mods' entries hold it, and as Python's json module wrote the edits these ports
//! reproduce byte for byte: `json.dumps(value, indent="\t")`, its strings, its floats (`repr`).
//! Parsing keeps the keys' order (serde_json's preserve_order), as Python's dicts do.

use serde::Deserialize;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Number, Value};
use std::collections::HashSet;
use std::fmt;

pub fn parse(text: &str) -> Result<Value, String> {
    serde_json::from_str(text).map_err(|e| format!("not JSON: {e}"))
}

/// That `text` is JSON in which no object holds the same key twice (a parser, serde_json's as
/// Python's, would keep the last one without a word).
pub fn no_key_twice(text: &str) -> Result<(), String> {
    let mut d = serde_json::Deserializer::from_str(text);
    NoTwice::deserialize(&mut d).and_then(|_| d.end()).map_err(|e| format!("not JSON, or an object with a key twice: {e}"))
}

struct NoTwice;

impl<'de> Deserialize<'de> for NoTwice {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<NoTwice, D::Error> {
        d.deserialize_any(NoTwice)
    }
}

impl<'de> Visitor<'de> for NoTwice {
    type Value = NoTwice;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("JSON")
    }
    fn visit_bool<E: de::Error>(self, _: bool) -> Result<NoTwice, E> {
        Ok(NoTwice)
    }
    fn visit_i64<E: de::Error>(self, _: i64) -> Result<NoTwice, E> {
        Ok(NoTwice)
    }
    fn visit_u64<E: de::Error>(self, _: u64) -> Result<NoTwice, E> {
        Ok(NoTwice)
    }
    fn visit_f64<E: de::Error>(self, _: f64) -> Result<NoTwice, E> {
        Ok(NoTwice)
    }
    fn visit_str<E: de::Error>(self, _: &str) -> Result<NoTwice, E> {
        Ok(NoTwice)
    }
    fn visit_unit<E: de::Error>(self) -> Result<NoTwice, E> {
        Ok(NoTwice)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<NoTwice, A::Error> {
        while seq.next_element::<NoTwice>()?.is_some() {}
        Ok(NoTwice)
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<NoTwice, A::Error> {
        let mut keys = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(de::Error::custom(format!("the key {key:?} twice")));
            }
            map.next_value::<NoTwice>()?;
        }
        Ok(NoTwice)
    }
}

/// `json.dumps(value, indent="\t", ensure_ascii=ascii)`, each of its lines after `depth` tabs:
/// how the scripts rendered an object to find it in its file, or to write it there.
pub fn render(value: &Value, depth: usize, ascii: bool) -> String {
    let mut out = "\t".repeat(depth);
    write(value, depth, ascii, &mut out);
    out
}

fn write(value: &Value, level: usize, ascii: bool, out: &mut String) {
    let open = |out: &mut String, i: usize, level: usize| {
        out.push_str(if i == 0 { "\n" } else { ",\n" });
        out.push_str(&"\t".repeat(level + 1));
    };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&number(n)),
        Value::String(s) => out.push_str(&string(s, ascii)),
        Value::Array(items) if items.is_empty() => out.push_str("[]"),
        Value::Object(map) if map.is_empty() => out.push_str("{}"),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                open(out, i, level);
                write(item, level + 1, ascii, out);
            }
            out.push('\n');
            out.push_str(&"\t".repeat(level));
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                open(out, i, level);
                out.push_str(&string(key, ascii));
                out.push_str(": ");
                write(item, level + 1, ascii, out);
            }
            out.push('\n');
            out.push_str(&"\t".repeat(level));
            out.push('}');
        }
    }
}

/// A number as Python writes it: an integer as it is, a float as its repr.
pub fn number(n: &Number) -> String {
    match (n.as_i64(), n.as_u64()) {
        (Some(i), _) => i.to_string(),
        (_, Some(u)) => u.to_string(),
        _ => float(n.as_f64().unwrap_or(f64::NAN)),
    }
}

/// Python's repr of a float: the shortest digits that read back as it, written out from 1e-4 to
/// below 1e16, else with an exponent of two digits at least ("1e-05", "1.5e+16").
pub fn float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    // Rust's `{:e}` writes the same shortest digits: "-6.0666666984558105e1", "0e0", "1e16".
    let s = format!("{x:e}");
    let (mantissa, exp) = s.split_once('e').expect("an exponent");
    let exp: i32 = exp.parse().expect("an exponent");
    let (sign, mantissa) = mantissa.strip_prefix('-').map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        if exp < 0 {
            return format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize));
        }
        let whole = exp as usize + 1;
        if digits.len() > whole {
            format!("{sign}{}.{}", &digits[..whole], &digits[whole..])
        } else {
            format!("{sign}{digits}{}.0", "0".repeat(whole - digits.len()))
        }
    } else {
        let (first, rest) = digits.split_at(1);
        let rest = if rest.is_empty() { String::new() } else { format!(".{rest}") };
        format!("{sign}{first}{rest}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// A string as Python's json writes it: with `ascii` (its default), every character outside
/// printable ASCII escaped; else only the control characters, the quote and the backslash.
pub fn string(s: &str, ascii: bool) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ascii && !(' '..='~').contains(&c) => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Python's repr, for the floats its scripts wrote.
    #[test]
    fn floats_are_written_as_python_writes_them() {
        for (x, py) in [(0.0, "0.0"), (-0.0, "-0.0"), (1.0, "1.0"), (100.0, "100.0"), (60.666666984558105, "60.666666984558105"), (0.001, "0.001"), (0.0001, "0.0001"), (1e-5, "1e-05"), (1.5e-7, "1.5e-07"), (1e15, "1000000000000000.0"), (1e16, "1e+16"), (1.5e16, "1.5e+16"), (0.1 + 0.2, "0.30000000000000004"), (-2.5, "-2.5")] {
            assert_eq!(float(x), py, "{x}");
        }
    }

    /// json.dumps(indent="\t"): one key per line, ": " and ",", empty containers kept short, ASCII
    /// escapes by default.
    #[test]
    fn values_are_rendered_as_json_dumps_with_a_tab() {
        let v = json!({"b": [1, 2.5, "é\"\n"], "a": {}, "c": [], "d": {"x": null, "y": true}});
        assert_eq!(render(&v, 1, true), "\t{\n\t\t\"b\": [\n\t\t\t1,\n\t\t\t2.5,\n\t\t\t\"\\u00e9\\\"\\n\"\n\t\t],\n\t\t\"a\": {},\n\t\t\"c\": [],\n\t\t\"d\": {\n\t\t\t\"x\": null,\n\t\t\t\"y\": true\n\t\t}\n\t}");
        assert_eq!(string("ゲーム\u{7f}🎮", true), "\"\\u30b2\\u30fc\\u30e0\\u007f\\ud83c\\udfae\"");
        assert_eq!(string("ゲーム\u{7f}\u{1}", false), "\"ゲーム\u{7f}\\u0001\"");
    }

    #[test]
    fn a_key_twice_is_found() {
        assert!(no_key_twice(r#"{"a": [{"b": 1, "c": 2}], "d": 3.5}"#).is_ok());
        assert!(no_key_twice(r#"{"a": [{"b": 1, "b": 2}]}"#).unwrap_err().contains("\"b\""));
        assert!(no_key_twice(r#"{"a": 1} x"#).is_err());
    }
}
