//! The texts Gyroship adds in game, in the game's text regions (D73, D74).
//!
//! The game reads the console's language code once at boot and picks one of its 14 text
//! regions; Gyroship's own texts follow the same region. Everything outside the game (the
//! command line, the logs) stays English.

/// The console's language codes (the Settings services' list), in its order. `--language`
/// takes one of them, in any case.
pub const LANGUAGE_CODES: [&str; 20] = [
    "ja", "en-US", "fr", "de", "it", "es", "zh-CN", "ko", "nl", "pt", "ru", "zh-TW", "en-GB", "fr-CA", "es-419", "zh-Hans", "zh-Hant", "pt-BR", "pl", "th",
];

/// The console code a `--language` value names, in its canonical case.
pub fn language_code(value: &str) -> Option<&'static str> {
    LANGUAGE_CODES.into_iter().find(|c| c.eq_ignore_ascii_case(value))
}

/// The game's text regions, in its order: 0 en-US, 1 en-GB, 2 de-DE, 3 es-ES, 4 es-419,
/// 5 it-IT, 6 pt-BR, 7 nl-NL, 8 fr-FR, 9 fr-CA, 10 ja-JP, 11 ko-KR, 12 zh-TW, 13 zh-CN.
pub const REGIONS: usize = 14;
const LANGUAGE: [&str; REGIONS] = ["en", "en", "de", "es", "es", "it", "pt", "nl", "fr", "fr", "ja", "ko", "zh", "zh"];
const COUNTRY: [&str; REGIONS] = ["us", "gb", "de", "es", "419", "it", "br", "nl", "fr", "ca", "jp", "kr", "tw", "cn"];

/// The text region the game picks for a console language code: the first region whose
/// language starts the code and whose country ends it, else that language's first region,
/// else US English. `zh-Hans` and `zh-Hant` stand for `zh-cn` and `zh-tw`.
pub fn region(code: &str) -> usize {
    let code = match code {
        "zh-Hans" => "zh-cn".to_string(),
        "zh-Hant" => "zh-tw".to_string(),
        c => c.to_ascii_lowercase(),
    };
    let mut first = None;
    for r in 0..REGIONS {
        if code.starts_with(LANGUAGE[r]) {
            if code.ends_with(COUNTRY[r]) {
                return r;
            }
            first.get_or_insert(r);
        }
    }
    first.unwrap_or(0)
}

/// A region's tag, as practice.txt's `font` lines write it: "en-US", "es-419", "ja-JP"...
pub fn tag(region: usize) -> String {
    let r = region.min(REGIONS - 1);
    format!("{}-{}", LANGUAGE[r], COUNTRY[r].to_ascii_uppercase())
}

/// A text Gyroship draws, or adds to the game's own menus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Text {
    QuitGame,
    Speed,
}

/// Every text, in the table's order.
pub const ALL: [Text; 2] = [Text::QuitGame, Text::Speed];
const _: () = assert!(ALL.len() == TEXTS.len());

/// One row per `Text`, one column per region. Written upper case, like the game's HUD and
/// buttons, with the game's own words where it has them. An empty variant (en-GB, es-419,
/// fr-CA) shows its parent's text.
const TEXTS: [[&str; REGIONS]; 2] = [
    ["QUIT GAME", "", "SPIEL BEENDEN", "SALIR DEL JUEGO", "", "ESCI DAL GIOCO", "SAIR DO JOGO", "SPEL AFSLUITEN", "QUITTER LE JEU", "", "ゲームを終了", "게임 종료", "退出遊戲", "退出游戏"],
    ["SPEED", "", "TEMPO", "VELOCIDAD", "", "VELOCITÀ", "VELOCIDADE", "SNELHEID", "VITESSE", "", "スピード", "스피드", "速度", "速度"],
];

/// Each region's parent: a variant's base language, else itself.
const PARENT: [usize; REGIONS] = [0, 0, 2, 3, 3, 5, 6, 7, 8, 8, 10, 11, 12, 13];

/// The text in a region: its own, else its parent's, else US English.
pub fn text(t: Text, region: usize) -> &'static str {
    let row = TEXTS[t as usize];
    let region = region.min(REGIONS - 1);
    [row[region], row[PARENT[region]], row[0]].into_iter().find(|s| !s.is_empty()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_console_code_lands_on_the_region_the_game_picks() {
        let want = [
            ("ja", 10),
            ("en-US", 0),
            ("fr", 8),
            ("de", 2),
            ("it", 5),
            ("es", 3),
            ("zh-CN", 13),
            ("ko", 11),
            ("nl", 7),
            ("pt", 6),
            ("ru", 0),
            ("zh-TW", 12),
            ("en-GB", 1),
            ("fr-CA", 9),
            ("es-419", 4),
            ("zh-Hans", 13),
            ("zh-Hant", 12),
            ("pt-BR", 6),
            ("pl", 0),
            ("th", 0),
        ];
        assert_eq!(want.map(|(c, _)| c), LANGUAGE_CODES);
        for (code, wanted) in want {
            assert_eq!(region(code), wanted, "{code}");
        }
    }

    #[test]
    fn a_language_value_is_a_console_code_in_any_case() {
        assert_eq!(language_code("FR"), Some("fr"));
        assert_eq!(language_code("fr-ca"), Some("fr-CA"));
        assert_eq!(language_code("ZH-HANS"), Some("zh-Hans"));
        assert_eq!(language_code("xx"), None);
        assert_eq!(language_code("fr-"), None);
    }

    #[test]
    fn a_region_tag_is_its_language_and_country() {
        assert_eq!([0, 4, 10, 13].map(tag), ["en-US", "es-419", "ja-JP", "zh-CN"]);
    }

    #[test]
    fn every_text_exists_in_every_base_language() {
        for t in ALL {
            for r in 0..REGIONS {
                let s = TEXTS[t as usize][r];
                // The variants alone may be empty, and fall back to their parent.
                assert!(!s.is_empty() || PARENT[r] != r, "{t:?} in region {r}");
                // Precomposed letters only: the game's fonts are drawn glyph by glyph.
                assert!(!s.chars().any(|c| ('\u{300}'..='\u{36f}').contains(&c)), "{t:?} in region {r}: {s}");
                assert!(!text(t, r).is_empty());
            }
        }
        assert_eq!(text(Text::QuitGame, 9), "QUITTER LE JEU");
        assert_eq!(text(Text::Speed, 4), "VELOCIDAD");
        assert_eq!(text(Text::Speed, 1), "SPEED");
    }
}
