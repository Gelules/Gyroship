// The launcher's page: the fields mirror the settings object, saved as it changes; the
// buttons call the launcher's commands. The page itself reaches no network: the launcher
// looks for updates and fetches them (D109).
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const $ = (id) => document.getElementById(id);
const fields = [...document.querySelectorAll('[data-key]')];
let settings = {};

function notice(text) {
  $('notice').textContent = text || '';
  $('notice').hidden = !text;
}

// A box marked data-invert shows a setting that says the opposite (no_update_check: on by
// default, as settings are false until set).
function show() {
  for (const f of fields) {
    if (f.type === 'checkbox') f.checked = !!settings[f.dataset.key] !== ('invert' in f.dataset);
    else f.value = settings[f.dataset.key] || '';
  }
  follow();
}

// What hangs on other fields: the practice options without a practice file, the debug menu
// box under a hooks file of the player's own, the paths the "usual" button would set.
function follow() {
  for (const f of document.querySelectorAll('.needs-practice')) f.disabled = !settings.practice;
  $('hooks-note').hidden = !settings.hooks;
  const data = (settings.data || '').replace(/[\\/]+$/, '');
  $('usual-note').textContent = data
    ? `Sets the empty Save folder to ${data}\\save and the empty Shader cache folder to ${data}\\pipeline-cache.`
    : 'Sets the empty Save and Shader cache folders inside the data folder.';
}

// True when the settings reached their file; the player is told when they did not.
function save() {
  return invoke('set_settings', { settings }).then(() => true, (e) => (notice('The settings could not be saved: ' + e), false));
}

async function refreshBuilds(replace = true) {
  // The chosen build is among those found while it exists; one that is gone (renamed by
  // a new port, deleted) gives way to the first one found. Not while an update's rebuild
  // runs: the build it makes is the one to play.
  const found = await invoke('list_builds', { settings });
  if (replace && found.length && !found.includes(settings.build)) {
    if (settings.build) notice(`Build not found (${settings.build}): using ${found[0]}.`);
    settings.build = found[0];
    await save();
  }
  const all = found.length ? found : [settings.build].filter(Boolean);
  $('builds').replaceChildren(...(all.length ? all.map((b) => new Option(b, b, false, b === settings.build)) : [new Option('No build yet: finish the Setup tab', '')]));
}

async function refreshLayers() {
  const found = await invoke('list_layers', { settings });
  // A layer chosen elsewhere (in the settings file, by hand) stays offered.
  if (settings.layer && !found.some(([path]) => path === settings.layer)) found.push([settings.layer, settings.layer]);
  $('layers').replaceChildren(new Option('None', ''), ...found.map(([path, about]) => new Option(about, path, false, path === settings.layer)));
}

function lamp(id, state) {
  $(id).className = 'lamp' + (state ? ' ' + state : '');
}

// What building and playing need on this PC (Rust, the C++ tools, the runtime), and whether
// the tools are built: step 2's lamps and its Install buttons, shown only when winget can.
let needs = null;
let tools = false;
async function refreshNeeds() {
  needs = await invoke('list_needs');
  showNeeds();
}

function showNeeds() {
  if (!needs) return;
  const n = needs;
  const rust = n.rust_ok ? `Rust ${n.rust} is installed.` : n.rust ? `Rust ${n.rust} is too old: run rustup update, then come back.` : 'Rust is not installed (if rustup is, run rustup default stable).';
  const rows = [
    ['rust', n.rust_ok, rust, !n.rust],
    ['cpp', n.cpp, n.cpp ? "Microsoft's C++ build tools are installed." : "Microsoft's C++ build tools are not installed.", true],
    ['runtime', n.runtime, n.runtime ? 'The Visual C++ runtime is installed.' : 'The Visual C++ runtime is not installed.', true],
  ];
  for (const [id, ok, text, installable] of rows) {
    lamp(`lamp-${id}`, ok ? 'on' : 'bad');
    $(`${id}-state`).textContent = text;
    $(`install-${id}`).hidden = ok || !installable || !n.winget;
  }
  const all = n.rust_ok && n.cpp && n.runtime;
  $('winget-note').hidden = n.winget || all;
  lamp('lamp-needs', all && tools ? 'on' : all ? '' : 'bad');
}

// The Setup tab's state: flagged on its tab, listed on it, a lamp per step; and the launch
// strip's line.
async function refreshChecks() {
  const c = await invoke('check', { settings });
  const unfinished = c.setup.length > 0;
  $('tab-setup').classList.toggle('alert', unfinished);
  // An SVG element has no hidden property: the attribute itself goes.
  $('setup-alert').toggleAttribute('hidden', !unfinished);
  $('problems').hidden = !unfinished;
  $('problems-list').replaceChildren(...c.setup.map((p) => Object.assign(document.createElement('li'), { textContent: p })));
  lamp('lamp-folders', c.folders ? 'on' : 'bad');
  lamp('lamp-recompile', c.recompiled ? 'on' : c.folders && !c.tools ? 'bad' : '');
  lamp('lamp-build', c.built ? 'on' : '');
  lamp('lamp-tools', c.tools ? 'on' : c.folders ? 'bad' : '');
  $('tools-state').textContent = c.tools ? "Gyroship's tools are built." : c.folders ? "Gyroship's tools are not built yet." : "Gyroship's tools: set the folders first (step 1).";
  // Offered even once built: a Git clone's tools are built again after each update.
  $('build-tools').hidden = !c.folders;
  tools = c.tools;
  showNeeds();
  $('exefs-note').textContent = c.exefs || '';
  $('exefs-note').hidden = !c.exefs;
  // The dump recognised: the update the practice files and the debug menu box are made for.
  // Nothing said while the templates' build cannot be read (no repository set yet).
  const known = c.dump && c.dump_known != null;
  $('dump-note').textContent = !known ? '' : c.dump_known
    ? 'Recognised: the update the practice files and the debug menu are made for.'
    : `Not the update the practice files are made for (build ${c.dump.slice(0, 8)}…): leave the practice file and the debug menu off, as on this build they would write to the wrong places in the game's memory.`;
  $('dump-note').className = 'note' + (known && !c.dump_known ? ' bad' : ' good');
  $('dump-note').hidden = !known;
  const ready = !unfinished && !c.exefs;
  $('ready-text').textContent = unfinished ? `Setup is not finished (${c.setup[0]}).` : c.exefs ? `${c.exefs}.` : 'Ready.';
  lamp('ready-lamp', ready ? 'on' : 'bad');
  $('to-setup').hidden = !unfinished;
}

async function changed(field) {
  // A path pasted from the file explorer ("Copy as path") comes inside quotes: without them.
  const quoted = /^\s*"(.*)"\s*$/.exec(field.value);
  if (quoted) field.value = quoted[1];
  settings[field.dataset.key] = field.type === 'checkbox' ? field.checked !== ('invert' in field.dataset) : field.value.trim();
  follow();
  if (await save()) notice('');
  const key = field.dataset.key;
  if (['repo', 'data'].includes(key)) await Promise.all([refreshBuilds(), refreshLayers()]);
  if (['repo', 'data', 'exefs', 'build', 'probes'].includes(key)) await refreshChecks();
}

// The usual places under the data folder, for the fields still empty.
async function usual() {
  const { data, repo } = settings;
  if (!data || !repo) return notice('Set the repository and the data folder first.');
  const under = (path) => data.replace(/[\\/]+$/, '') + '\\' + path;
  const defaults = { save: under('save'), cache: under('pipeline-cache') };
  for (const [key, value] of Object.entries(defaults)) if (!settings[key]) settings[key] = value;
  show();
  if (await save()) await Promise.all([refreshBuilds(), refreshLayers(), refreshChecks()]);
}

async function lastSession() {
  const r = await invoke('last_session', { settings });
  // A session whose end was not recorded has no exit code and no duration to show.
  const end = r && r.ended ? [['Ended', `${r.outcome} (exit code ${r.code})`], ['Lasted', `${Math.floor(r.seconds / 60)} min ${r.seconds % 60} s`]] : [['Ended', r && r.outcome]];
  const rows = r ? [['Session', r.session], ...end, ['Summary', r.summary || 'none'], ['Logs', r.folder]] : [['Session', 'none yet']];
  $('last').replaceChildren(...rows.flatMap(([k, v]) => [Object.assign(document.createElement('dt'), { textContent: k }), Object.assign(document.createElement('dd'), { textContent: v })]));
  $('messages').textContent = r ? r.messages || 'No message kept (the game ran in a console, or said nothing).' : '';
}

function tab(name) {
  for (const s of document.querySelectorAll('main > section')) s.hidden = s.id !== name;
  for (const b of document.querySelectorAll('nav button')) b.classList.toggle('on', b.dataset.tab === name);
  if (name === 'logs') lastSession();
}

// A step or an update runs: the launcher's own Busy, as this page started it.
let running = false;
function busy(on) {
  running = on;
  for (const b of document.querySelectorAll('[data-step], #go, #check-update, #update-retry')) b.disabled = on;
}

// A setup step, its messages below the steps; false when it did not start (said there too).
async function runStep(kind) {
  $('output').textContent = '';
  busy(true);
  try {
    await invoke('run_step', { settings, kind });
    return true;
  } catch (e) {
    $('output').textContent = e;
    busy(false);
    return false;
  }
}

// Updates (D109). The banner over every tab says how one goes; its buttons are those named.
let release = null;
const from = window.__UPDATED_FROM__;
function banner(text, buttons = ['update-close']) {
  $('update-text').textContent = text;
  for (const id of ['update-retry', 'update-delete', 'update-page', 'update-close']) $(id).hidden = !buttons.includes(id);
  $('update').hidden = !text;
}

// GitHub asked for a newer release. A release's copy installs it (`install`); a Git clone, or a
// launcher outside a Gyroship folder, is only told.
async function checkUpdate(install) {
  $('update-state').textContent = 'Looking for a new version…';
  $('check-update').disabled = true;
  try {
    const u = await invoke('check_update');
    release = u.release;
    if (!release) return void ($('update-state').textContent = `Gyroship ${u.version} is the latest version.`);
    $('update-state').textContent = `Gyroship ${release.tag} is out (this copy is ${u.version}).`;
    if (u.blocker) banner(`Gyroship ${release.tag} is out: ${u.blocker}.`, ['update-page', 'update-close']);
    else if (install) await applyUpdate();
  } catch (e) {
    // Offline, or no release published yet: nothing to fix, nothing in the way.
    $('update-state').textContent = `Could not look for a new version: ${e}`;
  } finally {
    $('check-update').disabled = running;
  }
}

// The launcher downloads the release and installs it next to this copy, then starts the new
// copy's launcher and ends: this page goes with it. A failure leaves this copy as it was. Not
// while a step runs: the launcher would refuse, and the page would wrongly free its buttons.
async function applyUpdate() {
  if (running) return banner(`Gyroship ${release.tag} is out: once this step ends, press Check now in Setup to install it.`, ['update-page', 'update-close']);
  banner(`Updating to Gyroship ${release.tag}: the launcher restarts in the new version once it is installed.`, ['update-page']);
  busy(true);
  try {
    await invoke('apply_update', { settings });
  } catch (e) {
    busy(false);
    banner(`The update to Gyroship ${release.tag} did not start: ${e}`, ['update-retry', 'update-page', 'update-close']);
  }
}

// After an update, the game is built again for the new version: Recompile, then Build. The
// steps left, while they run. The previous copy can go once it is built, from this window only.
let rebuild = null;
const leftBehind = () => (from ? ` The previous copy is still in ${from}: delete it here, or later in the file explorer.` : '');
async function rebuildNext(code) {
  const kind = code === 0 && rebuild.shift();
  if (kind) {
    if (!(await runStep(kind))) rebuildNext(-1);
    return;
  }
  const built = code === 0;
  rebuild = null;
  if (built) {
    // The game it just built is the one to play.
    const found = await invoke('list_builds', { settings });
    if (found.length) settings.build = found[0];
    await save();
  }
  await refreshBuilds();
  await refreshChecks();
  // Until this copy's game is built, the previous copy keeps the only one: no offer to delete it.
  if (built) banner(`Gyroship ${window.__VERSION__} is ready to play. Shortcuts to the previous copy's launcher still start it: point them to this one.${leftBehind()}`, ['update-delete', 'update-close']);
  else banner(`Gyroship ${window.__VERSION__} is installed, but its game is not built yet: see the messages in Setup, then Recompile and Build. Until then, the previous copy's launcher still plays the previous version (${from}).`);
}

async function deletePrevious() {
  $('update-delete').disabled = true;
  try {
    await invoke('delete_previous');
    banner(`The previous copy is deleted (${from}).`);
  } catch (e) {
    $('update-delete').disabled = false;
    banner(`The previous copy is still there: ${e}`, ['update-delete', 'update-close']);
  }
}

async function main() {
  // The listeners first: nothing the launcher says is missed.
  await listen('line', (e) => {
    const out = $('output');
    out.textContent += e.payload + '\n';
    out.scrollTop = out.scrollHeight;
  });
  await listen('done', (e) => {
    $('output').textContent += `\nended with exit code ${e.payload}\n`;
    busy(false);
    // A close refused while the step ran was said in the notice: no longer true.
    notice('');
    refreshNeeds();
    if (rebuild) return rebuildNext(e.payload);
    refreshBuilds().then(refreshChecks);
  });
  await listen('notice', (e) => notice(e.payload));
  await listen('update', (e) => banner(e.payload, ['update-page']));
  await listen('update-failed', (e) => {
    busy(false);
    banner(`The update failed: ${e.payload}. This copy works as before.`, ['update-retry', 'update-page', 'update-close']);
  });
  const [loaded, problem] = await invoke('get_settings');
  settings = loaded;
  notice(problem || window.__NOTICE__);
  // A language code the list does not name (en-US, pt, ru..., or one written by hand) stays offered.
  const language = document.querySelector('[data-key=language]');
  if (settings.language && ![...language.options].some((o) => o.value === settings.language)) language.add(new Option(settings.language, settings.language));
  show();
  // A text field is saved as it is typed: closing the window right after loses nothing.
  for (const f of fields) f.addEventListener(f.matches('select, [type=checkbox]') ? 'change' : 'input', () => changed(f));
  for (const b of document.querySelectorAll('nav button')) b.addEventListener('click', () => tab(b.dataset.tab));
  $('to-setup').addEventListener('click', () => tab('setup'));
  $('usual').addEventListener('click', usual);
  $('folder').addEventListener('click', () => invoke('open_logs', { settings }).catch(notice));
  $('go').addEventListener('click', async () => {
    $('state').textContent = '';
    try {
      await invoke('play', { settings });
    } catch (e) {
      $('state').textContent = e;
    }
  });
  for (const b of document.querySelectorAll('[data-step]')) b.addEventListener('click', () => runStep(b.dataset.step));
  $('check-update').addEventListener('click', () => checkUpdate(true));
  $('update-retry').addEventListener('click', applyUpdate);
  $('update-page').addEventListener('click', () => release && invoke('open_release', { url: release.page }).catch(notice));
  $('update-delete').addEventListener('click', deletePrevious);
  $('update-close').addEventListener('click', () => banner(''));
  await Promise.all([refreshBuilds(!from), refreshLayers()]);
  await refreshChecks();
  // Running cargo and vswhere takes a moment: the page is already usable meanwhile.
  refreshNeeds();
  // Back in the window after fixing something elsewhere (a folder made, Rust installed, the game
  // built in a terminal): the checks see it.
  window.addEventListener('focus', () => {
    refreshBuilds(!rebuild).then(refreshChecks).catch(notice);
    if (!needs || !(needs.rust_ok && needs.cpp && needs.runtime)) refreshNeeds().catch(notice);
  });
  // Back from a game (the launcher says so when it opens this window): how it ended comes first.
  if (window.__AFTER_GAME__) tab('logs');
  if (from) {
    // Just updated: the game is built again for this version, in front of the player.
    banner(`Updated to Gyroship ${window.__VERSION__}. Building the game for it: Recompile, then Build.`, []);
    tab('setup');
    rebuild = ['recompile', 'build'];
    rebuildNext(0);
  } else if (window.__FIRST_WINDOW__ && !settings.no_update_check) {
    // Once per launcher start, even when its first window comes after a game (a shortcut's --play).
    checkUpdate(true);
  } else {
    $('update-state').textContent = `This is Gyroship ${window.__VERSION__}.`;
  }
}

main().catch((e) => notice('The launcher failed to start: ' + e));
