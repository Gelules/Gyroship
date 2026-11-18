// The launcher's page: the fields mirror the settings object, saved as it changes; the
// buttons call the launcher's commands. Nothing here reaches the network.
const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const $ = (id) => document.getElementById(id);
const fields = [...document.querySelectorAll('[data-key]')];
let settings = {};

function notice(text) {
  $('notice').textContent = text || '';
  $('notice').hidden = !text;
}

function show() {
  for (const f of fields) {
    if (f.type === 'checkbox') f.checked = !!settings[f.dataset.key];
    else f.value = settings[f.dataset.key] || '';
  }
}

// True when the settings reached their file; the player is told when they did not.
function save() {
  return invoke('set_settings', { settings }).then(() => true, (e) => (notice('The settings could not be saved: ' + e), false));
}

async function refreshBuilds() {
  // The chosen build is among those found while it exists; one that is gone (renamed by
  // a new port, deleted) gives way to the first one found.
  const found = await invoke('list_builds', { settings });
  if (found.length && !found.includes(settings.build)) {
    if (settings.build) notice(`The chosen build is gone (${settings.build}): ${found[0]} takes its place.`);
    settings.build = found[0];
    await save();
  }
  const all = found.length ? found : [settings.build].filter(Boolean);
  $('builds').replaceChildren(...all.map((b) => new Option(b, b, false, b === settings.build)));
}

async function refreshLayers() {
  const found = await invoke('list_layers', { settings });
  // A layer chosen elsewhere (in the settings file, by hand) stays offered.
  if (settings.layer && !found.some(([path]) => path === settings.layer)) found.push([settings.layer, settings.layer]);
  $('layers').replaceChildren(new Option("none: the game's own files", ''), ...found.map(([path, about]) => new Option(about, path, false, path === settings.layer)));
}

async function changed(field) {
  // A path pasted from the file explorer ("Copy as path") comes inside quotes: without them.
  const quoted = /^\s*"(.*)"\s*$/.exec(field.value);
  if (quoted) field.value = quoted[1];
  settings[field.dataset.key] = field.type === 'checkbox' ? field.checked : field.value.trim();
  if (await save()) notice('');
  if (['repo', 'data'].includes(field.dataset.key)) await Promise.all([refreshBuilds(), refreshLayers()]);
}

// The usual places under the data folder, for the fields still empty.
async function usual() {
  const { data, repo } = settings;
  if (!data || !repo) return notice('Set the repository and the data folder first.');
  const under = (path) => data.replace(/[\\/]+$/, '') + '\\' + path;
  const defaults = { save: under('save'), cache: under('pipeline-cache') };
  for (const [key, value] of Object.entries(defaults)) if (!settings[key]) settings[key] = value;
  show();
  if (await save()) await Promise.all([refreshBuilds(), refreshLayers()]);
}

async function lastSession() {
  const r = await invoke('last_session', { settings });
  // A session whose end was not recorded has no exit code and no duration to show.
  const end = r && r.ended ? [['Ended', `${r.outcome} (exit code ${r.code})`], ['Lasted', `${Math.floor(r.seconds / 60)} min ${r.seconds % 60} s`]] : [['Ended', r && r.outcome]];
  const rows = r ? [['Session', r.session], ...end, ['Summary', r.summary || 'none in the journal'], ['Journals', r.folder]] : [['Session', 'none yet']];
  $('last').replaceChildren(...rows.flatMap(([k, v]) => [Object.assign(document.createElement('dt'), { textContent: k }), Object.assign(document.createElement('dd'), { textContent: v })]));
  $('messages').textContent = r ? r.messages || 'No message kept (the game ran in a console, or said nothing).' : '';
}

function tab(name) {
  for (const s of document.querySelectorAll('section')) s.hidden = s.id !== name;
  for (const b of document.querySelectorAll('nav button')) b.classList.toggle('on', b.dataset.tab === name);
  if (name === 'logs') lastSession();
}

function busy(on) {
  for (const b of document.querySelectorAll('[data-step], #go')) b.disabled = on;
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
    refreshBuilds();
  });
  await listen('notice', (e) => notice(e.payload));
  const [loaded, problem] = await invoke('get_settings');
  settings = loaded;
  notice(problem || window.__NOTICE__);
  show();
  // A text field is saved as it is typed: closing the window right after loses nothing.
  for (const f of fields) f.addEventListener(f.matches('select, [type=checkbox]') ? 'change' : 'input', () => changed(f));
  for (const b of document.querySelectorAll('nav button')) b.addEventListener('click', () => tab(b.dataset.tab));
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
  for (const b of document.querySelectorAll('[data-step]')) {
    b.addEventListener('click', async () => {
      $('output').textContent = '';
      busy(true);
      try {
        await invoke('run_step', { settings, kind: b.dataset.step });
      } catch (e) {
        $('output').textContent = e;
        busy(false);
      }
    });
  }
  await Promise.all([refreshBuilds(), refreshLayers()]);
  // Back from a game (the launcher says so when it opens this window): how it ended comes first.
  if (window.__AFTER_GAME__) tab('logs');
}

main().catch((e) => notice('The launcher failed to start: ' + e));
