const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const LIMIT0 = 5, STEP = 5, MAXLIM = 15, MAXGEN = 5, MAXKW = 5;
const movieMap = new Map();

let catalogPromise = null;

async function api(path) {
  const res = await fetch('/api' + path, { headers: { Accept: 'application/json' } });
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(data?.error || `Request failed (${res.status})`);
  return data;
}

function getCatalog() {
  if (!catalogPromise) catalogPromise = api('/movies/catalog').catch((e) => { catalogPromise = null; throw e; });
  return catalogPromise;
}

const getGenres = () => api('/genres');
const getMovie = (id) => api(`/movies/${id}`);
const getRecs = (id, n) => api(`/recommendations/${id}?n=${n}`);
const suggestKw = (q) => api(`/keywords/suggest?q=${encodeURIComponent(q)}`);
const discover = ({ genres = [], keywords = [], n = 5 }) =>
  api(`/recommendations/discover?genres=${encodeURIComponent(genres.join(','))}&keywords=${encodeURIComponent(keywords.join(','))}&n=${n}`);

async function searchTitles(q = '') {
  const movies = await getCatalog();
  const s = q.trim().toLowerCase();
  return s ? movies.filter((m) => m.title.toLowerCase().includes(s)) : movies;
}

async function starterMovies(limit = 5) {
  const movies = await getCatalog();
  const n = Math.min(limit, movies.length);
  const arr = [...movies];
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(Math.random() * (arr.length - i));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return Promise.all(arr.slice(0, n).map((m) => getMovie(m.movie_id)));
}

const state = {
  seed: null, recs: [], searchLimit: LIMIT0, starter: [],
  genres: [], selGenres: [], selKeywords: [],
  discoverResults: [], discoverLimit: LIMIT0, hasRunDiscover: false,
};

function attachPosterFallback(root) {
  root.querySelectorAll('img.poster-img').forEach((img) => {
    img.onerror = () => { img.outerHTML = `<div class="poster-fallback">${esc(img.alt)}</div>`; };
  });
}

function cardHTML(m) {
  movieMap.set(m.movie_id, m);
  const poster = m.poster_url
    ? `<img class="poster-img" src="${esc(m.poster_url)}" alt="${esc(m.title)}" loading="lazy" referrerpolicy="no-referrer">`
    : `<div class="poster-fallback">${esc(m.title)}</div>`;
  return `<article class="card">
    <div class="poster" data-id="${m.movie_id}">${poster}</div>
    <button class="card-title" data-id="${m.movie_id}">${esc(m.title)}</button>
    <p class="card-meta">${esc((m.genres || []).slice(0, 3).join(' · '))}</p>
  </article>`;
}

function renderGrid(el, movies) {
  el.innerHTML = movies.map(cardHTML).join('');
  attachPosterFallback(el);
}

const fmtCount = (n) => `${n} ${n === 1 ? 'film' : 'films'}`;

function renderMoreBar(bar, status, btn, count, limit) {
  btn.classList.remove('is-loading');
  if (count === 0) { bar.hidden = true; return; }
  bar.hidden = false;
  if (limit >= MAXLIM) { status.textContent = `Showing ${count} of max ${MAXLIM}`; btn.disabled = true; btn.textContent = 'Limit Reached'; }
  else if (count < limit) { status.textContent = `Showing all ${count} available`; btn.disabled = true; btn.textContent = 'No More Results'; }
  else { status.textContent = `Showing ${count} of up to ${MAXLIM}`; btn.disabled = false; btn.textContent = 'Suggest More'; }
}

function renderSearch() {
  $('#titleSpinner').hidden = true;
  $('#clearTitle').hidden = !$('#titleInput').value;

  if (state.seed) {
    $('#seedBanner').hidden = false;
    $('#seedPoster').src = state.seed.poster_url || '';
    $('#seedPoster').alt = state.seed.title;
    $('#seedTitle').textContent = state.seed.title;
    $('#seedGenres').textContent = state.seed.genres.join(' · ') || 'Film';
    $('#searchHeading').textContent = `Similar to ${state.seed.title}`;
    $('#searchCount').textContent = fmtCount(state.recs.length);
    $('#searchEmpty').hidden = state.recs.length > 0;
    renderGrid($('#searchGrid'), state.recs);
    renderMoreBar($('#searchMoreBar'), $('#searchMoreStatus'), $('#searchMoreBtn'), state.recs.length, state.searchLimit);
  } else {
    $('#seedBanner').hidden = true;
    $('#searchMoreBar').hidden = true;
    $('#searchHeading').textContent = 'Featured Picks';
    $('#searchCount').textContent = fmtCount(state.starter.length);
    $('#searchEmpty').hidden = state.starter.length > 0;
    renderGrid($('#searchGrid'), state.starter);
  }
}

async function selectMovie(id, limit = LIMIT0, append = false) {
  state.searchLimit = Math.min(limit, MAXLIM);
  $('#searchError').hidden = true;
  if (!append) { $('#titleSpinner').hidden = false; $('#clearTitle').hidden = true; }
  try {
    const [seed, recs] = await Promise.all([getMovie(id), getRecs(id, state.searchLimit)]);
    state.seed = seed;
    if (append) {
      const have = new Set(state.recs.map((m) => m.movie_id));
      state.recs = [...state.recs, ...recs.filter((m) => !have.has(m.movie_id))];
    } else {
      state.recs = recs;
    }
    renderSearch();
  } catch (e) {
    $('#titleSpinner').hidden = true;
    $('#searchMoreBtn').classList.remove('is-loading');
    $('#searchError').hidden = false;
    $('#searchError').textContent = e.message;
  }
}

let titleReq = 0;
function hideTitleDD() { $('#titleDropdown').hidden = true; }

function renderTitleDD(items) {
  const dd = $('#titleDropdown');
  dd.innerHTML = items.length
    ? items.map((m) => `<li><button class="dd-item" data-id="${m.movie_id}">${esc(m.title)}</button></li>`).join('')
    : `<li class="dd-empty">No matching movies found</li>`;
  dd.hidden = false;
}

async function openTitleDD(q) {
  const reqId = ++titleReq;
  const items = await searchTitles(q).catch(() => []);
  if (reqId !== titleReq) return;
  renderTitleDD(items);
}

function renderGenres() {
  $('#genreBox').innerHTML = state.genres.map((g) => {
    const sel = state.selGenres.includes(g);
    const disabled = !sel && state.selGenres.length >= MAXGEN;
    return `<button class="chip ${sel ? 'sel' : ''}" ${disabled ? 'disabled' : ''} data-genre="${esc(g)}">${esc(g)}</button>`;
  }).join('');
}

function renderSelKeywords() {
  $('#selectedKeywords').innerHTML = state.selKeywords
    .map((k) => `<button class="chip sel" data-kw="${esc(k)}">${esc(k)} &times;</button>`).join('');
}

function updateHint() {
  const g = state.selGenres.length >= MAXGEN, k = state.selKeywords.length >= MAXKW;
  $('#discoverHint').textContent = g && k ? `Maximum ${MAXGEN} genres and ${MAXKW} keywords selected.`
    : g ? `Maximum ${MAXGEN} genres selected.`
    : k ? `Maximum ${MAXKW} keywords selected.` : '';
}

function renderDiscover() {
  renderGenres();
  renderSelKeywords();
  updateHint();

  const hasFilters = state.selGenres.length || state.selKeywords.length;
  $('#discoverClearBtn').hidden = !hasFilters;
  $('#keywordInput').disabled = state.selKeywords.length >= MAXKW;

  if (!state.hasRunDiscover) {
    $('#discoverCount').textContent = '';
    $('#discoverGrid').innerHTML = '';
    $('#discoverEmpty').hidden = false;
    $('#discoverEmpty').textContent = 'Select genres or keywords and click Find Films.';
    $('#discoverMoreBar').hidden = true;
    return;
  }

  $('#discoverCount').textContent = fmtCount(state.discoverResults.length);

  if (state.discoverResults.length === 0) {
    $('#discoverGrid').innerHTML = '';
    $('#discoverEmpty').hidden = false;
    $('#discoverEmpty').textContent = 'No matching films found.';
    $('#discoverMoreBar').hidden = true;
    return;
  }

  $('#discoverEmpty').hidden = true;
  renderGrid($('#discoverGrid'), state.discoverResults);
  renderMoreBar($('#discoverMoreBar'), $('#discoverMoreStatus'), $('#discoverMoreBtn'), state.discoverResults.length, state.discoverLimit);
}

async function runDiscover(limit = LIMIT0) {
  if (!state.selGenres.length && !state.selKeywords.length) {
    $('#discoverBtn').classList.remove('is-loading');
    $('#discoverHint').textContent = 'Select at least one genre or add a keyword.';
    return;
  }
  state.discoverLimit = Math.min(limit, MAXLIM);
  state.hasRunDiscover = true;
  $('#discoverError').hidden = true;
  try {
    state.discoverResults = await discover({ genres: state.selGenres, keywords: state.selKeywords, n: state.discoverLimit });
    renderDiscover();
  } catch (e) {
    $('#discoverError').hidden = false;
    $('#discoverError').textContent = e.message;
  } finally {
    $('#discoverBtn').classList.remove('is-loading');
    $('#discoverMoreBtn').classList.remove('is-loading');
  }
}

function addKeyword(kw) {
  const t = kw.trim();
  if (!t) return;
  if (state.selKeywords.some((k) => k.toLowerCase() === t.toLowerCase())) return;
  if (state.selKeywords.length >= MAXKW) { $('#discoverHint').textContent = `You can select up to ${MAXKW} keywords.`; return; }
  state.selKeywords.push(t);
  $('#keywordInput').value = '';
  $('#keywordDropdown').hidden = true;
  renderDiscover();
}

async function openModal(id) {
  let m = movieMap.get(id);
  if (!m || !m.overview) m = await getMovie(id).catch(() => m);
  if (!m) return;
  $('#modalPoster').src = m.poster_url || '';
  $('#modalPoster').alt = m.title;
  $('#modalTitle').textContent = m.title;
  $('#modalOverview').textContent = m.overview || '';
  $('#modalDirector').textContent = m.director ? `Director: ${m.director}` : '';
  $('#modalCast').textContent = m.cast?.length ? `Cast: ${m.cast.join(', ')}` : '';
  $('#modalGenres').textContent = `Genres: ${m.genres?.join(' · ') || 'Unavailable'}`;
  $('#modalKeywords').textContent = `Keywords: ${m.keywords?.join(' · ') || 'Unavailable'}`;
  $('#modal').showModal();
}

function setView(v) {
  $('#searchView').hidden = v !== 'search';
  $('#discoverView').hidden = v !== 'discover';
  document.querySelectorAll('.nav').forEach((b) => b.classList.toggle('active', b.dataset.view === v));
}

document.querySelectorAll('.nav').forEach((b) => (b.onclick = () => setView(b.dataset.view)));
$('#brand').onclick = (e) => { e.preventDefault(); setView('search'); };

$('#randomBtn').onclick = async () => {
  try {
    const movies = await getCatalog();
    const pool = movies.filter((m) => m.movie_id !== state.seed?.movie_id);
    const pick = pool[Math.floor(Math.random() * pool.length)];
    $('#titleInput').value = pick.title;
    setView('search');
    selectMovie(pick.movie_id);
  } catch (e) {
    $('#searchError').hidden = false;
    $('#searchError').textContent = e.message;
  }
};

let titleDeb;
$('#titleInput').addEventListener('input', () => {
  $('#clearTitle').hidden = !$('#titleInput').value;
  clearTimeout(titleDeb);
  titleDeb = setTimeout(() => openTitleDD($('#titleInput').value), 150);
});
$('#titleInput').addEventListener('focus', () => openTitleDD($('#titleInput').value));

document.addEventListener('click', (e) => {
  if (!e.target.closest('.combobox')) { hideTitleDD(); $('#keywordDropdown').hidden = true; }
});

$('#titleDropdown').addEventListener('click', (e) => {
  const btn = e.target.closest('.dd-item');
  if (!btn) return;
  $('#titleInput').value = btn.textContent;
  hideTitleDD();
  selectMovie(Number(btn.dataset.id));
});

$('#clearTitle').onclick = () => { $('#titleInput').value = ''; $('#clearTitle').hidden = true; $('#titleInput').focus(); };

$('#clearSeed').onclick = () => {
  state.seed = null; state.recs = []; state.searchLimit = LIMIT0;
  $('#titleInput').value = '';
  renderSearch();
};

$('#searchMoreBtn').onclick = () => {
  if (!state.seed) return;
  $('#searchMoreBtn').classList.add('is-loading');
  selectMovie(state.seed.movie_id, Math.min(state.searchLimit + STEP, MAXLIM), true);
};

$('#genreBox').addEventListener('click', (e) => {
  const btn = e.target.closest('.chip');
  if (!btn || btn.disabled) return;
  const g = btn.dataset.genre;
  state.selGenres = state.selGenres.includes(g)
    ? state.selGenres.filter((x) => x !== g)
    : state.selGenres.length < MAXGEN ? [...state.selGenres, g] : state.selGenres;
  renderDiscover();
});

$('#selectedKeywords').addEventListener('click', (e) => {
  const btn = e.target.closest('.chip');
  if (!btn) return;
  state.selKeywords = state.selKeywords.filter((k) => k !== btn.dataset.kw);
  renderDiscover();
});

let kwDeb, kwReq = 0;
$('#keywordInput').addEventListener('input', () => {
  clearTimeout(kwDeb);
  const v = $('#keywordInput').value;
  if (!v.trim()) { kwReq++; $('#keywordDropdown').hidden = true; return; }
  kwDeb = setTimeout(async () => {
    const reqId = ++kwReq;
    const sug = (await suggestKw(v).catch(() => []))
      .filter((k) => !state.selKeywords.some((s) => s.toLowerCase() === k.toLowerCase()));
    if (reqId !== kwReq) return;
    const dd = $('#keywordDropdown');
    dd.innerHTML = sug.length
      ? sug.map((k) => `<li><button class="dd-item" data-kw="${esc(k)}">${esc(k)}</button></li>`).join('')
      : '';
    dd.hidden = sug.length === 0;
  }, 150);
});

$('#keywordDropdown').addEventListener('click', (e) => {
  const btn = e.target.closest('.dd-item');
  if (btn) addKeyword(btn.dataset.kw);
});

$('#keywordInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); if ($('#keywordInput').value.trim()) addKeyword($('#keywordInput').value); }
});

$('#discoverBtn').onclick = () => {
  if ($('#keywordInput').value.trim()) addKeyword($('#keywordInput').value);
  $('#discoverBtn').classList.add('is-loading');
  runDiscover(LIMIT0);
};

$('#discoverClearBtn').onclick = () => {
  state.selGenres = []; state.selKeywords = []; state.discoverResults = [];
  state.hasRunDiscover = false; state.discoverLimit = LIMIT0;
  $('#keywordInput').value = ''; $('#discoverHint').textContent = '';
  renderDiscover();
};

$('#discoverMoreBtn').onclick = () => {
  $('#discoverMoreBtn').classList.add('is-loading');
  runDiscover(Math.min(state.discoverLimit + STEP, MAXLIM));
};

document.querySelectorAll('.grid').forEach((g) => {
  g.addEventListener('click', (e) => {
    const el = e.target.closest('[data-id]');
    if (el) openModal(Number(el.dataset.id));
  });
});

$('#modalClose').onclick = () => $('#modal').close();
$('#modal').addEventListener('click', (e) => { if (e.target === $('#modal')) $('#modal').close(); });

(async function init() {
  try {
    const [genres, starters] = await Promise.all([getGenres(), starterMovies(LIMIT0)]);
    state.genres = genres;
    state.starter = starters;
    renderSearch();
    renderDiscover();
  } catch (e) {
    $('#searchError').hidden = false;
    $('#searchError').textContent = e.message || 'Cannot reach the Filmantic server.';
  }
})();
