// Shared utilities for Hog Charts

const logoUrl   = (id) => `https://a.espncdn.com/i/teamlogos/ncaa/500/${id}.png`;
const playerUrl = (id) => `https://a.espncdn.com/i/headshots/mens-college-basketball/players/full/${id}.png`;

// fetch() wrapper — returns parsed JSON or null on error
async function loadJSON(path) {
  try {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  } catch (e) {
    console.error("loadJSON failed:", path, e);
    return null;
  }
}

// Lazy-load Plotly (4.6MB) on first use instead of eagerly in every <head>.
// Only the shot-density chart needs it, and only after a card/modal opens.
// Returns a promise resolving to window.Plotly; caches so it loads once.
let _plotlyPromise = null;
function ensurePlotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  if (_plotlyPromise) return _plotlyPromise;
  _plotlyPromise = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "https://cdn.plot.ly/plotly-2.35.2.min.js";
    s.charset = "utf-8";
    s.onload = () => resolve(window.Plotly);
    s.onerror = () => { _plotlyPromise = null; reject(new Error("Plotly failed to load")); };
    document.head.appendChild(s);
  });
  return _plotlyPromise;
}

// Show an error banner inside any element
function showError(el, msg) {
  if (typeof el === "string") el = document.getElementById(el);
  if (!el) return;
  el.innerHTML = `<div style="color:var(--t1);padding:24px;font-weight:700">${msg}</div>`;
}

// Matches numpy.interp: clamp outside range, linear interpolation inside.
function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 0; i < xs.length - 1; i++) {
    if (x >= xs[i] && x <= xs[i + 1]) {
      const t = (x - xs[i]) / (xs[i + 1] - xs[i]);
      return ys[i] + t * (ys[i + 1] - ys[i]);
    }
  }
  return ys[ys.length - 1];
}

// Sign-aware formatting: "+3.2" / "-1.0"
function fmt(v, decimals = 1) {
  const n = Number(v);
  const s = Math.abs(n).toFixed(decimals);
  return n >= 0 ? `+${s}` : `-${s}`;
}

function fmtPlain(v, decimals = 1) {
  return Number(v).toFixed(decimals);
}

function fmtPct(v, decimals = 1) {
  return v != null ? (Number(v) * 100).toFixed(decimals) : "—";
}

// ── Site navigation (single source of truth) ────────────────────────────────
// The link list lives here so adding/renaming a page is a one-line edit instead
// of touching every HTML file. Each page's <nav> ships with only the brand
// anchor; populateNav() injects these links after it at load. data-page matches
// markActiveNav()'s filename check (space-separated for a hub + its sub-pages).
const NAV_LINKS = [
  { href: "index.html",        label: "Home" },
  { href: "scout-report.html", label: "Scout" },
  { href: "prediction.html",   label: "Predictor" },
  { href: "net-ratings.html",  label: "Team Stats" },
  { href: "player-stats.html", label: "Player Stats" },
  { href: "lineup-stats.html", label: "Lineups" },
  { href: "shot-charts.html",  label: "Shot Charts" },
  { href: "postseason.html",   label: "Postseason" },
  { href: "glossary.html",     label: "Glossary" },
];

function populateNav() {
  const nav = document.querySelector("nav");
  if (!nav || nav.querySelector("a[data-page]")) return;   // absent or already built
  const brand = nav.querySelector(".nav-brand");
  const frag = document.createDocumentFragment();
  NAV_LINKS.forEach(link => {
    const a = document.createElement("a");
    a.href = link.href;
    a.dataset.page = link.page || link.href;
    a.textContent = link.label;
    frag.appendChild(a);
  });
  if (brand && brand.nextSibling) nav.insertBefore(frag, brand.nextSibling);
  else nav.appendChild(frag);
}

// Mark the nav link for the current page active
function markActiveNav() {
  const page = location.pathname.replace(/\/$/, "").split("/").pop() || "index.html";
  document.querySelectorAll("nav a[data-page]").forEach(a => {
    // data-page may list several pages (a hub + its sub-pages) space-separated.
    if (a.dataset.page.split(/\s+/).includes(page)) a.classList.add("active");
  });
}

// Build a responsive nav: wrap the page links in a collapsible panel and add a
// hamburger toggle for phones. Done in JS so every page's <nav> markup is the
// single existing flat list — no per-page edits needed.
function buildResponsiveNav() {
  const nav = document.querySelector("nav");
  if (!nav || nav.querySelector(".nav-toggle")) return;

  const links = [...nav.querySelectorAll("a:not(.nav-brand)")];
  if (!links.length) return;

  const panel = document.createElement("div");
  panel.className = "nav-links";
  links.forEach(a => panel.appendChild(a));

  const toggle = document.createElement("button");
  toggle.className = "nav-toggle";
  toggle.type = "button";
  toggle.setAttribute("aria-label", "Toggle menu");
  toggle.setAttribute("aria-expanded", "false");
  toggle.innerHTML = "<span></span><span></span><span></span>";

  nav.appendChild(toggle);
  nav.appendChild(panel);

  const setOpen = (open) => {
    nav.classList.toggle("open", open);
    toggle.setAttribute("aria-expanded", String(open));
  };
  toggle.addEventListener("click", () => setOpen(!nav.classList.contains("open")));
  panel.addEventListener("click", e => { if (e.target.closest("a")) setOpen(false); });
  // Close the menu if the viewport grows back to desktop width.
  window.matchMedia("(min-width: 701px)").addEventListener("change", e => {
    if (e.matches) setOpen(false);
  });
}

// ── Searchable select ───────────────────────────────────────────────────────
// Upgrades a native <select> into a type-to-filter combobox WITHOUT changing the
// markup contract: the <select> stays in the DOM as the source of truth, so all
// existing code (sel.value reads, "change" listeners) keeps working. Picking an
// option sets sel.value and dispatches a real "change" event.
//
// It self-gates on option count (only activates at >= threshold) and watches the
// <select> with a MutationObserver, so selects that are populated *after* load
// (most of ours are) upgrade automatically once their options arrive — and small
// dropdowns (season, position, class) are left as plain native selects.
const SS_THRESHOLD = 15;

function makeSearchable(sel, threshold = SS_THRESHOLD) {
  if (sel._ssInit) return;
  sel._ssInit = true;

  const wrap  = document.createElement("div");
  wrap.className = "ss-wrap";
  const input = document.createElement("input");
  input.type = "text"; input.className = "ss-input"; input.autocomplete = "off";
  input.setAttribute("role", "combobox");
  input.style.display = "none";   // hidden until activated (small selects stay native)
  const list  = document.createElement("div");
  list.className = "ss-list"; list.style.display = "none";

  sel.parentNode.insertBefore(wrap, sel.nextSibling);
  wrap.appendChild(sel); wrap.appendChild(input); wrap.appendChild(list);

  // Placeholder when nothing is selected — derived from the field's <label>.
  const labelEl = sel.id ? document.querySelector(`label[for="${sel.id}"]`) : null;
  const basePlaceholder = labelEl ? `Search ${labelEl.textContent.trim().toLowerCase()}…` : "Search…";

  let active = false, items = [], filtered = [], hi = -1;

  const selectedLabel = () => {
    const o = sel.options[sel.selectedIndex];
    return o ? o.textContent : "";
  };
  const readOptions = () => {
    items = [...sel.options].map(o => ({ value: o.value, label: o.textContent }));
  };
  // The input stays EMPTY (a real search bar); the current pick lives in the
  // placeholder so you still see what's selected. An empty value (e.g. the
  // "All teams" option) falls back to the generic "Search …" placeholder.
  function syncDisplay() {
    input.value = "";
    input.placeholder = sel.value !== "" ? selectedLabel() : basePlaceholder;
    input.disabled = sel.disabled;
  }
  function setActive(on) {
    active = on;
    sel.style.display   = on ? "none" : "";
    input.style.display = on ? "" : "none";
  }
  function refresh() {
    readOptions();
    setActive(items.length >= threshold);   // enforce display every time, not just on change
    if (active) syncDisplay();
  }
  function openList(q = "") {
    const ql = q.toLowerCase();
    filtered = items.filter(it => it.label.toLowerCase().includes(ql));
    list.innerHTML = filtered.length
      ? filtered.map((it, i) =>
          `<div class="ss-opt${it.value === sel.value ? " sel" : ""}" data-i="${i}">${it.label}</div>`).join("")
      : `<div class="ss-empty">No matches</div>`;
    list.style.display = ""; hi = -1;
  }
  const closeList = () => { list.style.display = "none"; input.value = ""; };
  function markHi() {
    [...list.children].forEach((c, i) => c.classList.toggle("hi", i === hi));
    if (hi >= 0 && list.children[hi]) list.children[hi].scrollIntoView({ block: "nearest" });
  }
  function choose(it) {
    if (!it) return;
    sel.value = it.value; list.style.display = "none";
    syncDisplay();
    // Fire BOTH events: a native <select> emits "input" and "change" on user
    // selection, and different pages bind to different ones.
    sel.dispatchEvent(new Event("input",  { bubbles: true }));
    sel.dispatchEvent(new Event("change", { bubbles: true }));
  }

  // Empty on focus so you can type immediately; the list shows all options.
  input.addEventListener("focus", () => { if (active) openList(""); });
  input.addEventListener("input", () => openList(input.value));
  input.addEventListener("keydown", e => {
    if (list.style.display === "none" && e.key === "ArrowDown") { openList(input.value); return; }
    if (e.key === "ArrowDown")      { hi = Math.min(hi + 1, filtered.length - 1); markHi(); e.preventDefault(); }
    else if (e.key === "ArrowUp")   { hi = Math.max(hi - 1, 0); markHi(); e.preventDefault(); }
    else if (e.key === "Enter")     { choose(hi >= 0 ? filtered[hi] : (filtered.length === 1 ? filtered[0] : null)); e.preventDefault(); }
    else if (e.key === "Escape")    { closeList(); input.blur(); }
  });
  // mousedown (not click) so it fires before the input loses focus
  list.addEventListener("mousedown", e => {
    const opt = e.target.closest(".ss-opt");
    if (!opt) return;
    e.preventDefault();
    choose(filtered[+opt.dataset.i]);
  });
  document.addEventListener("click", e => { if (!wrap.contains(e.target)) closeList(); });

  new MutationObserver(refresh).observe(sel, {
    childList: true, attributes: true, attributeFilter: ["disabled"],
  });
  refresh();
}

function enhanceAllSelects(threshold = SS_THRESHOLD) {
  document.querySelectorAll("select:not([data-no-search])").forEach(s => makeSearchable(s, threshold));
}

document.addEventListener("DOMContentLoaded", () => {
  populateNav();          // inject shared links before the nav is wrapped/marked
  buildResponsiveNav();
  markActiveNav();
  enhanceAllSelects();
});
