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

// Mark the nav link for the current page active
function markActiveNav() {
  const page = location.pathname.replace(/\/$/, "").split("/").pop() || "index.html";
  document.querySelectorAll("nav a[data-page]").forEach(a => {
    if (a.dataset.page === page) a.classList.add("active");
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

document.addEventListener("DOMContentLoaded", () => {
  buildResponsiveNav();
  markActiveNav();
});
