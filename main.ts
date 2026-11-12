// pumpfun-top5-forecast-01 entry: horizon-averaged Monte Carlo forecaster.
// Node 22, standard library only. Reads /input (task.json + history.ndjson),
// writes /output/forecast.json. Deterministic: one fixed-seed PRNG, fixed
// iteration order, no wall clock, no network, no Math.random.
//
// Model, in one paragraph: every candidate coin's future log10 market cap
// (in SOL) is a shrunk-drift normal whose mean reverts toward its last mark
// as the horizon grows, mixed with a "death" component that sends young
// bonding-curve coins back to the curve floor. At each answer-read horizon
// from task.json we draw the whole candidate chart many times with a seeded
// PRNG, count who holds each place and which size bin each place's cap falls
// in, and average over horizons. Place probabilities keep a large unnamed
// reserve because the rules' own practice shows most late top-5 holders were
// coins the history never ranked: no forecast can name them, so honest mass
// must stay unnamed. Sizes come from the same draws (order statistics), not
// from a separate guessed distribution, so order and size stay coherent.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MICRO = 1_000_000;

// --- Fixed calibration (set in advance, never tuned on any answer) ---
/** Monte Carlo draws per answer-read horizon. 12000 keeps the standard error of a
 *  p=0.30 place probability near 0.004 (vs 0.007 at 4000) while the whole run stays
 *  far under the 300s cap; simulation noise must not decide places the model ties. */
const DRAWS_PER_HORIZON = 12000;
/** Candidate coins simulated; the rest share the unnamed mass. */
const MAX_CANDIDATES = 220;
/** Unnamed mass per place: coins born after the cut take places too. */
const RESERVE_PLACE = 0.42;
/** Unnamed mass per place's size distribution. */
const RESERVE_SIZE = 0.1;
/** Rows named per place (rule cap is 100; stay below it so the unnamed share stays well defined). */
const MAX_ROWS_PER_PLACE = 80;
/** Drift saturates: effective drift hours = TAU * (1 - exp(-h / TAU)). */
const DRIFT_TAU_H = 3.5;
/** |shrunk drift| cap, log10 SOL per hour. Raw young-coin slopes run far hotter and must not extrapolate. */
const MAX_DRIFT_PER_H = 0.6;
/** Shrinkage strength toward the cross-sectional drift prior (in marks). */
const SHRINK_K = 3;
/** Baseline future sd (mark noise + jump risk), log10 SOL. */
const SIGMA0 = 0.2;
/** Cap on future sd. */
const SIGMA_CAP = 1.2;
/** Death component: log10 cap distribution of a faded curve coin. */
const FLOOR_MEAN = 1.5;
const FLOOR_SD = 0.35;
/** Death probability by venue at age zero; age damps it toward 40%. */
const Q_POOL = 0.08;
const Q_COMPLETE = 0.15;
const Q_CURVE = 0.3;
/** Clamp on a coin's predicted mean log10 cap. */
const MU_LO = 1.0;
const MU_HI = 6.5;

interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
}

interface Mark {
  at: string;
  ok: boolean;
  rows: Row[] | null;
}

interface Point {
  tH: number;
  logCap: number;
  rank: number;
  venue: string;
  complete: boolean;
}

interface Coin {
  mint: string;
  createdMs: number;
  points: Point[];
  last: Point;
  drift: number;
  vol: number;
  q: number;
  ageH: number;
}

/** log10 of a cap in SOL, as the scorer takes it: log10(lamports) - 9. */
function sizeOf(lamports: string | number): number {
  return Math.log10(Number(BigInt(lamports))) - 9;
}

/** Deterministic PRNG (mulberry32). One stream, consumed in fixed order. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Inverse standard normal CDF (Acklam's rational approximation). */
function normInv(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  const phigh = 1 - plow;
  let q: number;
  let r: number;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) / ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) / ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q / (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1);
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  reads: string[];
  t0: string;
  size: { edges: number[] };
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8').split('\n').filter((l) => l !== '');
const marks = lines
  .map((l) => JSON.parse(l) as Mark)
  .filter((m) => m.ok === true && Array.isArray(m.rows) && m.rows.length > 0);
if (marks.length === 0) throw new Error('the history holds no ok mark with rows');

const t0Ms = Date.parse(task.t0);
const lastMark = marks[marks.length - 1]!;
const lastMs = Date.parse(lastMark.at);

// --- Build per-coin series ---
const byMint = new Map<string, Coin>();
for (const m of marks) {
  const tH = (Date.parse(m.at) - t0Ms) / 3600000;
  for (const r of m.rows!) {
    let c = byMint.get(r.mint);
    if (!c) {
      c = { mint: r.mint, createdMs: Date.parse(r.created_at), points: [], last: null as unknown as Point, drift: 0, vol: 0, q: 0, ageH: 0 };
      byMint.set(r.mint, c);
    }
    const p: Point = { tH, logCap: sizeOf(r.market_cap_lamports), rank: r.rank, venue: r.venue, complete: r.complete };
    c.points.push(p);
    c.last = p;
  }
}
const coins = [...byMint.values()];
for (const c of coins) {
  c.ageH = Math.max(0, (lastMs - c.createdMs) / 3600000);
}

// --- Drift + volatility per coin, shrunk toward cross-sectional priors ---
interface Fit { slope: number; residSd: number; n: number }
const fits = new Map<string, Fit>();
for (const c of coins) {
  const pts = c.points;
  if (pts.length < 2) {
    fits.set(c.mint, { slope: 0, residSd: 0, n: pts.length });
    continue;
  }
  const n = pts.length;
  const mx = pts.reduce((a, p) => a + p.tH, 0) / n;
  const my = pts.reduce((a, p) => a + p.logCap, 0) / n;
  let num = 0;
  let den = 0;
  for (const p of pts) {
    num += (p.tH - mx) * (p.logCap - my);
    den += (p.tH - mx) * (p.tH - mx);
  }
  const slope = den > 0 ? num / den : 0;
  const intercept = my - slope * mx;
  let ss = 0;
  for (const p of pts) ss += (p.logCap - (intercept + slope * p.tH)) ** 2;
  const residSd = n > 2 ? Math.sqrt(ss / (n - 2)) : Math.abs(pts[1]!.logCap - pts[0]!.logCap) / 2;
  fits.set(c.mint, { slope, residSd, n });
}
const multi = coins.filter((c) => fits.get(c.mint)!.n >= 2);
const priorSlope = clamp(median(multi.map((c) => fits.get(c.mint)!.slope)), -0.2, 0.4);
const priorVol = clamp(median(multi.map((c) => fits.get(c.mint)!.residSd)), 0.15, 0.8);
for (const c of coins) {
  const f = fits.get(c.mint)!;
  const shrunk = (f.n * f.slope + SHRINK_K * priorSlope) / (f.n + SHRINK_K);
  c.drift = clamp(shrunk, -MAX_DRIFT_PER_H, MAX_DRIFT_PER_H);
  c.vol = clamp((f.n * f.residSd + SHRINK_K * priorVol) / (f.n + SHRINK_K), 0.15, 0.8);
  const base = c.last.venue === 'pool' ? Q_POOL : c.last.complete ? Q_COMPLETE : Q_CURVE;
  c.q = base * (0.4 + 0.6 * Math.exp(-c.ageH / 4));
}

// --- Candidates: last mark's chart plus recent top-30 risers, capped ---
const lastMints = new Set(lastMark.rows!.map((r) => r.mint));
const candSet = new Set<string>(lastMints);
const recent = marks.slice(-3);
for (const m of recent) {
  for (const r of m.rows!.slice(0, 30)) candSet.add(r.mint);
}
let candidates = [...candSet].map((mint) => byMint.get(mint)!).filter(Boolean);
candidates.sort((a, b) => b.last.logCap - a.last.logCap || (a.mint < b.mint ? -1 : 1));
candidates = candidates.slice(0, MAX_CANDIDATES);
// Fixed sampling order, independent of values.
candidates.sort((a, b) => (a.mint < b.mint ? -1 : a.mint > b.mint ? 1 : 0));
if (candidates.length < task.places) throw new Error(`only ${candidates.length} candidates for ${task.places} places`);

// --- Horizons from task.json (never hardcoded) ---
const readsMs = (task.reads ?? []).map((r) => Date.parse(r)).filter((x) => Number.isFinite(x));
const horizons = readsMs.length > 0
  ? readsMs.map((t) => clamp((t - lastMs) / 3600000, 0, 24))
  : [10];
const edges = task.size.edges;
const nBins = edges.length + 1;

// Correct bin mapping: bin 0 below edges[0]; bin i (1..edges.length-1) is
// [edges[i-1], edges[i]); bin edges.length is >= edges[last].
function binIndex(value: number): number {
  if (value < edges[0]!) return 0;
  for (let i = 1; i < edges.length; i++) {
    if (value < edges[i]!) return i;
  }
  return edges.length;
}

const places = task.places;
const placeCounts: Map<string, number>[] = Array.from({ length: places }, () => new Map());
const sizeCounts: number[][] = Array.from({ length: places }, () => new Array(nBins).fill(0));
const rankSums = new Map<string, number>();
for (const c of candidates) rankSums.set(c.mint, 0);

const rng = mulberry32(0x9e3779b9);
const uniform = (): number => {
  const u = rng();
  return u <= 0 ? 1e-12 : u >= 1 ? 1 - 1e-12 : u;
};

const sampled = new Array<number>(candidates.length);
for (const h of horizons) {
  const hEff = DRIFT_TAU_H * (1 - Math.exp(-h / DRIFT_TAU_H));
  const mus: number[] = new Array(candidates.length);
  const sigmas: number[] = new Array(candidates.length);
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i]!;
    mus[i] = clamp(c.last.logCap + c.drift * hEff, MU_LO, MU_HI);
    sigmas[i] = Math.min(SIGMA_CAP, Math.sqrt(SIGMA0 * SIGMA0 + c.vol * c.vol * Math.max(h, 0.25)));
  }
  for (let d = 0; d < DRAWS_PER_HORIZON; d++) {
    for (let i = 0; i < candidates.length; i++) {
      const c = candidates[i]!;
      if (uniform() < c.q) {
        sampled[i] = FLOOR_MEAN + FLOOR_SD * normInv(uniform());
      } else {
        sampled[i] = mus[i]! + sigmas[i]! * normInv(uniform());
      }
    }
    const order = candidates.map((c, i) => i).sort((a, b) => {
      const dv = sampled[b]! - sampled[a]!;
      if (dv !== 0) return dv;
      const ma = candidates[a]!.mint;
      const mb = candidates[b]!.mint;
      return ma < mb ? -1 : 1;
    });
    for (let k = 0; k < places; k++) {
      const idx = order[k]!;
      const mint = candidates[idx]!.mint;
      const pc = placeCounts[k]!;
      pc.set(mint, (pc.get(mint) ?? 0) + 1);
      sizeCounts[k]![binIndex(sampled[idx]!)]!++;
    }
    for (let pos = 0; pos < order.length; pos++) {
      const mint = candidates[order[pos]!]!.mint;
      rankSums.set(mint, rankSums.get(mint)! + (pos + 1));
    }
  }
}

const totalDraws = DRAWS_PER_HORIZON * horizons.length;
const floor6 = (p: number): number => Math.floor(p * MICRO) / MICRO;

const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let k = 0; k < places; k++) {
  const rows = [...placeCounts[k]!.entries()]
    .map(([mint, count]) => ({ mint, p: floor6((count / totalDraws) * (1 - RESERVE_PLACE)) }))
    .filter((r) => r.p > 0)
    .sort((a, b) => b.p - a.p || (a.mint < b.mint ? -1 : 1))
    .slice(0, MAX_ROWS_PER_PLACE);
  for (const r of rows) probabilities.push({ mint: r.mint, place: k + 1, p: r.p });
}

const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let k = 0; k < places; k++) {
  for (let b = 0; b < nBins; b++) {
    const p = floor6((sizeCounts[k]![b]! / totalDraws) * (1 - RESERVE_SIZE));
    if (p > 0) sizes.push({ place: k + 1, bin: b, p });
  }
}

// top5 by mean simulated rank (ties: higher place-1 share, then mint).
const meanRank = (mint: string): number => rankSums.get(mint)! / totalDraws;
const p1 = (mint: string): number => (placeCounts[0]!.get(mint) ?? 0) / totalDraws;
const top5 = [...candidates]
  .sort((a, b) => meanRank(a.mint) - meanRank(b.mint) || p1(b.mint) - p1(a.mint) || (a.mint < b.mint ? -1 : 1))
  .slice(0, places)
  .map((c) => c.mint);

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);
