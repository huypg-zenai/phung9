import { parentPort, workerData } from 'node:worker_threads';
import * as THREE from 'three';
import { postValidate } from './validator/postValidation.js';
import { appendFileSync } from 'node:fs';

const SafeFunction = Function;
const NowMs = Date.now.bind(Date);
// [aplus] high-resolution clock + env flags captured before run() traps Date/performance/process.
const HrNow = (typeof performance !== 'undefined' && typeof performance.now === 'function')
  ? performance.now.bind(performance) : NowMs;
let APLUS_ROUNDTRIP_LOG = null;   // [aplus] item 4: opt-in, logging only (APLUS_ROUNDTRIP_CHECK=1)
try {
  if (process.env.APLUS_ROUNDTRIP_CHECK === '1') {
    APLUS_ROUNDTRIP_LOG = process.env.APLUS_ROUNDTRIP_LOG || '/tmp/aplus_roundtrip.jsonl';
  }
} catch {}
// [pgh1v3] switch: PGH_APLUS=1 turns on A+ items 1-3 (map-slot rejects + extra bakes). Read once at worker load,
// before run() traps process. Anything else (default '0') = the unpatched worker's behaviour, byte for byte.
let APLUS_ON = false;
try { APLUS_ON = process.env.PGH_APLUS === '1'; } catch {}

const TRAPPED_GLOBALS = [
  'setTimeout', 'setInterval', 'setImmediate', 'queueMicrotask',
  'fetch', 'XMLHttpRequest', 'WebSocket',
  'crypto',
  'Date', 'performance',
  'eval', 'Function',
  'process', 'global',
  'WeakRef', 'FinalizationRegistry',
  'SharedArrayBuffer', 'Atomics',
];

const CODEGEN_PROTOTYPES = [
  SafeFunction.prototype,
  Object.getPrototypeOf(function*(){}),
  Object.getPrototypeOf(async function(){}),
  Object.getPrototypeOf(async function*(){}),
];

function trapViolation(name) {
  throw new Error(`Runtime violation: ${name} is forbidden`);
}

function send(msg) {
  parentPort.postMessage(msg);
}

function geometryIsDirty(geo) {
  const attrs = geo.attributes || {};
  for (const name in attrs) {
    if (attrs[name] && attrs[name].version > 0) return true;
  }
  if (geo.index && geo.index.version > 0) return true;
  const morph = geo.morphAttributes || {};
  for (const name in morph) {
    const list = morph[name];
    if (Array.isArray(list)) {
      for (const a of list) if (a && a.version > 0) return true;
    }
  }
  return false;
}

// Curve classes ObjectLoader can rebuild (three/src/extras/curves). A user-defined `class X extends THREE.Curve`
// (or a CurvePath) used as a TubeGeometry path / ExtrudeGeometry extrudePath makes ObjectLoader throw
// "Curves[data.path.type] is not a constructor" on the render side, so such geometries must be baked.
const BUILTIN_CURVES = new Set([
  'ArcCurve', 'CatmullRomCurve3', 'CubicBezierCurve', 'CubicBezierCurve3', 'EllipseCurve',
  'LineCurve', 'LineCurve3', 'QuadraticBezierCurve', 'QuadraticBezierCurve3', 'SplineCurve',
]);

// Parametric geometries whose JSON form ObjectLoader cannot rehydrate:
//  - dirty attributes (the module edited vertices after construction) -> parameters no longer describe the mesh
//  - no `static fromJSON` on the class (EdgesGeometry, WireframeGeometry, user subclasses) -> "Geometries[data.type].fromJSON is not a function"
//  - TubeGeometry / ExtrudeGeometry driven by a non-built-in curve -> "Curves[data.path.type] is not a constructor"
function geometryNeedsBake(geo, multiMaterial = false) {
  if (geometryIsDirty(geo)) return true;
  const ctor = THREE[geo.type];
  if (!ctor || typeof ctor.fromJSON !== 'function') return true;
  if (geo.type === 'TubeGeometry') {
    const path = geo.parameters && geo.parameters.path;
    if (!path || !BUILTIN_CURVES.has(path.type)) return true;
  }
  if (geo.type === 'ExtrudeGeometry') {
    const ep = geo.parameters && geo.parameters.options && geo.parameters.options.extrudePath;
    if (ep && !BUILTIN_CURVES.has(ep.type)) return true;
  }
  // [aplus] item 3: toJSON keeps only `parameters` for a parametric geometry, so whatever the module added or
  // removed without bumping an attribute version (setAttribute('color', ...) for vertex colours, morphs,
  // drawRange, groups of a multi-material mesh) is lost on our render side, while the contest renders the
  // live object. Bake those too, so the JSON keeps the real buffers.
  if (APLUS_ON && geometryLosesDataInJSON(geo, multiMaterial)) return true;   // [pgh1v3] gated
  return false;
}

// [aplus] item 3 helpers -----------------------------------------------------------------------------
// Every built-in parametric geometry (three r183 and r185: Box, Capsule, Circle, Cone, Cylinder, Polyhedron family,
// Plane, Ring, Sphere, Torus, TorusKnot, Lathe, Shape, Extrude, Tube) builds exactly these attributes.
const PARAMETRIC_ATTRS = new Set(['position', 'normal', 'uv']);

// What ObjectLoader will build from this geometry's JSON (parameters + shapes), or null if it cannot.
function rebuildLikeLoader(geo) {
  try {
    const data = JSON.parse(JSON.stringify(geo.toJSON()));
    const ps = geo.parameters && geo.parameters.shapes;
    const shapesJSON = ps === undefined ? [] : (Array.isArray(ps) ? ps : [ps]).map((sh) => sh.toJSON());
    const loader = new THREE.ObjectLoader();
    const shapes = loader.parseShapes(JSON.parse(JSON.stringify(shapesJSON)));
    const out = loader.parseGeometries([data], shapes)[data.uuid];
    return out && out.isBufferGeometry ? out : null;
  } catch {
    return null;
  }
}

// Groups only matter for a mesh with a material array (three ignores them otherwise; e.g. ShapeGeometry(shape)
// has no groups live but one after fromJSON, which renders the same with one material). Only for those
// rare geometries do we pay for a rebuild.
function groupsDiffer(geo) {
  // The throwaway rebuild creates objects whose UUIDs come from Math.random. Give it its own stream so the
  // module's seeded sequence (and so every UUID in the serialized JSON) stays exactly as without the check.
  const seeded = Math.random;
  let s2 = 0x9E3779B9;
  Math.random = () => { s2 = (Math.imul(s2, 1664525) + 1013904223) >>> 0; return s2 / 4294967296; };
  try {
    const fresh = rebuildLikeLoader(geo);
    if (!fresh) return true;
    const ga = geo.groups || [], gb = fresh.groups || [];
    fresh.dispose();
    if (ga.length !== gb.length) return true;
    for (let i = 0; i < ga.length; i++) {
      if (ga[i].start !== gb[i].start || ga[i].count !== gb[i].count || ga[i].materialIndex !== gb[i].materialIndex) return true;
    }
    return false;
  } finally {
    Math.random = seeded;
  }
}

// Returns '' when the parameters-only JSON reproduces the live geometry as far as we can check cheaply.
function jsonLossReason(geo, multiMaterial) {
  const attrs = geo.attributes || {};
  for (const name in attrs) {
    if (!PARAMETRIC_ATTRS.has(name)) return `extra attribute '${name}'`;   // the vertex-colour class
  }
  for (const name of PARAMETRIC_ATTRS) {
    if (!attrs[name]) return `missing attribute '${name}'`;
  }
  const morph = geo.morphAttributes || {};
  for (const name in morph) {
    if (Array.isArray(morph[name]) && morph[name].length > 0) return 'morph attributes';
  }
  const dr = geo.drawRange;
  if (dr && (dr.start !== 0 || dr.count !== Infinity)) return 'drawRange';
  if (multiMaterial && groupsDiffer(geo)) return 'groups of a multi-material mesh';
  return '';
}

let lastBakeReasons = null;   // [aplus] per-run record of item-3 bakes (for logging/tests only)
let rebuildCheckMs = 0;       // [aplus] time spent in the item-3 check (the patch's added cost)

function geometryLosesDataInJSON(geo, multiMaterial) {
  const t0 = HrNow();
  const reason = jsonLossReason(geo, multiMaterial);
  rebuildCheckMs += HrNow() - t0;
  if (reason && lastBakeReasons) lastBakeReasons.push(`${geo.type}: ${reason}`);
  return reason !== '';
}

// [aplus] items 1+2: material map slots the contest renderer would crash on or render black.
// Slots are matched by key pattern (any '*Map', 'map', 'matcap'), so new three slots are covered too.
// envMap is skipped: the contest (render-pool.js) and our object-page null it before rendering.
const MAP_KEY_RE = /Map$|^map$|^matcap$/;

function describeValue(v) {
  try {
    if (v === null || typeof v !== 'object') return typeof v;
    if (typeof v.type === 'string' && v.type) return v.type;
    if (v.isDataTexture) return 'DataTexture';
    return (v.constructor && v.constructor.name) || 'object';
  } catch {
    return 'object';
  }
}

function describeOwner(o, mat) {
  const on = o.name ? ` "${o.name}"` : '';
  const mn = mat.name ? ` "${mat.name}"` : '';
  return `${o.type || 'Object3D'}${on} material${mn} (${mat.type || 'Material'})`;
}

// Only objects three actually draws with their material (a stray .material on a Group is never used).
const isDrawable = (o) => !!(o && (o.isMesh || o.isLine || o.isPoints || o.isSprite));

function checkMaterialMaps(root) {
  const failures = [];
  const seenMat = new Set();
  const seenKey = new Set();
  root.traverse((o) => {
    if (!isDrawable(o)) return;
    const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
    for (const mat of mats) {
      if (!mat || typeof mat !== 'object' || seenMat.has(mat)) continue;
      seenMat.add(mat);
      for (const key of Object.keys(mat)) {
        if (key === 'envMap' || !MAP_KEY_RE.test(key)) continue;
        const v = mat[key];
        if (!v) continue;   // three only uses a slot when it is truthy
        let rule, detail;
        if (!v.isTexture) {
          rule = 'NON_TEXTURE_MAP';
          detail = `${describeOwner(o, mat)}.${key} holds a ${describeValue(v)}, not a THREE.Texture; `
            + `the contest renderer crashes on it ("Cannot read properties of undefined (reading 'elements')"). `
            + `Remove it or assign a real Texture.`;
        } else if (v.version === 0 && !v.isRenderTargetTexture) {
          rule = 'TEXTURE_NOT_UPLOADED';
          detail = `${describeOwner(o, mat)}.${key} is a ${describeValue(v)} whose needsUpdate was never set `
            + `(version 0); the contest never uploads it and renders it black. `
            + `Set ${key}.needsUpdate = true after filling its data.`;
        } else {
          continue;
        }
        const sig = `${rule}:${key}:${describeValue(v)}`;
        if (seenKey.has(sig) || failures.length >= 4) continue;
        seenKey.add(sig);
        failures.push({ stage: 'post_validation', rule, detail });
      }
    }
  });
  return failures;
}

// [aplus] item 4 (opt-in, logging only): ObjectLoader round trip of the final JSON vs the live object.
function positionChecksum(attr) {
  let sum = 0;
  for (let i = 0; i < attr.count; i++) sum += Math.abs(attr.getX(i)) + 2 * Math.abs(attr.getY(i)) + 3 * Math.abs(attr.getZ(i));
  return sum;
}

function roundTripDiffs(live, json) {
  const diffs = [];
  const rt = new THREE.ObjectLoader().parse(json);
  const L = [], R = [];
  live.traverse((o) => L.push(o));
  rt.traverse((o) => R.push(o));
  if (L.length !== R.length) return [`node count ${L.length} vs ${R.length}`];
  for (let i = 0; i < L.length && diffs.length < 8; i++) {
    const a = L[i], b = R[i], nm = a.name || a.type;
    if (a.type !== b.type) diffs.push(`${nm}: type ${a.type}->${b.type}`);
    if (a.geometry && b.geometry) {
      const ka = Object.keys(a.geometry.attributes).sort().join(','), kb = Object.keys(b.geometry.attributes).sort().join(',');
      if (ka !== kb) diffs.push(`${nm}: attributes [${ka}] -> [${kb}]`);
      const pa = a.geometry.attributes.position, pb = b.geometry.attributes.position;
      if (pa && pb && pa.count !== pb.count) diffs.push(`${nm}: vertex count ${pa.count}->${pb.count}`);
      else if (pa && pb) {   // content check: catches in-place edits made without needsUpdate
        const sa = positionChecksum(pa), sb = positionChecksum(pb);
        if (Math.abs(sa - sb) > 1e-4 * Math.max(1, Math.abs(sa))) diffs.push(`${nm}: positions differ`);
      }
      const ia = a.geometry.index ? a.geometry.index.count : -1, ib = b.geometry.index ? b.geometry.index.count : -1;
      if (ia !== ib) diffs.push(`${nm}: index ${ia}->${ib}`);
    }
    if (!isDrawable(a)) continue;
    const ma = Array.isArray(a.material) ? a.material : (a.material ? [a.material] : []);
    const mb = Array.isArray(b.material) ? b.material : (b.material ? [b.material] : []);
    if (ma.length !== mb.length) { diffs.push(`${nm}: material count`); continue; }
    for (let j = 0; j < ma.length; j++) {
      if (!ma[j] || !mb[j]) continue;
      if (ma[j].type !== mb[j].type) diffs.push(`${nm}: material ${ma[j].type}->${mb[j].type}`);
      for (const k of Object.keys(ma[j])) {
        if (k === 'envMap' || !MAP_KEY_RE.test(k)) continue;
        if (!!ma[j][k] !== !!mb[j][k]) diffs.push(`${nm}.${k}: ${ma[j][k] ? 'dropped' : 'added'}`);
      }
    }
  }
  return diffs;
}

function fnv1a(text) {   // tiny source id for the log (node:crypto would add worker start-up time)
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h.toString(16).padStart(8, '0');
}

function logRoundTrip(rawResult, object) {
  if (APLUS_ROUNDTRIP_LOG === null || !object) return;
  const t0 = HrNow();
  let diffs;
  try {
    diffs = roundTripDiffs(rawResult, object);
  } catch (err) {
    diffs = [`round trip threw: ${err && err.message ? err.message : String(err)}`];
  }
  if (diffs.length === 0) return;
  try {
    appendFileSync(APLUS_ROUNDTRIP_LOG, JSON.stringify({
      source_fnv1a: fnv1a(String(workerData.source)), ms: Math.round((HrNow() - t0) * 10) / 10, diffs,
    }) + '\n');
  } catch {}
}
// [aplus] end helpers --------------------------------------------------------------------------------

function bakeModifiedParametricGeometries(root) {
  const baked = new Map();
  const decided = new Map();   // [aplus] decide once per shared geometry (the rebuild check is not free)
  const multi = new Set();     // [aplus] geometries drawn with a material array (their groups matter)
  if (APLUS_ON) root.traverse((o) => { if (o.geometry && Array.isArray(o.material)) multi.add(o.geometry); });
  root.traverse((o) => {
    const geo = o.geometry;
    if (!APLUS_ON) {   // [pgh1v3] switch off: the unpatched worker's exact decision
      if (!geo || geo.parameters === undefined || !geometryNeedsBake(geo)) return;
    } else {
      if (!geo || geo.parameters === undefined) return;
      if (!decided.has(geo)) decided.set(geo, geometryNeedsBake(geo, multi.has(geo)));
      if (!decided.get(geo)) return;
    }
    if (!baked.has(geo)) {
      const bg = new THREE.BufferGeometry().copy(geo);
      bg.userData = {
        ...(geo.userData || {}),
        originalGeometry: { type: geo.type, parameters: { ...geo.parameters } },
      };
      baked.set(geo, bg);
    }
    o.geometry = baked.get(geo);
  });
}

function run() {
  const { source } = workerData;

  const origRandom = Math.random;
  const savedGlobals = new Map();
  const savedCtors = [];

  let moduleLoadMs = 0;
  let executionMs = 0;
  const moduleStart = NowMs();

  try {
    let seed = 0x12345678;
    Math.random = () => {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };

    for (const name of TRAPPED_GLOBALS) {
      if (!(name in globalThis)) continue;
      savedGlobals.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
      try {
        Object.defineProperty(globalThis, name, {
          get() { trapViolation(name); },
          configurable: true,
        });
      } catch {
        try { globalThis[name] = undefined; } catch {}
      }
    }

    for (const proto of CODEGEN_PROTOTYPES) {
      const desc = Object.getOwnPropertyDescriptor(proto, 'constructor');
      savedCtors.push({ proto, desc });
      Object.defineProperty(proto, 'constructor', {
        get() { trapViolation('Function constructor'); },
        configurable: true,
      });
    }

    const body = `'use strict';\n${source}`;
    let factory;
    try {
      factory = new SafeFunction(body);
    } catch (err) {
      moduleLoadMs = NowMs() - moduleStart;
      send({
        failures: [{
          stage: 'module_load',
          rule: 'EXECUTION_THREW',
          detail: `compilation failed: ${err && err.message ? err.message : String(err)}`,
        }],
        metrics: null,
        moduleLoadMs,
        executionMs: 0,
      });
      return;
    }
    moduleLoadMs = NowMs() - moduleStart;

    const execStart = NowMs();
    let generate;
    try {
      generate = factory();
    } catch (err) {
      executionMs = NowMs() - execStart;
      const rule = err && err.message && err.message.includes('Runtime violation')
        ? 'RUNTIME_VIOLATION'
        : 'EXECUTION_THREW';
      send({
        failures: [{ stage: 'module_load', rule, detail: err && err.message ? err.message : String(err) }],
        metrics: null,
        moduleLoadMs,
        executionMs,
      });
      return;
    }

    if (typeof generate !== 'function') {
      executionMs = NowMs() - execStart;
      send({
        failures: [{
          stage: 'module_load',
          rule: 'INVALID_RETURN_TYPE',
          detail: `default export is not a function (got ${typeof generate})`,
        }],
        metrics: null,
        moduleLoadMs,
        executionMs,
      });
      return;
    }

    let rawResult;
    try {
      rawResult = generate(THREE);
    } catch (err) {
      executionMs = NowMs() - execStart;
      const rule = err && err.message && err.message.includes('Runtime violation')
        ? 'RUNTIME_VIOLATION'
        : 'EXECUTION_THREW';
      send({
        failures: [{ stage: 'execution', rule, detail: err && err.message ? err.message : String(err) }],
        metrics: null,
        moduleLoadMs,
        executionMs,
      });
      return;
    }
    executionMs = NowMs() - execStart;

    if (
      rawResult !== null &&
      (typeof rawResult === 'object' || typeof rawResult === 'function') &&
      typeof rawResult.then === 'function'
    ) {
      send({
        failures: [{
          stage: 'execution',
          rule: 'ASYNC_NOT_ALLOWED',
          detail: 'generate() returned a Promise or thenable',
        }],
        metrics: null,
        moduleLoadMs,
        executionMs,
      });
      return;
    }

    let postResult;
    try {
      postResult = postValidate(rawResult);
    } catch (err) {
      send({
        failures: [{
          stage: 'post_validation',
          rule: 'EXECUTION_THREW',
          detail: err && err.message ? err.message : String(err),
        }],
        metrics: null,
        moduleLoadMs,
        executionMs,
      });
      return;
    }

    let object = null;
    let aplusMs = null;
    let aplusBakes = null;
    if (postResult.failures.length === 0) {
      // [aplus] items 1+2: reject slots the contest renders differently from our JSON copy; the failure
      // goes back through the normal js_errors -> code_repair loop like any other post-validation error.
      const tMaps = HrNow();
      const mapFailures = APLUS_ON ? checkMaterialMaps(rawResult) : [];   // [pgh1v3] gated
      aplusMs = HrNow() - tMaps;
      if (mapFailures.length > 0) {
        send({
          failures: mapFailures,
          metrics: postResult.metrics,
          moduleLoadMs,
          executionMs,
          aplusMs,
        });
        return;
      }
      try {
        lastBakeReasons = [];
        rebuildCheckMs = 0;
        bakeModifiedParametricGeometries(rawResult);
        aplusBakes = lastBakeReasons;
        aplusMs += rebuildCheckMs;
        object = rawResult.toJSON();
      } catch (err) {
        send({
          failures: [{
            stage: 'post_validation',
            rule: 'SERIALIZE_THREW',
            detail: `toJSON() failed: ${err && err.message ? err.message : String(err)}`,
          }],
          metrics: postResult.metrics,
          moduleLoadMs,
          executionMs,
        });
        return;
      }
    }

    if (object) logRoundTrip(rawResult, object);   // [aplus] item 4, no-op unless APLUS_ROUNDTRIP_CHECK=1

    send({
      failures: postResult.failures,
      metrics: postResult.metrics,
      moduleLoadMs,
      executionMs,
      object,
      // [aplus] extra fields; serialize_runner.mjs forwards neither, so its output is unchanged
      // [pgh1v3] and with the switch off they are not even sent (message = the unpatched worker's)
      ...(APLUS_ON ? { aplusMs, aplusBakes } : {}),
    });
  } finally {
    Math.random = origRandom;
    for (const [name, desc] of savedGlobals) {
      try { Object.defineProperty(globalThis, name, desc); }
      catch { try { globalThis[name] = desc?.value; } catch {} }
    }
    for (const { proto, desc } of savedCtors) {
      try { Object.defineProperty(proto, 'constructor', desc); }
      catch {}
    }
  }
}

try {
  run();
} catch (err) {
  send({
    failures: [{
      stage: 'execution',
      rule: 'EXECUTION_THREW',
      detail: err && err.stack ? err.stack : String(err),
    }],
    metrics: null,
    moduleLoadMs: 0,
    executionMs: 0,
  });
}
