/**
 * Minimal Puppeteer-based sidecar for renderer stage.
 *
 * Exposes these HTTP endpoints:
 *   GET  /ping              → 200 when browser ready, 204 while warming up
 *   POST /render/views      → body {source|object, views?, options?} → JSON {views: {key: b64}}
 *   POST /render/grid       → LEGACY/debug-only: {source, options?} → PNG (2×2 grid)
 *
 * Deliberately omits the validation pool from the original render-service-js
 * server.js because the js_checker stage has already validated the code.
 */

import http from 'node:http';

import { startStaticServer } from './static-server.js';
import { ensureBrowser, closeBrowser, getBrowserPid } from './browser.js';
import { renderGrid, renderViews, buildParams } from './renderer.js';
import { objectPool } from './object-pool.js';

const VIEW_PRESETS = {
  // theta (azimuth, degrees from +Z axis, ccw around +Y), phi (elevation from horizon)
  front:       { theta:   0.0, phi:   0.0 },
  back:        { theta: 180.0, phi:   0.0 },
  right:       { theta:  90.0, phi:   0.0 },
  left:        { theta: 270.0, phi:   0.0 },
  top:         { theta:   0.0, phi:  89.0 },
  bottom:      { theta:   0.0, phi: -89.0 },
  perspective: { theta:  24.0, phi: -15.0 },
  // phi follows sphericalToCartesian's convention (y = -radius*sin(phi)):
  // +phi views from below, -phi from above. Keep in sync with the judge labels.
  front_left:  { theta:  30.0, phi:   0.0 },
  front_right: { theta: 330.0, phi:   0.0 },
  front_below: { theta:   0.0, phi:  15.0 },
  front_above: { theta:   0.0, phi: -30.0 },
  top_down:    { theta:   0.0, phi: -90.0 },
};

const PORT = parseInt(process.env.PORT || '8003', 10);

let ready = false;
let shuttingDown = false;

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks).toString('utf-8');
}

function sendJson(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
  });
  res.end(body);
}

function sendPng(res, buf) {
  res.writeHead(200, {
    'Content-Type': 'image/png',
    'Content-Length': buf.length,
  });
  res.end(buf);
}

// Unknown lighting is rejected instead of silently falling back to 'studio'.
const LIGHTING_MODES = new Set(['follow', 'studio', 'neutral']);

function lightingError(options) {
  const lighting = options.lighting;
  if (lighting == null) return null;
  if (!LIGHTING_MODES.has(lighting)) {
    return `unknown lighting '${lighting}'; expected one of: ${[...LIGHTING_MODES].join(', ')}`;
  }
  return null;
}

async function handleRenderViews(req, res) {
  const t0 = Date.now();
  let body;
  try {
    body = await readBody(req);
  } catch (err) {
    sendJson(res, 400, { error: `body read failed: ${err.message}` });
    return;
  }

  let payload;
  try {
    payload = JSON.parse(body);
  } catch (err) {
    sendJson(res, 400, { error: `invalid JSON: ${err.message}` });
    return;
  }

  // Renders from scene_json (warm page, no code execution) when "object" is
  // given, or executes "source" in a fresh context otherwise.
  const source = payload.source;
  const object = payload.object;
  const hasSource = typeof source === 'string' && source.length > 0;
  const hasObject = object != null && typeof object === 'object';
  if (hasSource === hasObject) {
    sendJson(res, 400, { error: 'exactly one of "source" or "object" is required' });
    return;
  }

  // A "views" entry: preset name, {name, bg?, key?}, or raw {theta, phi, key, bg?}
  // (key required). bg = per-view background hex, key names the response entry.
  const viewSpecs = Array.isArray(payload.views)
    ? payload.views
    : ['front', 'right', 'top', 'perspective'];

  const thetas = [];
  const phis = [];
  const labels = [];
  const bgs = [];
  const sizes = [];
  const lightings = [];  // [jgrid] optional per-view lighting (object path only)
  for (const spec of viewSpecs) {
    const name = typeof spec === 'string' ? spec : spec?.name;
    if (name == null && typeof spec === 'object' && spec !== null) {
      if (!Number.isFinite(spec.theta) || !Number.isFinite(spec.phi)) {
        sendJson(res, 400, { error: 'raw-angle view spec requires finite "theta" and "phi"' });
        return;
      }
      if (typeof spec.key !== 'string' || spec.key.length === 0) {
        sendJson(res, 400, { error: 'raw-angle view spec requires a non-empty "key"' });
        return;
      }
      thetas.push(spec.theta);
      phis.push(spec.phi);
      labels.push(spec.key);
      bgs.push(spec.bg || null);
      sizes.push(Number.isInteger(spec.size) ? spec.size : null);
      lightings.push(spec.lighting || null);
      continue;
    }
    if (typeof spec === 'object' && spec !== null && (spec.theta != null || spec.phi != null)) {
      sendJson(res, 400, { error: 'view spec cannot mix "name" with "theta"/"phi"' });
      return;
    }
    const preset = VIEW_PRESETS[name];
    if (!preset) {
      sendJson(res, 400, { error: `unknown view '${name}'. Known: ${Object.keys(VIEW_PRESETS).join(', ')}` });
      return;
    }
    thetas.push(preset.theta);
    phis.push(preset.phi);
    labels.push(typeof spec === 'string' ? name : (spec.key || name));
    bgs.push(typeof spec === 'string' ? null : (spec.bg || null));
    sizes.push(typeof spec === 'object' && spec !== null && Number.isInteger(spec.size) ? spec.size : null);
    lightings.push(typeof spec === 'object' && spec !== null ? (spec.lighting || null) : null);
  }
  const badLighting = lightings.find((l) => l !== null && !LIGHTING_MODES.has(l));
  if (badLighting !== undefined) {
    sendJson(res, 400, { error: `unknown view lighting '${badLighting}'; expected one of: ${[...LIGHTING_MODES].join(', ')}` });
    return;
  }
  if (new Set(labels).size !== labels.length) {
    sendJson(res, 400, { error: 'duplicate view keys in request' });
    return;
  }

  const options = { ...(payload.options || {}), thetas, phis, bgs, sizes: sizes.some((x) => x) ? sizes : null };
  if (lightings.some((x) => x)) options.lightings = lightings;  // [jgrid] absent unless a view asks for it

  const lightingErr = lightingError(options);
  if (lightingErr) {
    sendJson(res, 400, { error: lightingErr });
    return;
  }

  try {
    const pngBufs = hasObject
      ? await objectPool.renderObjectViews(object, buildParams(options))
      : await renderViews(source, options);
    const ms = Date.now() - t0;
    // Response: JSON { views: { name: base64_png, ... }, ms }
    const views = {};
    let total = 0;
    for (let i = 0; i < labels.length; i++) {
      views[labels[i]] = pngBufs[i].toString('base64');
      total += pngBufs[i].length;
    }
    console.log(`[render_runner] /render/views ok (${hasObject ? 'object' : 'code'}): ${labels.length} views ${total}B in ${ms}ms`);
    // [jgrid] report that per-view lighting was applied (absent otherwise, so off responses are unchanged)
    sendJson(res, 200, pngBufs.lightingsApplied ? { views, ms, lightings_applied: true } : { views, ms });
  } catch (err) {
    const ms = Date.now() - t0;
    console.error(`[render_runner] /render/views fail in ${ms}ms: ${err.message}`);
    sendJson(res, 500, { error: err.message || String(err) });
  }
}

async function handleRenderGrid(req, res) {
  const t0 = Date.now();
  let body;
  try {
    body = await readBody(req);
  } catch (err) {
    sendJson(res, 400, { error: `body read failed: ${err.message}` });
    return;
  }

  let payload;
  try {
    payload = JSON.parse(body);
  } catch (err) {
    sendJson(res, 400, { error: `invalid JSON: ${err.message}` });
    return;
  }

  const source = payload.source;
  if (typeof source !== 'string' || source.length === 0) {
    sendJson(res, 400, { error: 'missing "source" string' });
    return;
  }

  const options = payload.options || {};

  const lightingErr = lightingError(options);
  if (lightingErr) {
    sendJson(res, 400, { error: lightingErr });
    return;
  }

  try {
    const pngBuf = await renderGrid(source, options);
    const ms = Date.now() - t0;
    console.log(`[render_runner] /render/grid ok: ${pngBuf.length}B in ${ms}ms`);
    sendPng(res, pngBuf);
  } catch (err) {
    const ms = Date.now() - t0;
    console.error(`[render_runner] /render/grid fail in ${ms}ms: ${err.message}`);
    sendJson(res, 500, { error: err.message || String(err) });
  }
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');

    if (req.method === 'GET' && url.pathname === '/ping') {
      if (!ready) {
        res.writeHead(204);
        res.end();
        return;
      }
      sendJson(res, 200, { browserPid: getBrowserPid() });
      return;
    }

    if (req.method === 'POST' && url.pathname === '/render/grid') {
      if (!ready) {
        sendJson(res, 503, { error: 'renderer not ready' });
        return;
      }
      await handleRenderGrid(req, res);
      return;
    }

    if (req.method === 'POST' && url.pathname === '/render/views') {
      if (!ready) {
        sendJson(res, 503, { error: 'renderer not ready' });
        return;
      }
      await handleRenderViews(req, res);
      return;
    }

    sendJson(res, 404, { error: 'not found' });
  } catch (err) {
    console.error('[render_runner] request error:', err);
    try {
      sendJson(res, 500, { error: err.message || 'internal error' });
    } catch {}
  }
});

// ---- begin r42_keepalive block ----
// [r42_keepalive] PGH_RENDER_KEEPALIVE_MS: idle keep-alive time of this server. Unset / "" / "0" / anything that is
// not plain digits ("off", "-5", "65e3", "65000ms") = Node defaults untouched (keepAliveTimeout 5000 + Node's 1000 ms
// buffer, headersTimeout 60000). The Python client (httpx, keepalive_expiry 5 s counted from when IT finished reading
// the response) then reuses sockets this server already closed whenever its event loop lags > ~1 s -> RST ->
// ReadError -> retry / dropped draft. A value well above 5000 (Dockerfile: 65000) means the client always gives a
// socket up first. Clamped to [5000, 290000]: never shorter than the default, headersTimeout stays < requestTimeout.
const KEEPALIVE_RAW = (process.env.PGH_RENDER_KEEPALIVE_MS || '').trim();
const KEEPALIVE_MS = /^[0-9]+$/.test(KEEPALIVE_RAW) ? parseInt(KEEPALIVE_RAW, 10) : 0;
if (KEEPALIVE_MS > 0) {
  server.keepAliveTimeout = Math.min(290000, Math.max(5000, KEEPALIVE_MS));
  server.headersTimeout = Math.max(server.headersTimeout, server.keepAliveTimeout + 1000);
  console.log(`[render_runner] keep-alive: keepAliveTimeout=${server.keepAliveTimeout}ms headersTimeout=${server.headersTimeout}ms (PGH_RENDER_KEEPALIVE_MS=${KEEPALIVE_RAW}, node ${process.version})`);
}
// ---- end r42_keepalive block ----

async function gracefulShutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log(`[render_runner] ${signal} received, shutting down...`);
  server.close();
  try {
    await closeBrowser();
  } catch (err) {
    console.error('[render_runner] browser close error:', err.message);
  }
  process.exit(0);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

process.on('unhandledRejection', (err) => {
  console.error('[render_runner] unhandled rejection:', err);
  process.exit(1);
});

async function main() {
  console.log('[render_runner] starting static server...');
  await startStaticServer();

  server.listen(PORT, '127.0.0.1', () => {
    console.log(`[render_runner] listening on 127.0.0.1:${PORT}`);
  });

  console.log('[render_runner] launching browser...');
  await ensureBrowser();

  // Warm the object pool before reporting ready, so no request pays the
  // ~3s page + PMREM cost. Unset PREWARM_LIGHTING = old lazy behavior.
  const prewarmLighting = process.env.PREWARM_LIGHTING;
  if (prewarmLighting) {
    if (LIGHTING_MODES.has(prewarmLighting)) {
      console.log(`[render_runner] prewarming object pool (lighting=${prewarmLighting})...`);
      const t0 = Date.now();
      await objectPool.prewarm(prewarmLighting);
      console.log(`[render_runner] prewarm done in ${Date.now() - t0}ms`);
    } else {
      console.error(`[render_runner] unknown PREWARM_LIGHTING '${prewarmLighting}', skipping prewarm`);
    }
  }

  ready = true;
  console.log('[render_runner] ready');
}

main().catch((err) => {
  console.error('[render_runner] fatal startup error:', err);
  process.exit(1);
});
