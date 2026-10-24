/**
 * Browser lifecycle management.
 *
 * Launches headless Chromium once on startup, keeps it alive across requests.
 * The render pool manages per-context recovery; restartBrowser() is only
 * called when the browser process itself is unhealthy.
 */

import puppeteer from 'puppeteer';

let browser = null;
// Lets the disconnected handler tell a chromium crash from a deliberate close.
let intentionalClose = false;

async function closeQuietly(b) {
  intentionalClose = true;
  try { await b.close(); } catch {}
  intentionalClose = false;
}

const BASE_ARGS = [
  '--no-sandbox',
  '--disable-setuid-sandbox',
  '--disable-dev-shm-usage',
  '--disable-background-timer-throttling',
  '--disable-backgrounding-occluded-windows',
  '--disable-renderer-backgrounding',
  '--enable-webgl',
  '--ignore-gpu-blocklist',
];

const GPU_ARGS = process.env.USE_GL === 'egl'
  ? ['--use-gl=egl', '--enable-gpu', '--disable-gpu-sandbox']
  : ['--use-gl=angle', '--use-angle=swiftshader', '--in-process-gpu'];

export function isBrowserHealthy() {
  return browser != null && browser.connected;
}

export function getBrowserPid() {
  return browser?.process()?.pid ?? null;
}

export async function ensureBrowser() {
  if (browser && browser.connected) return browser;
  if (browser) {
    await closeQuietly(browser);
  }
  const protocolTimeout = parseInt(process.env.PROTOCOL_TIMEOUT_MS || '60000', 10);
  browser = await puppeteer.launch({
    headless: 'new',
    args: [...BASE_ARGS, ...GPU_ARGS],
    protocolTimeout,
  });
  browser.on('disconnected', () => {
    if (intentionalClose) return;
    console.error('[browser] chromium disconnected unexpectedly, exiting');
    process.exit(1);
  });
  return browser;
}

export async function restartBrowser() {
  if (browser) {
    await closeQuietly(browser);
  }
  browser = null;
  return ensureBrowser();
}

export async function closeBrowser() {
  if (browser) {
    await closeQuietly(browser);
    browser = null;
  }
}
