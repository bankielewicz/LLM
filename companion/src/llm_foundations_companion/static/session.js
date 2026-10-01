const CONNECT_PREFIX = "#/connect/";
const ACTIVE_INSTANCE_KEY = "llm-foundations-companion-active-instance";
const SESSION_KEY_PREFIX = "llm-foundations-companion-session:";
const TOKEN_PATTERN = /^[A-Za-z0-9_-]{43}$/;
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const IDLE_MILLISECONDS = 2 * 60 * 60 * 1000;

const initialFragment = window.location.hash;
let pairingSecret = initialFragment.startsWith(CONNECT_PREFIX)
  ? initialFragment.slice(CONNECT_PREFIX.length)
  : null;

// The fragment is removed synchronously, before this module initiates a fetch
// or imports the preserved reader. It is never copied into the DOM or a log.
if (pairingSecret !== null) {
  history.replaceState(null, "", window.location.pathname + window.location.search);
}

const statusText = document.getElementById("companion-status-text");
const retryButton = document.getElementById("companion-retry");
const disconnectButton = document.getElementById("companion-disconnect");
let session = null;
let refreshTimer = null;
let noSessionMessage = "Pair this tab";

const STATUS_LABELS = new Set([
  "Pair this tab",
  "Checking local runtime",
  "Local runtime connected",
  "Connected · storage read-only",
  "Incompatible local runtime",
  "Session expired · pair again",
  "Connection lost",
]);

function sessionStore() {
  try {
    const probe = "llmf-session-storage-probe";
    window.sessionStorage.setItem(probe, "1");
    window.sessionStorage.removeItem(probe);
    return window.sessionStorage;
  } catch {
    return null;
  }
}

const storage = sessionStore();

function setStatus(text, {canRetry = false, canDisconnect = false} = {}) {
  if (!STATUS_LABELS.has(text)) text = "Connection lost";
  statusText.textContent = text;
  retryButton.hidden = !canRetry;
  disconnectButton.hidden = !canDisconnect;
}

function isSessionRecord(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const keys = Object.keys(value).sort();
  const expected = [
    "absolute_expires_at",
    "access_token",
    "csrf_token",
    "idle_expires_at",
    "instance_id",
  ];
  return keys.length === expected.length
    && keys.every((key, index) => key === expected[index])
    && TOKEN_PATTERN.test(value.access_token)
    && TOKEN_PATTERN.test(value.csrf_token)
    && UUID_PATTERN.test(value.instance_id)
    && Number.isFinite(Date.parse(value.idle_expires_at))
    && Number.isFinite(Date.parse(value.absolute_expires_at));
}

function clearStoredSessions() {
  session = null;
  if (!storage) return;
  const removals = [];
  for (let index = 0; index < storage.length; index += 1) {
    const key = storage.key(index);
    if (key === ACTIVE_INSTANCE_KEY || key?.startsWith(SESSION_KEY_PREFIX)) {
      removals.push(key);
    }
  }
  for (const key of removals) storage.removeItem(key);
}

function saveSession(value) {
  if (!isSessionRecord(value)) throw new Error("The companion returned an invalid session.");
  clearStoredSessions();
  session = {...value};
  if (storage) {
    storage.setItem(SESSION_KEY_PREFIX + session.instance_id, JSON.stringify(session));
    storage.setItem(ACTIVE_INSTANCE_KEY, session.instance_id);
  }
}

function loadSession() {
  if (!storage) return null;
  try {
    const instanceId = storage.getItem(ACTIVE_INSTANCE_KEY);
    if (!instanceId || !UUID_PATTERN.test(instanceId)) return null;
    const value = JSON.parse(storage.getItem(SESSION_KEY_PREFIX + instanceId));
    if (!isSessionRecord(value) || value.instance_id !== instanceId) return null;
    if (Date.now() >= Date.parse(value.idle_expires_at)
        || Date.now() >= Date.parse(value.absolute_expires_at)) {
      clearStoredSessions();
      return null;
    }
    return value;
  } catch {
    clearStoredSessions();
    return null;
  }
}

async function timedFetch(resource, options, timeoutMilliseconds = 10000) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMilliseconds);
  try {
    return await fetch(resource, {...options, signal: controller.signal});
  } finally {
    window.clearTimeout(timeout);
  }
}

async function exchangePairingSecret(secret) {
  if (!TOKEN_PATTERN.test(secret)) {
    throw new Error("Pair this tab");
  }
  const response = await timedFetch("/api/v1/sessions", {
    method: "POST",
    cache: "no-store",
    credentials: "omit",
    headers: {"Content-Type": "application/json; charset=utf-8"},
    body: JSON.stringify({bootstrap_secret: secret}),
  });
  let value = null;
  try {
    value = await response.json();
  } catch {
    throw new Error("Connection lost");
  }
  if (!response.ok || !isSessionRecord(value)) {
    throw new Error("Session expired · pair again");
  }
  saveSession(value);
}

function extendLocalIdleExpiry(runtime) {
  const serverTime = Date.parse(runtime.server_time);
  const absolute = Date.parse(session.absolute_expires_at);
  if (!Number.isFinite(serverTime) || !Number.isFinite(absolute)) return;
  session.idle_expires_at = new Date(Math.min(serverTime + IDLE_MILLISECONDS, absolute)).toISOString();
  if (storage) {
    storage.setItem(SESSION_KEY_PREFIX + session.instance_id, JSON.stringify(session));
  }
}

async function refreshRuntime() {
  if (!session) {
    setStatus(noSessionMessage);
    return;
  }
  if (Date.now() >= Date.parse(session.absolute_expires_at)
      || Date.now() >= Date.parse(session.idle_expires_at)) {
    clearStoredSessions();
    noSessionMessage = "Session expired · pair again";
    setStatus(noSessionMessage);
    return;
  }
  try {
    const response = await timedFetch("/api/v1/runtime", {
      method: "GET",
      cache: "no-store",
      credentials: "omit",
      headers: {Authorization: "Bearer " + session.access_token},
    });
    if (response.status === 401 || response.status === 403) {
      clearStoredSessions();
      noSessionMessage = "Session expired · pair again";
      setStatus(noSessionMessage);
      return;
    }
    if (!response.ok) throw new Error("Runtime status is unavailable.");
    const runtime = await response.json();
    if (!runtime || runtime.instance_id !== session.instance_id) {
      clearStoredSessions();
      noSessionMessage = "Session expired · pair again";
      setStatus(noSessionMessage);
      return;
    }
    if (runtime.mode !== "local"
        || runtime.api_version !== "1.0"
        || typeof runtime.storage_writable !== "boolean") {
      setStatus("Incompatible local runtime", {canDisconnect: true});
      return;
    }
    extendLocalIdleExpiry(runtime);
    setStatus(
      runtime.storage_writable
        ? "Local runtime connected"
        : "Connected · storage read-only",
      {canDisconnect: true},
    );
  } catch {
    setStatus("Connection lost", {canRetry: true, canDisconnect: true});
  }
}

async function disconnect() {
  const current = session;
  disconnectButton.disabled = true;
  try {
    if (current) {
      await timedFetch("/api/v1/sessions/current", {
        method: "DELETE",
        cache: "no-store",
        credentials: "omit",
        headers: {
          Authorization: "Bearer " + current.access_token,
          "X-LLMF-CSRF": current.csrf_token,
          "Idempotency-Key": crypto.randomUUID(),
        },
      });
    }
  } catch {
    // Local credentials are still cleared when the service is unavailable.
  } finally {
    clearStoredSessions();
    disconnectButton.disabled = false;
    noSessionMessage = "Pair this tab";
    setStatus(noSessionMessage);
  }
}

async function initializeSession() {
  if (pairingSecret !== null) {
    setStatus("Checking local runtime");
    try {
      await exchangePairingSecret(pairingSecret);
    } catch (error) {
      clearStoredSessions();
      noSessionMessage = STATUS_LABELS.has(error.message)
        ? error.message
        : "Connection lost";
      setStatus(noSessionMessage);
    } finally {
      pairingSecret = null;
    }
  } else {
    session = loadSession();
  }
  await refreshRuntime();
  refreshTimer = window.setInterval(refreshRuntime, 60000);
}

retryButton.addEventListener("click", refreshRuntime);
disconnectButton.addEventListener("click", disconnect);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") refreshRuntime();
});
window.addEventListener("pagehide", () => {
  if (refreshTimer !== null) window.clearInterval(refreshTimer);
}, {once: true});

await initializeSession();
try {
  await import("./app.js");
} catch {
  const target = document.getElementById("app");
  const message = document.createElement("p");
  message.setAttribute("role", "alert");
  message.textContent = "The preserved course reader could not be opened.";
  target.replaceChildren(message);
}
