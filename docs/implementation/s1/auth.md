# S1 authentication and local-control boundary

Status: implemented for the S1 candidate. Focused WSL development tests pass;
the reusable native-Windows probe remains a separate platform-matrix result.
Product acceptance remains `NOT_RUN`.

## Implemented boundary

`AuthManager` owns instance-local credentials in memory. Each `pair` replaces
the one outstanding bootstrap secret. A bootstrap secret is CSPRNG-generated,
expires after 120 seconds, and is consumed only by a successful exchange. The
exchange creates independent 256-bit access and CSRF tokens plus an internal
random `session_id`; callers use the session ID as the stable subject and do not
persist a bearer token. The manager enforces 32 unexpired sessions, two-hour
idle expiry, 12-hour absolute expiry, immediate revocation, and restart
invalidation. Credential comparisons use `hmac.compare_digest`, and credential
dataclass representations omit secret values.

`AuthManager.authorize` fixes the HTTP security order through CSRF:

1. exact `Host: 127.0.0.1:8765`;
2. reject `OPTIONS`;
3. exact required mutation/session-exchange Origin, or exact Origin when one is
   present on an authenticated read;
4. `GET` requires `Sec-Fetch-Site: same-origin` or `none`;
5. bearer lookup and idle/absolute expiry;
6. matching mutation `X-LLMF-CSRF`.

The HTTP facade owns the later content-type, declared/streamed-size, strict JSON,
schema, reference, digest, semantic, and eligibility stages. Session exchange
stops after the Origin stage because it has no bearer session yet.

## Private control channel

`ControlServer` publishes canonical one-line `runtime/control.json` with a
current-user-only control secret. On WSL/Linux it serves a real Unix-domain
socket at `runtime/control.sock`, mode 0600, inside a mode-0700 runtime
directory. On Windows it uses a native `\\.\pipe\llm-foundations-<instance_id>`
named pipe created with a protected DACL granting the current user, SYSTEM, and
built-in Administrators. The pipe rejects remote clients. There is no TCP
fallback.

A WSL root whose fixed `runtime/control.sock` exceeds the Unix-socket pathname
limit fails with `STORAGE_UNAVAILABLE` and asks the learner to choose a shorter
root. The service does not substitute another transport or endpoint.

Each connection accepts exactly one canonical UTF-8 JSON request line and
returns one canonical JSON response line, with a 1 MiB limit in each direction.
The request has exactly `control_secret`, `operation`, and `arguments`. The
closed operation set is `pair`, `status`, `submit_job`, `cancel_job`,
`list_events`, `register_e01_receipt`, and `register_reuse_receipt`. S1
orchestration supplies callbacks; a named operation whose later slice is absent
returns `CAPABILITY_UNAVAILABLE` rather than synthetic success. Unknown
operations and multiple/malformed lines return `INVALID_REQUEST`; wrong secrets
return `AUTH_REQUIRED`; oversize lines return `PAYLOAD_TOO_LARGE`. The client
also rejects a response whose error object is outside the sealed vocabulary or
whose status/retryability/request ID binding is inconsistent.

`ControlClient.verify_instance` compares the status response's `instance_id`
and PID with the private discovery file. `pair` performs the same check before
requesting a bootstrap URL. Stop is idempotent before start, after a partial
prepare failure, and after normal service shutdown; it removes only the
descriptor and socket identity owned by that server instance.

## Verification boundary

Focused WSL tests cover bootstrap replacement/reuse/expiry, session limits and
lifetimes, ordered request failures, revocation/restart, actual Unix socket
permissions and exchanges, missing delegated capabilities, malformed/noncanonical
requests, multiple and oversize lines, and secret-free representations and
responses, plus explicit rejection of an overlong fixed Unix-socket path.
`companion/tests/native_windows_control_probe.py` is a non-pytest
probe for the qualified native interpreter. It exercises the actual named pipe,
pair/status, the private discovery-file ACL, absence of a TCP fallback, normal
shutdown cleanup, and cleanup after a partial prepare failure while emitting
credential-free structured JSON. Its recorded result belongs to the platform
matrix and is not inferred from WSL. Second-OS-user denial, browser behavior,
the complete HTTP order through body parsing, installation profiles, and every
sealed AC-RUN execution remain product acceptance work and are not established
by these development tests.
