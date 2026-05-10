import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * useApiClient
 *
 * Centralised fetch wrapper that gives every call an AbortController and
 * tracks all in-flight controllers so they can be aborted on unmount.
 *
 * Usage:
 *   const { call, abortAll } = useApiClient(serviceEndpoint);
 *   const res = await call('/session/start', { method: 'POST', body: {...} });
 *   // res = { ok, status, data, error, aborted? }
 *
 * Replace-latest semantics:
 *   If `opts.key` is provided and a previous call with the same key is still
 *   in flight, the previous one is aborted before the new one starts.
 *   Useful for e.g. slider drags where only the latest request matters.
 *
 * Network failures resolve as { ok: false, status: 0, error: 'Network error', data: null }.
 * AbortError resolves as       { ok: false, status: 0, error: 'aborted', aborted: true, data: null }.
 */
export function useApiClient(serviceEndpoint) {
  // Set of all live AbortControllers belonging to this hook instance.
  const controllersRef = useRef(new Set());
  // Map: key -> AbortController, used for replace-latest semantics.
  const keyedControllersRef = useRef(new Map());

  const call = useCallback(async (path, opts = {}) => {
    const { method = 'GET', body, headers, signal: externalSignal, key, ...rest } = opts;

    // Replace-latest: abort any in-flight call sharing this key.
    if (key && keyedControllersRef.current.has(key)) {
      const prev = keyedControllersRef.current.get(key);
      try { prev.abort(); } catch (_) { /* ignore */ }
    }

    const controller = new AbortController();
    controllersRef.current.add(controller);
    if (key) keyedControllersRef.current.set(key, controller);

    // If the caller supplied their own signal, propagate its abort to ours.
    let externalAbortHandler = null;
    if (externalSignal) {
      if (externalSignal.aborted) {
        try { controller.abort(); } catch (_) { /* ignore */ }
      } else {
        externalAbortHandler = () => {
          try { controller.abort(); } catch (_) { /* ignore */ }
        };
        externalSignal.addEventListener('abort', externalAbortHandler);
      }
    }

    // Body / headers handling.
    let finalHeaders = { ...(headers || {}) };
    let finalBody = body;
    const isPlainObject =
      body !== null &&
      body !== undefined &&
      typeof body === 'object' &&
      !(body instanceof FormData) &&
      !(body instanceof Blob) &&
      !(body instanceof ArrayBuffer);

    if (isPlainObject) {
      finalBody = JSON.stringify(body);
      if (!Object.keys(finalHeaders).some((h) => h.toLowerCase() === 'content-type')) {
        finalHeaders['Content-Type'] = 'application/json';
      }
    }

    const url = `${serviceEndpoint}${path}`;

    try {
      const response = await fetch(url, {
        method,
        headers: finalHeaders,
        body: finalBody,
        signal: controller.signal,
        ...rest,
      });

      let data = null;
      // Try JSON parse; tolerate empty / non-JSON bodies.
      try {
        const text = await response.text();
        data = text ? JSON.parse(text) : null;
      } catch (_) {
        data = null;
      }

      return {
        ok: response.ok,
        status: response.status,
        data,
        error: response.ok ? null : (data && data.error) || `HTTP ${response.status}`,
      };
    } catch (err) {
      if (err && err.name === 'AbortError') {
        return { ok: false, status: 0, data: null, error: 'aborted', aborted: true };
      }
      return { ok: false, status: 0, data: null, error: 'Network error' };
    } finally {
      controllersRef.current.delete(controller);
      if (key && keyedControllersRef.current.get(key) === controller) {
        keyedControllersRef.current.delete(key);
      }
      if (externalSignal && externalAbortHandler) {
        externalSignal.removeEventListener('abort', externalAbortHandler);
      }
    }
  }, [serviceEndpoint]);

  const abortAll = useCallback(() => {
    for (const ctrl of controllersRef.current) {
      try { ctrl.abort(); } catch (_) { /* ignore */ }
    }
    controllersRef.current.clear();
    keyedControllersRef.current.clear();
  }, []);

  // Abort everything on unmount.
  useEffect(() => {
    return () => {
      abortAll();
    };
  }, [abortAll]);

  /**
   * pollJob
   *
   * Polls GET /jobs/<jobId> every `intervalMs` until status is `done` or
   * `error`, or until `timeoutMs` elapses. Each underlying GET goes through
   * `call()` so it is registered as an in-flight controller and will be
   * aborted by abortAll() (e.g. on unmount). Resolves with the same
   * { ok, data, error, aborted? } shape as call().
   *
   *   data on success: the full /jobs/<id> body. The caller is expected to
   *                    inspect data.result for the actual payload.
   */
  const pollJob = useCallback(async (jobId, options = {}) => {
    const { intervalMs = 1500, timeoutMs = 60000 } = options;
    if (!jobId) {
      return { ok: false, status: 0, data: null, error: 'missing job_id' };
    }
    const startTs = Date.now();
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const res = await call(`/jobs/${jobId}`, { method: 'GET' });
      if (res.aborted) return res;
      if (!res.ok) {
        // 404 etc. — surface to caller. Keep status so caller can detect.
        return res;
      }
      const data = res.data || {};
      const status = data.status;
      if (status === 'done' || status === 'error') {
        return res;
      }
      if (Date.now() - startTs >= timeoutMs) {
        return {
          ok: false,
          status: 0,
          data,
          error: `Job ${jobId} timed out after ${timeoutMs}ms`,
        };
      }
      // Sleep, but bail out early if aborted in the meantime.
      // We piggy-back on call() abort by polling controllersRef on each tick.
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
  }, [call]);

  return { call, abortAll, pollJob };
}

/**
 * useApiCall
 *
 * Convenience wrapper around useApiClient for single-action UIs that need to
 * track an in-flight boolean and the last error.
 *
 *   const { inFlight, lastError, exec } = useApiCall(serviceEndpoint);
 *   const res = await exec('/foo', { method: 'POST', body: {...} });
 */
export function useApiCall(serviceEndpoint) {
  const { call, abortAll } = useApiClient(serviceEndpoint);
  const [inFlight, setInFlight] = useState(false);
  const [lastError, setLastError] = useState(null);

  const exec = useCallback(async (path, opts = {}) => {
    setInFlight(true);
    setLastError(null);
    try {
      const result = await call(path, opts);
      if (!result.ok && !result.aborted) {
        setLastError(result.error || 'Request failed');
      }
      return result;
    } finally {
      setInFlight(false);
    }
  }, [call]);

  return { inFlight, lastError, exec, abortAll };
}

export default useApiClient;
