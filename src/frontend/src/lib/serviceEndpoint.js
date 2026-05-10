const DEFAULT_BACKEND_PORT = '5005';

function stripTrailingSlash(value) {
  return String(value || '').replace(/\/+$/, '');
}

function isLoopbackHost(hostname) {
  const normalized = String(hostname || '').toLowerCase();
  return (
    normalized === 'localhost' ||
    normalized === '127.0.0.1' ||
    normalized === '::1' ||
    normalized === '[::1]'
  );
}

function hostsMatch(left, right) {
  return String(left || '').toLowerCase() === String(right || '').toLowerCase();
}

export function resolveServiceEndpoint(configuredEndpoint = process.env.REACT_APP_SERVICE_ENDPOINT) {
  const configured = stripTrailingSlash(configuredEndpoint);

  if (typeof window === 'undefined' || !window.location) {
    return configured || `http://localhost:${DEFAULT_BACKEND_PORT}`;
  }

  const pageHost = window.location.hostname;
  const pageProtocol = window.location.protocol === 'https:' ? 'https:' : 'http:';

  if (!configured) {
    return `${pageProtocol}//${pageHost}:${DEFAULT_BACKEND_PORT}`;
  }

  try {
    const parsed = new URL(configured);
    if (isLoopbackHost(parsed.hostname) && !hostsMatch(parsed.hostname, pageHost)) {
      return `${pageProtocol}//${pageHost}:${parsed.port || DEFAULT_BACKEND_PORT}`;
    }
  } catch {
    return configured;
  }

  return configured;
}
