import React, { createContext, useContext, useState, useEffect, useMemo } from 'react';
import { resolveServiceEndpoint } from '../lib/serviceEndpoint';

const ConfigContext = createContext({ config: null, serviceEndpoint: '' });

export function ConfigProvider({ children }) {
  const [config, setConfig] = useState(null);
  const serviceEndpoint = useMemo(() => resolveServiceEndpoint(), []);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${serviceEndpoint}/config`, { signal: controller.signal })
      .then(r => r.json())
      .then(setConfig)
      .catch(err => { if (err.name !== 'AbortError') setConfig({}); });
    return () => controller.abort();
  }, [serviceEndpoint]);

  return (
    <ConfigContext.Provider value={{ config, serviceEndpoint }}>
      {children}
    </ConfigContext.Provider>
  );
}

export const useConfig = () => useContext(ConfigContext);
