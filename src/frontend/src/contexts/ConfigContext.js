import React, { createContext, useContext, useState, useEffect } from 'react';

const ConfigContext = createContext({ config: null, serviceEndpoint: '' });

export function ConfigProvider({ children }) {
  const [config, setConfig] = useState(null);
  const serviceEndpoint = process.env.REACT_APP_SERVICE_ENDPOINT || 'http://localhost:5005';

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
