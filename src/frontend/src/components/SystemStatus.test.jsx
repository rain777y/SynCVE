/**
 * Tests for the SystemStatus component.
 */

import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import SystemStatus from './SystemStatus';

describe('SystemStatus', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    global.fetch = jest.fn();
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.restoreAllMocks();
  });

  test('renders system status heading', () => {
    global.fetch.mockResolvedValue({ ok: false });
    render(<SystemStatus />);
    expect(screen.getByText('System Status')).toBeInTheDocument();
  });

  test('renders all status items', () => {
    global.fetch.mockResolvedValue({ ok: false });
    render(<SystemStatus />);
    expect(screen.getByText('Face Detector')).toBeInTheDocument();
    expect(screen.getByText('Anti-Spoofing')).toBeInTheDocument();
    expect(screen.getByText('Detection Interval')).toBeInTheDocument();
    expect(screen.getByText('Image Quality')).toBeInTheDocument();
    expect(screen.getByText('Video Resolution')).toBeInTheDocument();
    expect(screen.getByText('GPU Acceleration')).toBeInTheDocument();
  });

  test('shows checking state initially', () => {
    global.fetch.mockImplementation(() => new Promise(() => {})); // never resolves
    render(<SystemStatus />);
    expect(screen.getByText('Checking...')).toBeInTheDocument();
  });

  test('shows online when backend responds OK', async () => {
    global.fetch.mockResolvedValue({ ok: true });
    await act(async () => {
      render(<SystemStatus />);
    });

    await waitFor(() => {
      expect(screen.getByText('Online')).toBeInTheDocument();
    });
  });

  test('shows offline when backend responds with error', async () => {
    global.fetch.mockResolvedValue({ ok: false });
    await act(async () => {
      render(<SystemStatus />);
    });

    await waitFor(() => {
      expect(screen.getByText('Offline')).toBeInTheDocument();
    });
  });

  test('shows offline when fetch throws (network error)', async () => {
    global.fetch.mockRejectedValue(new Error('Network error'));
    await act(async () => {
      render(<SystemStatus />);
    });

    await waitFor(() => {
      expect(screen.getByText('Offline')).toBeInTheDocument();
    });
  });

  test('displays image quality percentage', () => {
    global.fetch.mockResolvedValue({ ok: false });
    render(<SystemStatus />);
    // 0.95 * 100 = 95
    expect(screen.getByText('95%')).toBeInTheDocument();
  });

  test('displays video resolution', () => {
    global.fetch.mockResolvedValue({ ok: false });
    render(<SystemStatus />);
    expect(screen.getByText('1280x720')).toBeInTheDocument();
  });

  test('displays detection interval in ms', () => {
    global.fetch.mockResolvedValue({ ok: false });
    render(<SystemStatus />);
    expect(screen.getByText('1500ms')).toBeInTheDocument();
  });
});
