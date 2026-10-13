/**
 * AsrProbe — polls a speech-to-text server (Whisper on vLLM, default port 9000).
 *
 * Focus: requests, queue, latency — not host RAM/VRAM (the GPU panel has those).
 *
 * Endpoints:
 *   GET /v1/models   served model ids + root path (LoRA / checkpoint dir)
 *   GET /metrics     vLLM queue + e2e latency histogram, and the HTTP
 *                    instrumentator's per-handler request counts and body sizes
 *
 * Counters are cumulative since the server started, so per-minute rates and
 * recent latency come from deltas between samples kept here in memory.
 */
import { LLM_PROBE_TIMEOUT_MS, ASR_PORT } from "../config.js";
import { llmProbeHost } from "./llmHost.js";

/** How long samples are kept for windowed rates (ms). */
const WINDOW_MS = 15 * 60_000;
/** Rate window for requests/min (ms). */
const RATE_WINDOW_MS = 60_000;
/** Window for "recent" latency percentiles (ms). */
const RECENT_WINDOW_MS = 5 * 60_000;

/**
 * Parse Prometheus text exposition into [{ name, labels, value }].
 * @param {string} text
 */
export function parsePrometheus(text) {
  /** @type {{ name: string, labels: Record<string, string>, value: number }[]} */
  const out = [];
  for (const line of String(text).split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const m = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)/);
    if (!m) continue;
    const value = Number(m[3]);
    if (!Number.isFinite(value) && m[3] !== "+Inf") continue;
    /** @type {Record<string, string>} */
    const labels = {};
    if (m[2]) {
      for (const lm of m[2].matchAll(/([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"/g)) {
        labels[lm[1]] = lm[2];
      }
    }
    out.push({ name: m[1], labels, value });
  }
  return out;
}

/**
 * Quantile from cumulative histogram buckets (Prometheus-style linear
 * interpolation inside the bucket).
 * @param {{ le: number, count: number }[]} buckets cumulative, any order
 * @param {number} q 0–1
 * @returns {number | null}
 */
export function histogramQuantile(buckets, q) {
  const sorted = [...buckets].sort((a, b) => a.le - b.le);
  const total = sorted.length ? sorted[sorted.length - 1].count : 0;
  if (!(total > 0)) return null;
  const rank = q * total;
  let prevLe = 0;
  let prevCount = 0;
  for (const b of sorted) {
    if (b.count >= rank) {
      // Rank fell in the +Inf bucket: the best we can say is "above the last bound".
      if (!Number.isFinite(b.le)) return prevLe;
      const inBucket = b.count - prevCount;
      if (inBucket <= 0) return b.le;
      return prevLe + (b.le - prevLe) * ((rank - prevCount) / inBucket);
    }
    prevLe = Number.isFinite(b.le) ? b.le : prevLe;
    prevCount = b.count;
  }
  return prevLe;
}

/**
 * Pull the counters this probe tracks out of parsed samples.
 * @param {ReturnType<typeof parsePrometheus>} samples
 */
export function extractAsrCounters(samples) {
  let running = 0;
  let waiting = 0;
  let ok = 0;
  let errors = 0;
  let audioBytes = 0;
  let e2eSum = 0;
  let e2eCount = 0;
  let startedAtSec = null;
  /** @type {Map<number, number>} */
  const bucketMap = new Map();
  let sawVllm = false;

  for (const { name, labels, value } of samples) {
    const isAudio = typeof labels.handler === "string" && labels.handler.startsWith("/v1/audio/");
    switch (name) {
      case "vllm:num_requests_running":
        running += value;
        sawVllm = true;
        break;
      case "vllm:num_requests_waiting":
        waiting += value;
        break;
      case "vllm:e2e_request_latency_seconds_bucket": {
        const le = labels.le === "+Inf" ? Infinity : Number(labels.le);
        if (!Number.isNaN(le)) bucketMap.set(le, (bucketMap.get(le) || 0) + value);
        break;
      }
      case "vllm:e2e_request_latency_seconds_sum":
        e2eSum += value;
        break;
      case "vllm:e2e_request_latency_seconds_count":
        e2eCount += value;
        break;
      case "http_requests_total":
        if (isAudio) {
          if (String(labels.status || "").startsWith("2")) ok += value;
          else errors += value;
        }
        break;
      case "http_request_size_bytes_sum":
        if (isAudio) audioBytes += value;
        break;
      case "process_start_time_seconds":
        startedAtSec = value;
        break;
    }
  }

  return {
    sawVllm,
    running,
    waiting,
    ok,
    errors,
    audioBytes,
    e2eSum,
    e2eCount,
    buckets: [...bucketMap.entries()].map(([le, count]) => ({ le, count })),
    startedAt: startedAtSec != null ? Math.round(startedAtSec * 1000) : null,
  };
}

/** @param {string} p */
function basename(p) {
  const s = String(p).replace(/\/+$/, "");
  const i = s.lastIndexOf("/");
  return i >= 0 ? s.slice(i + 1) : s;
}

/** @param {number | null} v */
function round3(v) {
  return v == null || !Number.isFinite(v) ? null : Math.round(v * 1000) / 1000;
}

export class AsrProbe {
  /**
   * @param {object} spark
   * @param {number} [port]
   */
  constructor(spark, port = ASR_PORT) {
    this.spark = spark;
    this.port = Number.isInteger(port) && port >= 1 && port <= 65535 ? port : ASR_PORT;
    this.baseUrl = `http://${llmProbeHost(spark)}:${this.port}`;
    this.error = null;
    /** @type {{ t: number, c: ReturnType<typeof extractAsrCounters> }[]} */
    this._samples = [];
    /** @type {number | null} */
    this._lastRequestAt = null;
  }

  setTarget(spark, port) {
    this.spark = spark ?? this.spark;
    const next = Number(port);
    if (Number.isInteger(next) && next >= 1 && next <= 65535) {
      this.port = next;
    }
    this.baseUrl = `http://${llmProbeHost(this.spark)}:${this.port}`;
    this._samples = [];
    this._lastRequestAt = null;
  }

  dispose() {
    this._samples = [];
  }

  /**
   * @returns {Promise<object>}
   */
  async probe() {
    try {
      const signal = AbortSignal.timeout(LLM_PROBE_TIMEOUT_MS);
      const [modelsRes, metricsRes] = await Promise.all([
        fetch(`${this.baseUrl}/v1/models`, { signal }),
        fetch(`${this.baseUrl}/metrics`, { signal }),
      ]);
      if (!modelsRes.ok) {
        this.error = `HTTP ${modelsRes.status} on /v1/models`;
        return this._default();
      }
      const modelsJson = await modelsRes.json().catch(() => null);
      const data = Array.isArray(modelsJson?.data) ? modelsJson.data : [];
      const models = data.map((m) => String(m?.id ?? "")).filter(Boolean);
      const root = data.find((m) => m?.root)?.root;

      const c = metricsRes.ok
        ? extractAsrCounters(parsePrometheus(await metricsRes.text()))
        : null;
      this.error = null;

      const now = Date.now();
      const base = {
        available: true,
        error: metricsRes.ok ? null : `HTTP ${metricsRes.status} on /metrics`,
        port: this.port,
        engine: c?.sawVllm || data.some((m) => m?.owned_by === "vllm") ? "vLLM" : null,
        models,
        modelRoot: root ? basename(root) : null,
        maxModelLen: Number(data[0]?.max_model_len) || null,
      };
      if (!c) return { ...base, ...this._emptyStats() };

      // Counter went backwards → server restarted; history no longer comparable.
      const prev = this._samples[this._samples.length - 1];
      if (prev && (c.ok + c.errors < prev.c.ok + prev.c.errors || c.e2eCount < prev.c.e2eCount)) {
        this._samples = [];
        this._lastRequestAt = null;
      }
      if (prev && c.ok + c.errors > prev.c.ok + prev.c.errors) this._lastRequestAt = now;
      this._samples.push({ t: now, c });
      while (this._samples.length > 1 && now - this._samples[0].t > WINDOW_MS) {
        this._samples.shift();
      }

      const rateRef = this._sampleAtOrBefore(now - RATE_WINDOW_MS);
      const recentRef = this._sampleAtOrBefore(now - RECENT_WINDOW_MS);

      /** @type {number | null} */
      let requestsPerMin = null;
      if (rateRef && now - rateRef.t >= 5_000) {
        const d = c.ok + c.errors - (rateRef.c.ok + rateRef.c.errors);
        requestsPerMin = Math.max(0, (d / (now - rateRef.t)) * 60_000);
      }

      // Recent latency: bucket deltas against the reference sample.
      let recent = { count: 0, avg: null, p50: null, p95: null, windowSec: 0 };
      if (recentRef && recentRef !== this._samples[this._samples.length - 1]) {
        const prevBuckets = new Map(recentRef.c.buckets.map((b) => [b.le, b.count]));
        const deltaBuckets = c.buckets.map((b) => ({
          le: b.le,
          count: Math.max(0, b.count - (prevBuckets.get(b.le) || 0)),
        }));
        const count = c.e2eCount - recentRef.c.e2eCount;
        recent = {
          count,
          avg: count > 0 ? round3((c.e2eSum - recentRef.c.e2eSum) / count) : null,
          p50: count > 0 ? round3(histogramQuantile(deltaBuckets, 0.5)) : null,
          p95: count > 0 ? round3(histogramQuantile(deltaBuckets, 0.95)) : null,
          windowSec: Math.round((now - recentRef.t) / 1000),
        };
      }

      const recentBytes = recentRef ? c.audioBytes - recentRef.c.audioBytes : 0;

      return {
        ...base,
        running: c.running,
        waiting: c.waiting,
        requestsOk: c.ok,
        requestsError: c.errors,
        audioBytesTotal: c.audioBytes,
        requestsPerMin: requestsPerMin != null ? Math.round(requestsPerMin * 10) / 10 : null,
        recentLatency: recent,
        recentAudioBytes: Math.max(0, recentBytes),
        latency: {
          count: c.e2eCount,
          avg: c.e2eCount > 0 ? round3(c.e2eSum / c.e2eCount) : null,
          p50: round3(histogramQuantile(c.buckets, 0.5)),
          p95: round3(histogramQuantile(c.buckets, 0.95)),
        },
        startedAt: c.startedAt,
        lastRequestAt: this._lastRequestAt,
      };
    } catch (err) {
      this.error = err?.name === "TimeoutError" ? "Timed out" : err?.message || String(err);
      return this._default();
    }
  }

  /** Newest sample taken at or before `t`, else the oldest one we have. */
  _sampleAtOrBefore(t) {
    let found = null;
    for (const s of this._samples) {
      if (s.t <= t) found = s;
      else break;
    }
    return found ?? this._samples[0] ?? null;
  }

  _emptyStats() {
    return {
      running: 0,
      waiting: 0,
      requestsOk: null,
      requestsError: null,
      audioBytesTotal: null,
      requestsPerMin: null,
      recentLatency: { count: 0, avg: null, p50: null, p95: null, windowSec: 0 },
      recentAudioBytes: 0,
      latency: { count: 0, avg: null, p50: null, p95: null },
      startedAt: null,
      lastRequestAt: null,
    };
  }

  _default() {
    this._samples = [];
    return {
      available: false,
      error: this.error,
      port: this.port,
      engine: null,
      models: [],
      modelRoot: null,
      maxModelLen: null,
      ...this._emptyStats(),
    };
  }
}
