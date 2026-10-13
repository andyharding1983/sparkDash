import type { AsrMetrics } from "../../api/types";
import { useMetricsHistoryTail } from "../../hooks/metricsStore";
import { Panel } from "../ui/Panel";
import { Tag } from "../ui/Tag";
import { TrendLine } from "../ui/TrendLine";
import { MicIcon } from "../ui/icons";

interface AsrPanelProps {
  asr: AsrMetrics | null;
  asrPort: number;
  sparkId: string;
  className?: string;
}

function formatSeconds(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return "—";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  return `${s.toFixed(2)} s`;
}

function formatBytes(b: number | null | undefined): string {
  if (b == null || !Number.isFinite(b)) return "—";
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(0)} KB`;
  return `${Math.round(b)} B`;
}

function formatAgo(ms: number | null | undefined): string | null {
  if (ms == null || !Number.isFinite(ms)) return null;
  const sec = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (sec < 60) return `${sec}s ago`;
  const m = Math.floor(sec / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 48) return `${h}h ${m % 60}m ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function formatUptime(startedAt: number | null | undefined): string | null {
  if (startedAt == null || !Number.isFinite(startedAt)) return null;
  const sec = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
  const d = Math.floor(sec / 86400);
  const h = Math.floor((sec % 86400) / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function Tile({
  label,
  value,
  sub,
  tone = "",
  title,
}: {
  label: string;
  value: string;
  sub?: string | null;
  tone?: string;
  title?: string;
}) {
  return (
    <div className="sp-tile" title={title}>
      <b className={tone}>{value}</b>
      <span>{label}</span>
      {sub ? <small>{sub}</small> : null}
    </div>
  );
}

/**
 * Speech-to-text (Whisper on vLLM) card: throughput, queue, latency.
 * Counters are cumulative since the server started; "recent" figures are
 * deltas over the probe's last ~5 minutes.
 */
export function AsrPanel({ asr, asrPort, sparkId, className = "" }: AsrPanelProps) {
  const rpmHistory = useMetricsHistoryTail(sparkId, "asr.rpm");
  const available = Boolean(asr?.available);

  const recent = asr?.recentLatency;
  const hasRecent = (recent?.count ?? 0) > 0;
  const shownLatency = hasRecent ? recent : asr?.latency;
  const latencyScope = hasRecent
    ? `last ${Math.max(1, Math.round((recent?.windowSec ?? 0) / 60))} min · ${recent?.count} req`
    : "since start";

  const ok = asr?.requestsOk ?? 0;
  const err = asr?.requestsError ?? 0;
  const total = ok + err;
  const errPct = total > 0 ? (err / total) * 100 : 0;
  const lastAgo = formatAgo(asr?.lastRequestAt);
  const uptime = formatUptime(asr?.startedAt);
  const primaryModel = asr?.models?.[0] ?? null;
  const aliases = (asr?.models ?? []).slice(1);
  const waiting = asr?.waiting ?? 0;

  return (
    <Panel
      title="Speech-to-text"
      icon={<MicIcon />}
      className={className}
      bodyClassName="sp-stack"
      actions={
        <div className="sp-actions">
          {available ? (
            <Tag tone="info" title={`Engine: ${asr?.engine ?? "unknown"}`}>
              {asr?.engine ?? "STT"} · :{asrPort}
            </Tag>
          ) : (
            <Tag tone="bad">down · :{asrPort}</Tag>
          )}
        </div>
      }
    >
      {!available ? (
        <div className="sp-stack sp-stack--tight">
          <p className="sp-muted">Not reachable on :{asrPort}</p>
          <p className="sp-hint">
            {asr?.error || "Ensure the speech-to-text server is running on this host."}
          </p>
        </div>
      ) : (
        <>
          {primaryModel ? (
            <div className="sp-model">
              <div className="eyebrow">
                <span>Model</span>
                {uptime ? <span title="Server process uptime">up {uptime}</span> : null}
              </div>
              <code title={primaryModel}>{primaryModel}</code>
              {asr?.modelRoot || aliases.length > 0 ? (
                <small>
                  {asr?.modelRoot ? `weights: ${asr.modelRoot}` : ""}
                  {aliases.length > 0
                    ? `${asr?.modelRoot ? " · " : ""}alias: ${aliases.join(", ")}`
                    : ""}
                </small>
              ) : null}
            </div>
          ) : null}

          <div className="sp-metric">
            <span className="eyebrow">Requests</span>
            <div className="big-num sp-big-lg">
              {asr?.requestsPerMin != null ? asr.requestsPerMin.toFixed(1) : "—"}
              <small>/ min</small>
            </div>
            <TrendLine data={rpmHistory} height={44} color="var(--color-accent)" />
            <span className="sp-avg">
              {lastAgo ? `Last request ${lastAgo}` : "No requests seen since sparkDash started"}
            </span>
          </div>

          <div className="sp-tiles">
            <Tile
              label="Queue"
              value={`${asr?.running ?? 0} run`}
              sub={`${waiting} waiting`}
              tone={waiting > 0 ? "text-warning" : ""}
            />
            <Tile
              label="Latency p50"
              value={formatSeconds(shownLatency?.p50)}
              sub={latencyScope}
              title="End-to-end request latency (vllm:e2e_request_latency_seconds)"
            />
            <Tile
              label="Latency p95"
              value={formatSeconds(shownLatency?.p95)}
              sub={`avg ${formatSeconds(shownLatency?.avg)}`}
              tone={(shownLatency?.p95 ?? 0) > 5 ? "text-warning" : ""}
            />
            <Tile
              label="Transcribed"
              value={ok.toLocaleString()}
              sub={err > 0 ? `${err.toLocaleString()} errors (${errPct.toFixed(1)}%)` : "0 errors"}
              tone={err > 0 ? "text-warning" : ""}
              title="Requests to /v1/audio/* since the server started"
            />
          </div>

          <div className="sp-row sp-row--rule">
            <span className="text-muted">Audio in</span>
            <span className="mono text-text">
              {formatBytes(asr?.audioBytesTotal)} since start
              {(asr?.recentAudioBytes ?? 0) > 0
                ? ` · ${formatBytes(asr?.recentAudioBytes)} last 5 min`
                : ""}
            </span>
          </div>
          {hasRecent && asr?.latency?.p95 != null ? (
            <div className="sp-row">
              <span className="text-muted">Lifetime latency</span>
              <span className="mono text-text">
                p50 {formatSeconds(asr.latency.p50)} · p95 {formatSeconds(asr.latency.p95)}
              </span>
            </div>
          ) : null}
          {asr?.error ? <p className="sp-error">{asr.error}</p> : null}
        </>
      )}
    </Panel>
  );
}
