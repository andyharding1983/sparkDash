import test from "node:test";
import assert from "node:assert/strict";
import {
  AsrProbe,
  parsePrometheus,
  histogramQuantile,
  extractAsrCounters,
} from "../AsrProbe.js";

/** Trimmed /metrics from Whisper large-v3 on vLLM (whisper-atc-derby, port 9000). */
const METRICS = `# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{engine="0",model_name="whisper"} 1.0
vllm:num_requests_waiting{engine="0",model_name="whisper"} 2.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="0.5",model_name="whisper"} 90.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="1.0",model_name="whisper"} 405.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="2.5",model_name="whisper"} 538.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="5.0",model_name="whisper"} 551.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf",model_name="whisper"} 551.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="whisper"} 480.5
vllm:e2e_request_latency_seconds_count{engine="0",model_name="whisper"} 551.0
http_requests_total{handler="/v1/audio/transcriptions",method="POST",status="2xx"} 539.0
http_requests_total{handler="/v1/audio/transcriptions",method="POST",status="4xx"} 3.0
http_requests_total{handler="/v1/models",method="GET",status="2xx"} 99.0
http_request_size_bytes_sum{handler="/v1/audio/transcriptions"} 1.58055443e+08
http_request_size_bytes_sum{handler="/v1/models"} 0.0
process_start_time_seconds 1.79190281143e+09
`;

test("parsePrometheus reads names, labels and values", () => {
  const rows = parsePrometheus(METRICS);
  const run = rows.find((r) => r.name === "vllm:num_requests_running");
  assert.deepEqual(run, {
    name: "vllm:num_requests_running",
    labels: { engine: "0", model_name: "whisper" },
    value: 1,
  });
  assert.ok(!rows.some((r) => r.name.startsWith("#")));
});

test("extractAsrCounters counts only /v1/audio/* requests", () => {
  const c = extractAsrCounters(parsePrometheus(METRICS));
  assert.equal(c.sawVllm, true);
  assert.equal(c.running, 1);
  assert.equal(c.waiting, 2);
  assert.equal(c.ok, 539);
  assert.equal(c.errors, 3);
  assert.equal(c.audioBytes, 158_055_443);
  assert.equal(c.e2eCount, 551);
  assert.equal(c.startedAt, 1_791_902_811_430);
  assert.equal(c.buckets.length, 5);
});

test("histogramQuantile interpolates inside the bucket", () => {
  const { buckets } = extractAsrCounters(parsePrometheus(METRICS));
  // rank 275.5 of 551 falls in (0.5, 1.0]: 0.5 + 0.5 * (275.5 - 90) / 315
  assert.ok(Math.abs(histogramQuantile(buckets, 0.5) - 0.7944) < 1e-3);
  // rank 523.45 falls in (1.0, 2.5]
  assert.ok(Math.abs(histogramQuantile(buckets, 0.95) - (1 + 1.5 * (118.45 / 133))) < 1e-3);
  assert.equal(histogramQuantile([], 0.5), null);
  assert.equal(histogramQuantile([{ le: Infinity, count: 0 }], 0.5), null);
});

test("histogramQuantile in the +Inf bucket returns the last finite bound", () => {
  const b = [
    { le: 1, count: 1 },
    { le: Infinity, count: 10 },
  ];
  assert.equal(histogramQuantile(b, 0.95), 1);
});

test("AsrProbe defaults when host unreachable", async () => {
  const probe = new AsrProbe({ isLocal: true }, 1);
  const snap = await probe.probe();
  assert.equal(snap.available, false);
  assert.equal(snap.port, 1);
  assert.equal(snap.running, 0);
  assert.equal(snap.requestsPerMin, null);
  assert.deepEqual(snap.models, []);
  assert.ok(snap.error);
});

test("AsrProbe setTarget updates port and host", () => {
  const probe = new AsrProbe({ isLocal: false, lanIp: "10.0.0.5" }, 9000);
  assert.match(probe.baseUrl, /10\.0\.0\.5:9000/);
  probe.setTarget({ isLocal: true, lanIp: "10.0.0.5" }, 9100);
  assert.equal(probe.port, 9100);
  assert.match(probe.baseUrl, /127\.0\.0\.1:9100/);
});

/** Replace global fetch with canned /v1/models + /metrics responses. */
function stubFetch(metricsRef) {
  const real = globalThis.fetch;
  globalThis.fetch = async (url) => {
    if (String(url).endsWith("/v1/models")) {
      return new Response(
        JSON.stringify({
          data: [
            { id: "openai/whisper-large-v3", owned_by: "vllm", root: "/models/derby-lora1-e3/", max_model_len: 448 },
            { id: "derby-lora1-e3", owned_by: "vllm", root: "/models/derby-lora1-e3/" },
          ],
        }),
        { status: 200 }
      );
    }
    return new Response(metricsRef.text, { status: 200 });
  };
  return () => {
    globalThis.fetch = real;
  };
}

test("AsrProbe reports models, lifetime latency and windowed rates", async (t) => {
  const ref = { text: METRICS };
  t.after(stubFetch(ref));
  const probe = new AsrProbe({ isLocal: true }, 9000);

  const realNow = Date.now;
  let now = 1_000_000;
  Date.now = () => now;
  t.after(() => {
    Date.now = realNow;
  });

  const first = await probe.probe();
  assert.equal(first.available, true);
  assert.equal(first.engine, "vLLM");
  assert.deepEqual(first.models, ["openai/whisper-large-v3", "derby-lora1-e3"]);
  assert.equal(first.modelRoot, "derby-lora1-e3");
  assert.equal(first.maxModelLen, 448);
  assert.equal(first.requestsOk, 539);
  assert.equal(first.requestsError, 3);
  assert.equal(first.requestsPerMin, null, "one sample is not a rate");
  assert.equal(first.latency.count, 551);
  assert.equal(first.lastRequestAt, null);

  // 30 s later: 6 more successful transcriptions, all in the (0.5, 1.0] bucket.
  now += 30_000;
  ref.text = METRICS.replace("status=\"2xx\"} 539.0", "status=\"2xx\"} 545.0")
    .replace('le="1.0",model_name="whisper"} 405.0', 'le="1.0",model_name="whisper"} 411.0')
    .replace('le="2.5",model_name="whisper"} 538.0', 'le="2.5",model_name="whisper"} 544.0')
    .replace('le="5.0",model_name="whisper"} 551.0', 'le="5.0",model_name="whisper"} 557.0')
    .replace('le="+Inf",model_name="whisper"} 551.0', 'le="+Inf",model_name="whisper"} 557.0')
    .replace("seconds_sum{engine=\"0\",model_name=\"whisper\"} 480.5", "seconds_sum{engine=\"0\",model_name=\"whisper\"} 485.0")
    .replace("seconds_count{engine=\"0\",model_name=\"whisper\"} 551.0", "seconds_count{engine=\"0\",model_name=\"whisper\"} 557.0");
  const second = await probe.probe();
  assert.equal(second.requestsPerMin, 12, "6 requests in 30 s");
  assert.equal(second.recentLatency.count, 6);
  assert.equal(second.recentLatency.avg, 0.75);
  assert.ok(second.recentLatency.p50 > 0.5 && second.recentLatency.p50 <= 1.0);
  assert.equal(second.recentLatency.windowSec, 30);
  assert.equal(second.lastRequestAt, now);
});

test("AsrProbe drops its history when the server restarts", async (t) => {
  const ref = { text: METRICS };
  t.after(stubFetch(ref));
  const probe = new AsrProbe({ isLocal: true }, 9000);
  await probe.probe();
  ref.text = METRICS.replace("status=\"2xx\"} 539.0", "status=\"2xx\"} 2.0")
    .replace("seconds_count{engine=\"0\",model_name=\"whisper\"} 551.0", "seconds_count{engine=\"0\",model_name=\"whisper\"} 2.0");
  const after = await probe.probe();
  assert.equal(after.requestsOk, 2);
  assert.equal(after.requestsPerMin, null, "no rate across a counter reset");
  assert.equal(after.recentLatency.count, 0);
});
