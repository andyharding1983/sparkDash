import test from "node:test";
import assert from "node:assert/strict";
import fs from "fs";
import os from "os";
import path from "path";
import { tagGpuProcesses } from "../gpuServices.js";

/**
 * Fake /proc: LLM server (pid 100, listens :8888) with scheduler child 101;
 * STT server (pid 200, listens :9000) with EngineCore child 201; an unrelated
 * GPU process 300 under pid 1.
 */
function fakeProc() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "sparkdash-proc-"));
  fs.mkdirSync(path.join(root, "net"));
  const header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n";
  // 0x22B8 = 8888, 0x2328 = 9000; st 0A = LISTEN. A non-listening :9000 socket (st 01) must be ignored.
  fs.writeFileSync(
    path.join(root, "net/tcp"),
    header +
      "   0: 00000000:22B8 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 5001 1\n" +
      "   1: 00000000:2328 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 5002 1\n" +
      "   2: 0100007F:2328 0100007F:9999 01 00000000:00000000 00:00000000 00000000     0        0 5003 1\n"
  );
  fs.writeFileSync(path.join(root, "net/tcp6"), header);
  const proc = (pid, ppid, comm, sockets = []) => {
    const dir = path.join(root, String(pid));
    fs.mkdirSync(path.join(dir, "fd"), { recursive: true });
    fs.writeFileSync(path.join(dir, "stat"), `${pid} (${comm}) S ${ppid} 0 0 0\n`);
    sockets.forEach((inode, i) => fs.symlinkSync(`socket:[${inode}]`, path.join(dir, "fd", String(3 + i))));
  };
  proc(100, 1, "python3", [5001]);
  proc(101, 100, "sglang::scheduler");
  proc(200, 1, "vllm serve", [5002]);
  proc(201, 200, "VLLM::EngineCore", [5003]);
  proc(300, 1, "some) odd name");
  return root;
}

test("tagGpuProcesses follows parents to the listening service", async () => {
  const root = fakeProc();
  const out = await tagGpuProcesses(
    [
      { pid: 101, name: "sglang::scheduler", vramMB: 70_000 },
      { pid: 201, name: "VLLM::EngineCore", vramMB: 8_800 },
      { pid: 100, name: "python3", vramMB: 1_400 },
      { pid: 300, name: "other", vramMB: 500 },
    ],
    [
      { service: "llm", port: 8888 },
      { service: "asr", port: 9000 },
    ],
    root
  );
  assert.deepEqual(
    out.map((p) => [p.pid, p.service ?? null, p.port ?? null]),
    [
      [101, "llm", 8888],
      [201, "asr", 9000],
      [100, "llm", 8888],
      [300, null, null],
    ]
  );
});

test("tagGpuProcesses leaves processes alone with no services or no /proc", async () => {
  const procs = [{ pid: 101, name: "x", vramMB: 1 }];
  assert.equal(await tagGpuProcesses(procs, [], "/nonexistent"), procs);
  const out = await tagGpuProcesses(procs, [{ service: "llm", port: 8888 }], "/nonexistent-proc-root");
  assert.deepEqual(out, procs);
});
