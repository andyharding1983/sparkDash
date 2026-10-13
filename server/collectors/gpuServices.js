/**
 * Attribute GPU processes to the services sparkDash probes on this host.
 *
 * nvidia-smi only gives a PID and a process name ("VLLM::EngineCore",
 * "python3"), which says nothing about *which* server it is. Each probed
 * service listens on a known port, so: find the socket inodes listening on
 * those ports (/proc/net/tcp{,6}), then walk each GPU process and its parents
 * until one holds such a socket. Engine workers (SGLang scheduler, vLLM
 * EngineCore) are children of the HTTP server process, so the walk finds them.
 *
 * Local units only: it reads the host's /proc (docker `pid: host` +
 * `network_mode: host`). Remote processes are returned untagged.
 */
import fs from "fs/promises";
import path from "path";

/** Parent levels to climb from a GPU process before giving up. */
const MAX_DEPTH = 8;
/** Re-check a PID's service after this long (PIDs are reused). */
const CACHE_TTL_MS = 60_000;

/** @type {Map<string, { at: number, tag: { service: string, port: number } | null }>} */
const cache = new Map();

/**
 * Listening socket inode → port, for the ports we care about.
 * @param {string} procRoot
 * @param {Set<number>} ports
 */
async function listeningInodes(procRoot, ports) {
  /** @type {Map<string, number>} */
  const out = new Map();
  for (const file of ["net/tcp", "net/tcp6"]) {
    let text;
    try {
      text = await fs.readFile(path.join(procRoot, file), "utf8");
    } catch {
      continue;
    }
    for (const line of text.split("\n").slice(1)) {
      const cols = line.trim().split(/\s+/);
      // local_address is "HEXIP:HEXPORT"; st 0A = LISTEN; inode is column 9.
      if (cols.length < 10 || cols[3] !== "0A") continue;
      const port = parseInt(cols[1].split(":").pop() || "", 16);
      if (ports.has(port) && cols[9] !== "0") out.set(cols[9], port);
    }
  }
  return out;
}

/** @param {string} procRoot @param {number} pid */
async function parentPid(procRoot, pid) {
  try {
    const stat = await fs.readFile(path.join(procRoot, String(pid), "stat"), "utf8");
    // comm may contain spaces/parens; fields after the last ")" are fixed.
    const rest = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
    const ppid = parseInt(rest[1], 10);
    return Number.isInteger(ppid) ? ppid : 0;
  } catch {
    return 0;
  }
}

/** @param {string} procRoot @param {number} pid @param {Map<string, number>} inodes */
async function listenPortOf(procRoot, pid, inodes) {
  let fds;
  try {
    fds = await fs.readdir(path.join(procRoot, String(pid), "fd"));
  } catch {
    return null;
  }
  for (const fd of fds) {
    let target;
    try {
      target = await fs.readlink(path.join(procRoot, String(pid), "fd", fd));
    } catch {
      continue;
    }
    const m = /^socket:\[(\d+)\]$/.exec(target);
    if (m && inodes.has(m[1])) return inodes.get(m[1]) ?? null;
  }
  return null;
}

/**
 * Add `service` ("llm" | "asr" | "comfy") and `port` to each process that
 * belongs to a probed service. Returns new objects; untouched when nothing
 * matches or on any error.
 *
 * @template {{ pid: number }} P
 * @param {P[]} processes
 * @param {{ service: string, port: number }[]} services
 * @param {string} procRoot
 * @returns {Promise<Array<P & { service?: string, port?: number }>>}
 */
export async function tagGpuProcesses(processes, services, procRoot) {
  if (!Array.isArray(processes) || processes.length === 0 || services.length === 0) {
    return processes;
  }
  /** @type {Map<number, string>} */
  const serviceByPort = new Map();
  for (const s of services) {
    if (!serviceByPort.has(s.port)) serviceByPort.set(s.port, s.service);
  }
  const now = Date.now();
  /** @type {Map<string, number> | null} */
  let inodes = null;

  const out = [];
  for (const proc of processes) {
    const key = `${proc.pid}:${[...serviceByPort.keys()].join(",")}`;
    const hit = cache.get(key);
    let tag = hit && now - hit.at < CACHE_TTL_MS ? hit.tag : undefined;
    if (tag === undefined) {
      inodes ??= await listeningInodes(procRoot, new Set(serviceByPort.keys()));
      tag = null;
      let pid = proc.pid;
      for (let depth = 0; depth < MAX_DEPTH && pid > 1 && inodes.size > 0; depth++) {
        const port = await listenPortOf(procRoot, pid, inodes);
        if (port != null) {
          tag = { service: serviceByPort.get(port) ?? "other", port };
          break;
        }
        pid = await parentPid(procRoot, pid);
      }
      cache.set(key, { at: now, tag });
    }
    out.push(tag ? { ...proc, ...tag } : proc);
  }
  // Drop stale entries so the cache tracks only live PIDs.
  for (const [k, v] of cache) if (now - v.at > CACHE_TTL_MS * 2) cache.delete(k);
  return out;
}
