import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { access } from "node:fs/promises";
import { constants } from "node:fs";
import { InferenceError, checkCancellation } from "./inference-error";

export interface HelperResponse {
  type: string;
  id?: string;
  message?: string;
  text?: string;
  duration?: number;
  elapsed?: number;
  language?: string;
  value?: number;
  engineVersion?: string;
  includedTerms?: string[];
  omittedTerms?: string[];
  tokenCount?: number;
  tokenBudget?: number;
  probabilities?: number[];
  frameSeconds?: number;
}

export interface HelperConfiguration {
  name: string;
  executable: string;
  arguments: string[];
  requiredFiles: string[];
  loadTimeout: number;
  lineLimit: number;
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function decodeResponse(line: Buffer): HelperResponse | undefined {
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(line));
  } catch {
    return;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) return;
  const response = value as Record<string, unknown>;
  if (typeof response.type !== "string") return;
  const strings = ["id", "message", "text", "language", "engineVersion"];
  const numbers = ["duration", "elapsed", "value", "frameSeconds"];
  const integers = ["tokenCount", "tokenBudget"];
  if (
    strings.some(
      (key) =>
        response[key] != null &&
        (typeof response[key] !== "string" || /[\uD800-\uDFFF]/u.test(response[key])),
    )
  )
    return;
  if (numbers.some((key) => response[key] != null && typeof response[key] !== "number")) return;
  if (integers.some((key) => response[key] != null && !Number.isSafeInteger(response[key]))) return;
  const probabilities = response.probabilities;
  if (
    probabilities != null &&
    (!Array.isArray(probabilities) ||
      probabilities.some((value) => typeof value !== "number" || !(value >= 0 && value <= 1)))
  )
    return;
  for (const key of ["includedTerms", "omittedTerms"]) {
    const terms = response[key];
    if (
      terms != null &&
      (!Array.isArray(terms) ||
        terms.some((term) => typeof term !== "string" || /[\uD800-\uDFFF]/u.test(term)))
    )
      return;
  }
  // JSON null is the same as an absent optional field in Swift's decoder.
  for (const key of Object.keys(response)) if (response[key] === null) delete response[key];
  return response as unknown as HelperResponse;
}

/** A warm, bounded JSON-lines subprocess. Only one request may be outstanding. */
export class HelperProcess {
  private child?: ChildProcessWithoutNullStreams;
  private generation = 0;
  private loaded = false;
  private engineVersion?: string;
  private loading?: Promise<void>;
  private ready?: ReturnType<typeof deferred<void>>;
  private result?: ReturnType<typeof deferred<HelperResponse>>;
  private operation?: symbol;
  private requestID?: string;
  private progress?: (value: number) => void;
  private timeout?: ReturnType<typeof setTimeout>;
  private readonly retiring = new Map<ChildProcessWithoutNullStreams, Promise<void>>();

  constructor(private readonly configuration: HelperConfiguration) {}

  snapshot() {
    return {
      loaded: Boolean(
        this.loaded && this.child?.exitCode === null && this.child.signalCode === null,
      ),
      loading: this.loading !== undefined,
      busy: this.operation !== undefined,
      engineVersion: this.engineVersion,
    };
  }

  async ensureLoaded(signal?: AbortSignal) {
    checkCancellation(signal);
    if (this.snapshot().loaded) return;
    if (!this.loading) {
      const loading = this.start();
      this.loading = loading;
      // A reset or replacement must not be undone by an old task's finalizer.
      void loading
        .finally(() => {
          if (this.loading === loading) this.loading = undefined;
        })
        .catch(() => {});
    }
    const loading = this.loading;
    const cancel = () => {
      if (this.loading === loading) this.reset(new InferenceError("cancelled"));
    };
    signal?.addEventListener("abort", cancel, { once: true });
    try {
      await loading;
      checkCancellation(signal);
    } finally {
      signal?.removeEventListener("abort", cancel);
    }
  }

  async request(
    payload: object,
    id: string,
    timeout: number,
    onProgress?: (value: number) => void,
    signal?: AbortSignal,
  ) {
    checkCancellation(signal);
    if (this.operation) throw new InferenceError("busy");
    const line = JSON.stringify(payload) + "\n";
    const operation = Symbol();
    this.operation = operation;
    const cancel = () => {
      if (this.operation === operation) this.reset(new InferenceError("cancelled"));
    };
    signal?.addEventListener("abort", cancel, { once: true });
    try {
      await this.ensureLoaded(signal);
      checkCancellation(signal);
      const child = this.child;
      if (this.operation !== operation || !child || !this.snapshot().loaded)
        throw new InferenceError("cancelled");
      const result = deferred<HelperResponse>();
      this.result = result;
      this.requestID = id;
      this.progress = onProgress;
      const current = this.generation;
      this.setTimeout(timeout, `${this.configuration.name} inference timed out.`);
      child.stdin.write(line, (error) => {
        if (error) this.transportFailed(current, `Could not write to ${this.configuration.name}.`);
      });
      return await result.promise;
    } finally {
      signal?.removeEventListener("abort", cancel);
      if (this.operation === operation) this.operation = undefined;
    }
  }

  /** Cancellation preserves helpers that are idle and warm. */
  cancel() {
    if (this.operation || this.loading || this.ready) this.reset(new InferenceError("cancelled"));
  }

  async shutdown() {
    this.reset(new InferenceError("cancelled"));
    // Children retired by previous cancellation or protocol errors still belong
    // to this supervisor and must exit before the server releases its archive.
    await Promise.all(this.retiring.values());
  }

  private async start() {
    const configuration = this.configuration;
    const startingGeneration = this.generation;
    try {
      await access(configuration.executable, constants.X_OK);
    } catch {
      throw new InferenceError(
        "unavailable",
        `${configuration.name} executable is missing: ${configuration.executable}`,
      );
    }
    try {
      await Promise.all(configuration.requiredFiles.map((file) => access(file, constants.R_OK)));
    } catch {
      throw new InferenceError(
        "unavailable",
        `${configuration.name} model files are missing or unreadable.`,
      );
    }
    // Asset checks are asynchronous. A cancelled or replaced start cannot launch.
    if (this.generation !== startingGeneration) throw new InferenceError("cancelled");
    const current = ++this.generation;
    this.loaded = false;
    this.engineVersion = undefined;
    const ready = deferred<void>();
    this.ready = ready;
    let child: ChildProcessWithoutNullStreams;
    try {
      child = spawn(configuration.executable, configuration.arguments, {
        stdio: ["pipe", "pipe", "pipe"],
        shell: false,
      });
    } catch {
      this.reset(new InferenceError("unavailable", `Could not launch ${configuration.name}.`));
      return ready.promise;
    }
    this.child = child;
    let buffer = Buffer.alloc(0);
    child.stdout.on("data", (chunk: Buffer) => {
      if (this.generation !== current) return;
      // Never append an arbitrarily large chunk before checking its lines.
      let offset = 0;
      while (offset < chunk.length) {
        const newline = chunk.indexOf(0x0a, offset);
        const end = newline === -1 ? chunk.length : newline;
        if (buffer.length + end - offset > configuration.lineLimit) {
          this.transportFailed(current, "Native helper response exceeded its size limit.");
          return;
        }
        const piece = chunk.subarray(offset, end);
        const line = buffer.length ? Buffer.concat([buffer, piece]) : piece;
        if (newline === -1) {
          buffer = Buffer.from(line);
          return;
        }
        buffer = Buffer.alloc(0);
        this.receive(line, current);
        if (this.generation !== current) return;
        offset = newline + 1;
      }
    });
    child.stdout.on("error", () =>
      this.transportFailed(current, "Could not read native helper output."),
    );
    child.stdout.on("end", () => this.transportFailed(current, "Native helper closed its output."));
    child.stdin.on("error", () =>
      this.transportFailed(current, `Could not write to ${configuration.name}.`),
    );
    // Drain GPU diagnostics without retaining them or logging transcript content.
    child.stderr.resume();
    child.stderr.on("error", () => {});
    child.on("error", () =>
      this.transportFailed(current, `Could not launch ${configuration.name}.`),
    );
    child.on("close", () => this.transportFailed(current, "Native helper closed its output."));
    this.setTimeout(configuration.loadTimeout, `${configuration.name} model loading timed out.`);
    return ready.promise;
  }

  private receive(line: Buffer, current: number) {
    if (current !== this.generation) return;
    const response = decodeResponse(line);
    if (!response) {
      this.reset(
        new InferenceError("invalidResponse", `${this.configuration.name} returned invalid JSON.`),
      );
      return;
    }
    switch (response.type) {
      case "ready": {
        if (!this.ready || this.loaded) return;
        const ready = this.ready;
        this.ready = undefined;
        this.clearTimeout();
        this.loaded = true;
        this.engineVersion = response.engineVersion;
        ready.resolve();
        break;
      }
      case "progress":
        if (
          response.id === this.requestID &&
          response.value !== undefined &&
          Number.isFinite(response.value)
        ) {
          this.progress?.(Math.min(1, Math.max(0, response.value)));
        }
        break;
      case "result": {
        if (response.id !== this.requestID || !this.result) return;
        const result = this.result;
        this.result = undefined;
        this.requestID = undefined;
        this.progress = undefined;
        this.clearTimeout();
        result.resolve(response);
        break;
      }
      case "error":
        if (response.id !== undefined && response.id !== this.requestID) return;
        this.reset(
          new InferenceError(
            "unavailable",
            response.message ?? `${this.configuration.name} inference failed.`,
          ),
        );
        break;
      default:
        this.reset(
          new InferenceError(
            "invalidResponse",
            `${this.configuration.name} returned an unknown event.`,
          ),
        );
    }
  }

  private transportFailed(current: number, message: string) {
    if (this.generation === current) this.reset(new InferenceError("unavailable", message));
  }

  private setTimeout(seconds: number, message: string) {
    this.clearTimeout();
    this.timeout = setTimeout(
      () => this.reset(new InferenceError("timeout", message)),
      Math.max(0.001, seconds) * 1000,
    );
  }

  private clearTimeout() {
    if (this.timeout) clearTimeout(this.timeout);
    this.timeout = undefined;
  }

  private reset(error: InferenceError) {
    ++this.generation;
    this.loaded = false;
    this.engineVersion = undefined;
    this.clearTimeout();
    const ready = this.ready;
    const result = this.result;
    this.ready = undefined;
    this.result = undefined;
    this.requestID = undefined;
    this.operation = undefined;
    this.progress = undefined;
    this.loading = undefined;
    const child = this.child;
    this.child = undefined;
    child?.stdin.destroy();
    if (child) this.retire(child);
    ready?.reject(error);
    result?.reject(error);
  }

  private retire(child: ChildProcessWithoutNullStreams) {
    if (this.retiring.has(child) || child.exitCode !== null || child.signalCode !== null) return;
    const exited = deferred<void>();
    this.retiring.set(child, exited.promise);
    const escalation = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    }, 2000);
    const complete = () => {
      clearTimeout(escalation);
      child.removeListener("exit", complete);
      child.removeListener("close", complete);
      this.retiring.delete(child);
      exited.resolve();
    };
    // A failed spawn has no exit event; close also completes that retirement.
    child.once("exit", complete);
    child.once("close", complete);
    child.kill("SIGTERM");
  }
}
