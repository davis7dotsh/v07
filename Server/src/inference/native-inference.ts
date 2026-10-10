import { randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { access } from "node:fs/promises";
import { availableParallelism } from "node:os";
import type { ModelHintUsage } from "../api";
import { HelperProcess, type HelperResponse } from "./helper-process";
import { checkCancellation, InferenceError } from "./inference-error";
import { ModelVerifier, type ModelPin } from "./model-verification";
import { linuxProofModelPin, macProofManifestSHA256, speechModelPin } from "./model-pins";

export { InferenceError } from "./inference-error";

export interface InferenceConfiguration {
  speechHelper: string;
  speechModel: string;
  vadModel: string;
  proofHelper: string;
  proofModel: string;
  threads: number;
  speechLoadTimeout: number;
  speechTimeout: number;
  proofLoadTimeout: number;
  proofTimeout: number;
}

type ConfigurationInput = Pick<
  InferenceConfiguration,
  "speechHelper" | "speechModel" | "vadModel" | "proofHelper" | "proofModel"
> &
  Partial<
    Omit<
      InferenceConfiguration,
      "speechHelper" | "speechModel" | "vadModel" | "proofHelper" | "proofModel"
    >
  >;

function deadline(value: number | undefined, fallback: number) {
  return value !== undefined && Number.isFinite(value) && value > 0 && value <= 3600
    ? value
    : fallback;
}

export function createInferenceConfiguration(input: ConfigurationInput): InferenceConfiguration {
  const threads = input.threads ?? Math.min(8, Math.max(2, Math.floor(availableParallelism() / 2)));
  return {
    ...input,
    threads: Number.isFinite(threads) ? Math.min(32, Math.max(1, Math.trunc(threads))) : 2,
    speechLoadTimeout: deadline(input.speechLoadTimeout, 120),
    speechTimeout: deadline(input.speechTimeout, 180),
    proofLoadTimeout: deadline(input.proofLoadTimeout, 30),
    proofTimeout: deadline(input.proofTimeout, 18),
  };
}

export interface InferenceReadiness {
  available: boolean;
  message: string;
  speechLoaded: boolean;
  proofLoaded: boolean;
}

export interface SpeechInferenceResult {
  text: string;
  audioSeconds: number;
  processingSeconds: number;
  language: string;
  engineVersion?: string;
  modelSHA256?: string;
  hints?: ModelHintUsage;
}

export interface SpeechActivity {
  /** Silero speech probability per VAD window, in order. */
  probabilities: number[];
  frameSeconds: number;
  processingSeconds: number;
}

/** Recognize the whole file as context but return only words starting in [from, until) seconds. */
export interface KeptRange {
  from: number;
  until?: number;
}

export interface ProofInferenceResult {
  text: string;
  processingSeconds: number;
  engineVersion?: string;
  modelSHA256?: string;
}

export interface InferenceBackend {
  readiness(proofreadingEnabled?: boolean): Promise<InferenceReadiness>;
  warmUp(proofreadingEnabled?: boolean, signal?: AbortSignal): Promise<void>;
  transcribe(
    audioPath: string,
    language: string,
    vocabularyTerms: string[],
    onProgress?: (value: number) => void,
    signal?: AbortSignal,
    keep?: KeptRange,
  ): Promise<SpeechInferenceResult>;
  detectSpeech(audioPath: string, signal?: AbortSignal): Promise<SpeechActivity>;
  correct(
    text: string,
    terms: string[],
    language: string,
    systemPrompt: string,
    signal?: AbortSignal,
  ): Promise<ProofInferenceResult>;
  cancel(): Promise<void>;
  shutdown(): Promise<void>;
}

function bytes(value: string) {
  return Buffer.byteLength(value, "utf8");
}

function validLanguage(language: string) {
  return language.length > 0 && bytes(language) <= 32 && !language.includes("\0");
}

function vocabularyDiagnostics(
  response: HelperResponse,
  terms: string[],
): ModelHintUsage | undefined {
  const { includedTerms: included, omittedTerms: omitted, tokenCount, tokenBudget } = response;
  if (
    included === undefined &&
    omitted === undefined &&
    tokenCount === undefined &&
    tokenBudget === undefined
  )
    return;
  if (
    included &&
    omitted &&
    tokenCount !== undefined &&
    tokenBudget !== undefined &&
    tokenCount >= 0 &&
    tokenBudget >= 0 &&
    (tokenBudget > 0 || included.length === 0) &&
    tokenCount <= tokenBudget &&
    included.length + omitted.length === terms.length
  ) {
    const includedSet = new Set(included);
    const omittedSet = new Set(omitted);
    const combined = new Set([...included, ...omitted]);
    if (
      included.every((term) => !omittedSet.has(term)) &&
      combined.size === terms.length &&
      terms.every((term) => combined.has(term)) &&
      JSON.stringify(included) === JSON.stringify(terms.filter((term) => includedSet.has(term))) &&
      JSON.stringify(omitted) === JSON.stringify(terms.filter((term) => omittedSet.has(term)))
    ) {
      return { includedTerms: included, omittedTerms: omitted, tokenCount, tokenBudget };
    }
  }
  throw new InferenceError("invalidResponse", "Parakeet returned invalid vocabulary diagnostics.");
}

/** The helper executables own inference; the server owns paths, deadlines and pins. */
export class NativeInference implements InferenceBackend {
  private readonly configuration: InferenceConfiguration;
  private readonly speech: HelperProcess;
  private readonly proof: HelperProcess;
  private readonly verifier = new ModelVerifier();
  private readonly speechPin?: ModelPin;
  private readonly proofPin?: ModelPin;
  private readonly proofManifestSHA256?: string;

  constructor(configuration: ConfigurationInput) {
    this.configuration = createInferenceConfiguration(configuration);
    this.speechPin = speechModelPin;
    this.proofPin = process.platform === "darwin" ? undefined : linuxProofModelPin;
    this.proofManifestSHA256 = process.platform === "darwin" ? macProofManifestSHA256 : undefined;
    const config = this.configuration;
    this.speech = new HelperProcess({
      name: "Parakeet",
      executable: config.speechHelper,
      arguments: [
        "--model",
        config.speechModel,
        "--vad-model",
        config.vadModel,
        "--threads",
        String(config.threads),
      ],
      requiredFiles: [config.speechModel, config.vadModel],
      loadTimeout: config.speechLoadTimeout,
      lineLimit: 1_048_576,
    });
    this.proof = new HelperProcess({
      name: "Qwen",
      executable: config.proofHelper,
      arguments: ["--model", config.proofModel, "--threads", String(config.threads)],
      requiredFiles: [config.proofModel],
      loadTimeout: config.proofLoadTimeout,
      lineLimit: 65_536,
    });
  }

  async readiness(proofreadingEnabled = true): Promise<InferenceReadiness> {
    const config = this.configuration;
    const speechState = this.speech.snapshot();
    const proofState = this.proof.snapshot();
    const helpers = proofreadingEnabled
      ? [config.speechHelper, config.proofHelper]
      : [config.speechHelper];
    const models = [
      config.speechModel,
      config.vadModel,
      ...(proofreadingEnabled ? [config.proofModel] : []),
    ];
    let missing: string | undefined;
    for (const [paths, mode] of [
      [helpers, constants.X_OK],
      [models, constants.R_OK],
    ] as const) {
      for (const path of paths) {
        try {
          await access(path, mode);
        } catch {
          missing ??= path;
        }
      }
    }
    const speechVerified = await this.verifier.isVerified(config.speechModel, this.speechPin);
    const proofFileVerified = await this.verifier.isVerified(config.proofModel, this.proofPin);
    const proofVerified = proofFileVerified && (!this.proofManifestSHA256 || proofState.loaded);
    const warm = speechState.loaded && (!proofreadingEnabled || proofState.loaded);
    return {
      available: !missing && speechVerified && (!proofreadingEnabled || proofVerified),
      message: missing
        ? `Missing or inaccessible inference asset: ${missing}`
        : !speechVerified || (proofreadingEnabled && !proofVerified)
          ? "Models need integrity verification."
          : warm
            ? "Models are warm and ready."
            : "Models need to warm up.",
      speechLoaded: speechState.loaded,
      proofLoaded: proofState.loaded,
    };
  }

  async warmUp(proofreadingEnabled = true, signal?: AbortSignal) {
    try {
      await this.verifier.verify(this.configuration.speechModel, this.speechPin, signal);
    } catch (error) {
      // A readable file that fails the pin is usually an earlier install's Whisper weights.
      const readable = await access(this.configuration.speechModel, constants.R_OK).then(
        () => true,
        () => false,
      );
      if (error instanceof InferenceError && error.code === "unavailable" && readable)
        throw new InferenceError(
          "unavailable",
          "The speech model must be Parakeet v3. Run scripts/download-model.sh.",
        );
      throw error;
    }
    await this.speech.ensureLoaded(signal);
    if (proofreadingEnabled) {
      await this.verifier.verify(this.configuration.proofModel, this.proofPin, signal);
      await this.proof.ensureLoaded(signal);
    }
  }

  async transcribe(
    audioPath: string,
    language: string,
    vocabularyTerms: string[],
    onProgress?: (value: number) => void,
    signal?: AbortSignal,
    keep?: KeptRange,
  ): Promise<SpeechInferenceResult> {
    checkCancellation(signal);
    let readable = true;
    try {
      await access(audioPath, constants.R_OK);
    } catch {
      readable = false;
    }
    if (
      !readable ||
      !validLanguage(language) ||
      vocabularyTerms.length > 8192 ||
      vocabularyTerms.some(
        (term) =>
          !term.length ||
          bytes(term) > 16_384 ||
          term !== term.trim() ||
          /[\p{Cc}\p{Cf}]/u.test(term),
      ) ||
      vocabularyTerms.reduce((total, term) => total + bytes(term), 0) > 384 * 1024 ||
      (keep !== undefined &&
        !(keep.from >= 0 && (keep.until === undefined || keep.until > keep.from)))
    ) {
      throw new InferenceError("invalidRequest", "Audio, language, or Parakeet prompt is invalid.");
    }
    if (new Set(vocabularyTerms).size !== vocabularyTerms.length)
      throw new InferenceError("invalidRequest", "Parakeet vocabulary terms must be unique.");
    const digest = await this.verifier.verify(
      this.configuration.speechModel,
      this.speechPin,
      signal,
    );
    const request = {
      type: "transcribe",
      id: randomUUID(),
      path: audioPath,
      language,
      vocabularyTerms,
      ...(keep ? { keepFrom: keep.from, keepUntil: keep.until } : {}),
    };
    if (bytes(JSON.stringify(request)) >= 1_048_576)
      throw new InferenceError("invalidRequest", "The encoded vocabulary exceeds 1 MB.");
    const response = await this.speech.request(
      request,
      request.id,
      this.configuration.speechTimeout,
      onProgress,
      signal,
    );
    if (
      response.text === undefined ||
      bytes(response.text) > 256 * 1024 ||
      response.text.includes("\0") ||
      response.duration === undefined ||
      !Number.isFinite(response.duration) ||
      response.duration < 0 ||
      response.elapsed === undefined ||
      !Number.isFinite(response.elapsed) ||
      response.elapsed < 0 ||
      response.language === undefined ||
      !response.language.length ||
      bytes(response.language) > 32
    ) {
      await this.speech.shutdown();
      throw new InferenceError("invalidResponse", "Parakeet returned an invalid transcript.");
    }
    let hints: ModelHintUsage | undefined;
    try {
      hints = vocabularyDiagnostics(response, vocabularyTerms);
    } catch (error) {
      await this.speech.shutdown();
      throw error;
    }
    const state = this.speech.snapshot();
    return {
      text: response.text,
      audioSeconds: response.duration,
      processingSeconds: response.elapsed,
      language: response.language,
      ...(state.engineVersion !== undefined ? { engineVersion: state.engineVersion } : {}),
      ...(digest !== undefined ? { modelSHA256: digest } : {}),
      ...(hints !== undefined ? { hints } : {}),
    };
  }

  async detectSpeech(audioPath: string, signal?: AbortSignal): Promise<SpeechActivity> {
    checkCancellation(signal);
    const request = { type: "speech", id: randomUUID(), path: audioPath };
    const response = await this.speech.request(
      request,
      request.id,
      this.configuration.speechTimeout,
      undefined,
      signal,
    );
    const { probabilities, frameSeconds, duration, elapsed } = response;
    if (
      !probabilities ||
      frameSeconds === undefined ||
      !(frameSeconds > 0) ||
      duration === undefined ||
      probabilities.length !== Math.ceil(duration / frameSeconds - 1e-6) ||
      elapsed === undefined ||
      !Number.isFinite(elapsed) ||
      elapsed < 0
    ) {
      await this.speech.shutdown();
      throw new InferenceError("invalidResponse", "Speech detection returned an invalid result.");
    }
    return { probabilities, frameSeconds, processingSeconds: elapsed };
  }

  async correct(
    text: string,
    terms: string[],
    language: string,
    systemPrompt: string,
    signal?: AbortSignal,
  ): Promise<ProofInferenceResult> {
    checkCancellation(signal);
    if (
      !text.trim().length ||
      bytes(text) > 24 * 1024 ||
      text.includes("\0") ||
      terms.length > 256 ||
      terms.some((term) => !term.length || bytes(term) > 256 || term.includes("\0")) ||
      terms.reduce((total, term) => total + bytes(term), 0) > 16_384 ||
      !validLanguage(language) ||
      !systemPrompt.trim().length ||
      bytes(systemPrompt) > 4096 ||
      systemPrompt.includes("\0")
    ) {
      throw new InferenceError(
        "invalidRequest",
        "The transcript or dictionary exceeds the Qwen correction limit.",
      );
    }
    const request = { type: "correct", id: randomUUID(), text, terms, language, systemPrompt };
    if (bytes(JSON.stringify(request)) > 64 * 1024)
      throw new InferenceError("invalidRequest", "The encoded correction request exceeds 64 KB.");
    const digest = await this.verifier.verify(this.configuration.proofModel, this.proofPin, signal);
    const response = await this.proof.request(
      request,
      request.id,
      this.configuration.proofTimeout,
      undefined,
      signal,
    );
    if (
      response.text === undefined ||
      !response.text.length ||
      bytes(response.text) > 24 * 1024 ||
      response.text.includes("\0") ||
      response.elapsed === undefined ||
      !Number.isFinite(response.elapsed) ||
      response.elapsed < 0
    ) {
      await this.proof.shutdown();
      throw new InferenceError("invalidResponse", "Qwen returned an invalid correction.");
    }
    const state = this.proof.snapshot();
    const modelSHA256 = digest ?? this.proofManifestSHA256;
    return {
      text: response.text,
      processingSeconds: response.elapsed,
      ...(state.engineVersion !== undefined ? { engineVersion: state.engineVersion } : {}),
      ...(modelSHA256 !== undefined ? { modelSHA256 } : {}),
    };
  }

  async cancel() {
    this.verifier.cancel();
    this.speech.cancel();
    this.proof.cancel();
  }
  async shutdown() {
    await Promise.all([this.verifier.shutdown(), this.speech.shutdown(), this.proof.shutdown()]);
  }
}
