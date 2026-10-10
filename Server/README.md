# Inlay server

The server is an independent TypeScript/Fastify HTTP process that owns models, shared preferences, recordings, and history. Bun manages its dependencies and compiles standalone executables with the runtime included. Native inference helpers run separately. This guide covers model installation and running the server separately.

| Server                | Speech                                           | Proofreading                                |
| --------------------- | ------------------------------------------------ | ------------------------------------------- |
| Apple Silicon macOS   | Parakeet TDT 0.6B v3 / whisper.cpp / Metal       | Qwen3-4B-Instruct-2507 / Swift MLX / 4-bit  |
| Linux x86_64 or ARM64 | Parakeet TDT 0.6B v3 / whisper.cpp / CPU or CUDA | Qwen3-4B-Instruct-2507 / llama.cpp / Q4_K_M |

## Linux

The `Server release packages` workflow builds `inlay-server-linux-x64-cuda.tar.gz` as a workflow artifact on `server-v*` tags and manual dispatch. Download the `inlay-server-linux-x64-cuda` artifact from the Actions run and unzip it to get the tarball and its checksum. The package targets SM 80, 86, 89, 90, 120, and 121; Tesla T4 and RTX 20-series (SM 75) will not run it. The helpers link the CUDA runtime statically and load `libcuda.so.1` from the host driver, which must be 580.95.05 or newer for CUDA 13.0.2. Keep model weights outside the package.

The unit runs as the `inlay` user and listens on `127.0.0.1:8391`. For remote clients, put it behind an HTTPS reverse proxy or bind it to the host's Tailscale IP with a `sudo systemctl edit inlay-server` override (`Environment=INLAY_SERVER_HOST=100.x.y.z`); see [remote access](#remote-access). The `inlay` user must be able to open the host NVIDIA device nodes; if they are group-accessible only, add `inlay` to that group (usually `render` or `video`).

```sh
sudo useradd --system --user-group --home /var/lib/inlay --shell /usr/sbin/nologin inlay
sudo mkdir -p /opt/inlay
sudo install -d -o inlay -g inlay -m 750 /var/lib/inlay /var/lib/inlay/models
sudo install -d -o inlay -g inlay -m 700 /etc/inlay
sudo tar -xzf inlay-server-linux-x64-cuda.tar.gz -C /opt/inlay --strip-components=1 --no-same-owner
```

Place the pinned files at `/var/lib/inlay/models/ggml-parakeet-tdt-0.6b-v3-f16.bin` and `/var/lib/inlay/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` so `inlay` can read them. Install a token of at least 32 characters, with no whitespace, at `/etc/inlay/token`, owned by `inlay`, with mode `0600`:

```sh
printf '%s' 'replace-with-a-token-of-at-least-32-characters' | sudo install -o inlay -g inlay -m 600 /dev/stdin /etc/inlay/token
```

Linux packages (CPU and CUDA) include `inlay-server.service`. Install it outside the package:

```sh
sudo cp /opt/inlay/inlay-server.service /etc/systemd/system/inlay-server.service
sudo systemctl daemon-reload
sudo systemctl enable --now inlay-server
```

To update, stop the service, replace `/opt/inlay`, copy the packaged unit to `/etc/systemd/system` again, run `sudo systemctl daemon-reload`, and start it. Keep local changes in `systemctl edit` overrides so updates do not replace them. Weights and history under `/var/lib/inlay` stay put.

## Models

Run these commands from the repository root. Weights use about 4 GB of disk; runtime memory also includes model state and inference buffers. The server verifies pinned files before loading and keeps models warm. It does not download large weights automatically.

### Parakeet, on either platform

```sh
INLAY_MODEL_DIR="$PWD/.local/models" ./scripts/download-model.sh
```

This installs and verifies `ggml-parakeet-tdt-0.6b-v3-f16.bin`. The URL, revision, and checksum are pinned in `scripts/download-model.sh` and `Sources/InlayCore/SpeechModel.swift`. The server build separately downloads the pinned Silero VAD model. Existing Whisper weights cannot be reused; see [upgrade guidance](../README.md#upgrade-an-existing-installation).

Parakeet automatically recognizes 25 European languages. The language preference guides proofreading and does not force recognition. The preference values are unchanged; choose Automatic for languages without a named proofreading option. The helper exposes no detected language ID. Parakeet takes no recognition vocabulary; dictionary replacements and Qwen hints still apply. Older language preferences remain accepted for proofreading and never block automatic recognition; they do not expand Parakeet’s supported speech languages.

### Qwen on macOS

The MLX directory must contain exactly the six files listed below. Download the pinned revision:

```sh
(
  set -e
  inlay_qwen_dir="$PWD/.local/models/Qwen3-4B-Instruct-2507-MLX-4bit"
  inlay_qwen_url="https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit/resolve/50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
  mkdir -p "$inlay_qwen_dir"
  for file in model.safetensors config.json tokenizer.json tokenizer_config.json generation_config.json chat_template.jinja; do
    curl --fail --location --retry 3 --output "$inlay_qwen_dir/$file" "$inlay_qwen_url/$file"
  done
)
```

`Sources/InlayCore/TextModel.swift` defines the six-file size/hash manifest; the MLX helper verifies it before becoming ready. Use regular files, with no extra files or symlinks in the model directory.

### Qwen on Linux

```sh
mkdir -p .local/models
curl --fail --location --retry 3 \
  --output .local/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/a06e946bb6b655725eafa393f4a9745d460374c9/Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

Expected size: 2,497,281,120 bytes. SHA-256: `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597`. The server verifies both before loading.

## Build

Install Bun 1.4.2 and initialize submodules with `git submodule update --init --recursive`. macOS requires Apple Silicon and full Xcode with its Metal compiler for the MLX helper; the complete client/helper build uses Xcode 26+ and Swift 6.2+. If Metal is missing, run `xcodebuild -downloadComponent MetalToolchain`.

Linux requires Bun, a C/C++ toolchain, CMake, Git, curl, pkg-config, and libcurl development headers. Swift is not required. CUDA builds also need a compatible NVIDIA driver and CUDA toolkit. The [Dockerfile](Dockerfile) provides a pinned Ubuntu 24.04 build environment.

```sh
./scripts/build-server.sh                  # macOS Metal/MLX; Linux CPU
INLAY_CUDA=ON ./scripts/build-server.sh     # Linux with CUDA
```

Output is `build/server`: executable, native helpers, VAD, notices, and resources. Keep the package together; the Mac proofreader requires the adjacent Metal library and bundles. Large model weights and user data live outside it.

To compile only the coordinator, including its correction worker:

```sh
bun install --frozen-lockfile
bun run build:server                     # Native coordinator in build/server
bun run build:server --all               # Mac arm64, Linux x64 and Linux arm64
```

Cross builds live under `build/server-coordinators`. Bun cross-compiles the coordinator; complete installation archives combine it with helpers built on each matching platform. Installed packages need neither Bun nor Node. Full Linux release packages target Ubuntu 24.04 or a compatible glibc/libstdc++ environment; Mac packages require Apple Silicon and macOS 14+. Linux x64 coordinators use Bun's baseline CPU target. Native helper CPU/CUDA compatibility remains determined by its CMake build flags.

The release workflow produces complete platform tarballs and SHA-256 checksums, including the Linux CUDA package above. Extract a package, retain its `server` directory together, install the pinned model weights separately, then use the arguments below. Developer ID distribution still requires signing/notarization credentials; the draft Mac build is ad-hoc signed with Bun's executable entitlements.

The archive lock uses Bun FFI to call libc `flock`, matching the reference Swift server. The lock is held for the server process lifetime. A running Swift server and Bun server must never share a data directory.

`INLAY_BUILD_JOBS` controls build concurrency. For another CPU/GPU host, use `INLAY_NATIVE=OFF` and set `INLAY_CUDA_ARCHITECTURES` for the destination GPU. CPU support is useful for portable builds; validate CUDA support, memory, and dictation latency on the selected host.

## Run

From the repository root, with the models installed above:

```sh
./build/server/inlay-server \
  --host 127.0.0.1 --port 8391 \
  --data-dir "$PWD/.local/server" \
  --speech-helper "$PWD/build/server/helpers/inlay-engine" \
  --speech-model "$PWD/.local/models/ggml-parakeet-tdt-0.6b-v3-f16.bin" \
  --vad-model "$PWD/build/server/resources/silero-vad.bin" \
  --proof-helper "$PWD/build/server/helpers/inlay-text-engine" \
  --proof-model "$PWD/.local/models/Qwen3-4B-Instruct-2507-MLX-4bit"
```

On Linux, replace the last path with the GGUF file. Add `--dev` for a development label in health responses. If using the packaged distribution elsewhere, point helper/resource paths at that package and choose durable model/data paths.

Check `curl http://localhost:8391/v1/health`; HTTP reachability alone does not mean inference is available. The `ready` field means the server can accept a recording, including while other recordings are uploading or processing. Finished uploads queue for serial inference; model runtime fields separately report whether helpers are loaded. Quitting a client does not stop this process. Use launchd, systemd, or container supervision for boot/restart behavior; the scripts do not install a service.

For server-only development alongside an installed Inlay instance, use `--port 8392 --data-dir "$PWD/.local/typescript-server" --dev` with your helper/model arguments. Start the executable directly or use `bun run dev:server` with those arguments. The client dev runner starts the app and defaults to port 8391; avoid it when preserving a running installation.

| Argument                            | Environment variable                               |
| ----------------------------------- | -------------------------------------------------- |
| `--host`, `--port`                  | `INLAY_SERVER_HOST`, `INLAY_SERVER_PORT`           |
| `--data-dir`, `--token-file`        | `INLAY_SERVER_DATA_DIR`, `INLAY_SERVER_TOKEN_FILE` |
| `--speech-helper`, `--speech-model` | `INLAY_ENGINE_PATH`, `INLAY_SPEECH_MODEL`          |
| `--vad-model`                       | `INLAY_VAD_PATH`                                   |
| `--proof-helper`, `--proof-model`   | `INLAY_TEXT_ENGINE_PATH`, `INLAY_TEXT_MODEL`       |
| `--dev`                             | `INLAY_DEV=1`                                      |
| `--stream-speech`                   | `INLAY_STREAM_SPEECH=1`                            |

`--stream-speech` recognizes speech while a take is still uploading. Silero scores each new second of audio; once at least 8 seconds are waiting, the server cuts at the next pause that is at least as long as the speaker's median pause in that stretch, and Parakeet recognizes that piece with 4 seconds of audio before and 2 seconds after it as context. On release, only the audio after the last cut remains, so the wait after a long take drops from a whole-take pass (about 10–13 seconds for 80 seconds of speech on a 6-core CPU) to about a second. Pieces lose the whole-take context, so expect slightly more misheard words, mostly names and jargon. Speech detection costs about 4 ms of CPU per second of audio. Leave it off for best accuracy, or when the server is fast enough that whole-take recognition already feels instant.

The dev runner fixes its host to loopback and defaults to port 8391, `.local/server` for data, and `.local/server.log` for logs. Set `INLAY_SPEECH_MODEL` and `INLAY_TEXT_MODEL` when using the paths above. Without those overrides, macOS searches `~/Library/Application Support/Inlay/Models/ggml-parakeet-tdt-0.6b-v3-f16.bin` and `~/.inlay/models/Qwen3-4B-Instruct-2507-MLX-4bit`. Existing installations can retain their archive, token file, and Qwen model location with the arguments or overrides above, but must install the Parakeet weights; see [upgrade guidance](../README.md#upgrade-an-existing-installation).

## Remote access

Bind to a reachable address and pass `--token-file /absolute/path/to/token`. Nonloopback listeners require a token of at least 32 characters with no internal whitespace. In the Mac app, enter the endpoint and token under **This Mac**; tokens are stored in Keychain.

- Use an HTTPS reverse proxy for hosted servers and hostnames, including Tailscale MagicDNS names. The runner itself serves HTTP.
- HTTP is accepted for localhost and literal Tailscale IPs in `100.64.0.0/10` or `fd7a:115c:a1e0::/48` on your connected tailnet. Inlay checks the address range, not routing; use HTTPS if that private route cannot be assured.
- Ordinary LAN IPs require HTTPS. Endpoints cannot contain credentials, queries, or fragments. Credential-bearing redirects are not followed.

Keep the data directory on persistent storage and back it up. Only one runner can own it. See [storage](../docs/architecture.md#storage) and the [HTTP API](../docs/client-server-contract.md).

## Containers

Build from the repository root with initialized submodules:

```sh
docker build -f Server/Dockerfile --target cpu -t inlay-server:cpu .
docker build -f Server/Dockerfile --target cuda -t inlay-server:cuda .
```

`--target cpu` is the portable no-GPU image, and it is what an unqualified `docker build` selects. `--target cuda` is the container fallback when the host cannot place a matching CUDA 13.0.2 runtime next to the binary: that image copies the toolkit and sets `LD_LIBRARY_PATH`. `CUDA_ARCHITECTURES`, `CUDA_IMAGE`, `BUN_IMAGE`, `UBUNTU_IMAGE`, and `BUILD_JOBS` are build arguments. Choose CUDA architectures/toolkit/driver versions for your GPU. GPU containers require NVIDIA Container Toolkit and `--gpus all`; Linux containers on a Mac do not have Metal access.

Mount a directory containing the Parakeet `.bin` and Qwen `.gguf` files, plus a token file:

```sh
docker run --rm --name inlay-server \
  -p 127.0.0.1:8391:8391 \
  --mount type=volume,source=inlay-data,target=/data \
  --mount type=bind,source=/absolute/path/to/models,target=/models,readonly \
  --mount type=bind,source=/absolute/path/to/token,target=/run/secrets/inlay-token,readonly \
  inlay-server:cpu
```

The example is the no-GPU image and exposes only host loopback; use the remote-access setup above for clients on other machines. For the container fallback, use `inlay-server:cuda` and add `--gpus all`. The container runs as UID 10001, which must be able to read model/token files and write `/data`. The named volume preserves history across container replacement.

## Development and verification

From the repository root:

```sh
bun install --frozen-lockfile
bun run dev
```

The source server hot reloads, listens on `0.0.0.0:8392`, and keeps data and its automatically created private token under `.local/dev-server`. On the same Mac, open [server health](http://localhost:8392/v1/health). Build helpers once with `./scripts/build-server.sh` and set `INLAY_SPEECH_MODEL` and `INLAY_TEXT_MODEL` to enable dictation. Missing assets leave the server running with `ready: false`; HTTP reachability does not establish working inference. CLI arguments and the environment variables above override development defaults, except that inherited `INLAY_SERVER_DATA_DIR` is ignored. Use `--data-dir` to explicitly select another development archive.

```sh
bun run fmt
bun run fmt:check
bun run lint
bun run check
```

`check` runs strict TypeScript checking and verifies generated TypeScript bindings against OpenAPI. `lint` runs type-aware correctness rules with zero warnings, including promise handling and a ban on explicit `any`. `fmt` and `fmt:check` cover first-party TypeScript, JSON, YAML, and Markdown while excluding vendor/build/data directories. On macOS with Swift 6.2+, also run `bun run generate:api --check` and `swift build --force-resolved-versions` when changing Swift or the API.

Automated tests and test harnesses are not allowed. Use the native **Inlay Dev** app through computer use to verify changed behavior, including real dictation, progress, preferences, history, and delivery. The API rejects browser-origin requests and has no web UI. Follow [the development guide](../docs/development.md); report missing model/device access as unverified behavior.
