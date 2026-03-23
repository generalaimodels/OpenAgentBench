

# Universal Agent SDK Architecture: SOTA Plan for Omnipresent Software-Ecosystem Connectivity

---

## 1. Architectural Thesis: The SDK as a Universal Typed Connector Fabric

### 1.1 Problem Statement

Production agentic systems must operate across the **complete software ecosystem surface** — not within a single API or a curated tool catalog, but across arbitrary protocols, arbitrary applications, arbitrary operating systems, arbitrary authentication domains, and arbitrary interaction modalities (programmatic API, terminal CLI, graphical UI, browser DOM, IDE extension, file system, network socket). The fundamental engineering challenge is:

$$
\text{Connectivity}(S) = \bigcup_{p \in \mathcal{P}} \bigcup_{a \in \mathcal{A}} \bigcup_{e \in \mathcal{E}} \text{Interface}(p, a, e)
$$

where $\mathcal{P}$ is the set of all protocols (HTTP/S, gRPC, JSON-RPC, WebSocket, TCP, SSH/SCP, MCP stdio/SSE, IPC, serial), $\mathcal{A}$ is the set of all applications (Google Workspace, IDEs, terminals, browsers, desktop applications, databases, cloud consoles), and $\mathcal{E}$ is the set of all execution environments (Linux, macOS, Windows, iOS, Android, embedded, serverless, browser runtime, container).

A naive approach — building one adapter per $(p, a, e)$ triple — produces combinatorial explosion:

$$
|\text{Adapters}_{\text{naive}}| = |\mathcal{P}| \times |\mathcal{A}| \times |\mathcal{E}| \quad \text{(intractable)}
$$

The SOTA architecture instead decomposes connectivity into **orthogonal, composable layers** with typed contracts at every boundary, reducing the problem to:

$$
|\text{Adapters}_{\text{layered}}| = |\mathcal{P}| + |\mathcal{A}| + |\mathcal{E}| \quad \text{(linear, maintainable)}
$$

### 1.2 Fundamental Design Invariants

| Invariant | Statement | Enforcement |
|---|---|---|
| **INV-U1** | Every external system interaction passes through a typed connector contract | Compile-time type checking + runtime schema validation |
| **INV-U2** | No protocol-specific logic leaks above the transport abstraction layer | Static analysis dependency rule in CI |
| **INV-U3** | Every state-mutating operation carries an idempotency key | SDK-level enforcement before wire serialization |
| **INV-U4** | Every connector exposes health, latency, and error-rate observability | Middleware chain injection; no bypass path |
| **INV-U5** | Every authentication credential is scoped to least privilege and rotatable | Credential manager enforces scope ceiling |
| **INV-U6** | Every UI/browser/desktop interaction is observable, replayable, and reversible | Action log with screenshot/DOM-snapshot provenance |
| **INV-U7** | The SDK operates under bounded resource budgets (memory, CPU, token, cost) | Resource governor enforces hard limits per operation |

### 1.3 SDK Mega-Architecture: Seven-Layer Universal Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│  L6: AGENT ORCHESTRATION                                                │
│  Plan → Decompose → Route → Act → Verify → Critique → Repair → Commit  │
├─────────────────────────────────────────────────────────────────────────┤
│  L5: DOMAIN CONNECTORS                                                  │
│  Google Workspace │ IDEs │ Terminals │ Browsers │ Databases │ Cloud     │
├─────────────────────────────────────────────────────────────────────────┤
│  L4: INTERACTION MODALITIES                                             │
│  API (REST/gRPC) │ CLI (Terminal) │ GUI (Desktop) │ Browser (DOM/CDP)   │
├─────────────────────────────────────────────────────────────────────────┤
│  L3: CROSS-CUTTING SERVICES                                            │
│  Auth │ Memory │ Context │ Observability │ Rate Limiting │ Caching       │
├─────────────────────────────────────────────────────────────────────────┤
│  L2: PROTOCOL ADAPTERS                                                  │
│  HTTP/S │ gRPC │ JSON-RPC │ WebSocket │ TCP │ SSH/SCP │ MCP │ IPC       │
├─────────────────────────────────────────────────────────────────────────┤
│  L1: TRANSPORT ABSTRACTION                                              │
│  Connection Pool │ TLS │ Framing │ Compression │ Multiplexing           │
├─────────────────────────────────────────────────────────────────────────┤
│  L0: OS / RUNTIME ABSTRACTION                                          │
│  Linux │ macOS │ Windows │ iOS │ Android │ Browser │ Embedded            │
└─────────────────────────────────────────────────────────────────────────┘
```

**Dependency rule:** Layer $L_i$ may depend only on $L_j$ where $j < i$. No lateral dependencies within a layer except through explicit internal interfaces.

---

## 2. L0–L1: OS Runtime and Transport Abstraction

### 2.1 OS Abstraction Layer (L0)

The SDK must execute on any operating system and runtime. L0 provides a **platform-neutral interface** to OS primitives:

```
TYPE OsPlatform:
    // File System
    FUNC read_file(path: OsPath, options: ReadOptions) → Result<Bytes, OsError>
    FUNC write_file(path: OsPath, content: Bytes, options: WriteOptions) → Result<void, OsError>
    FUNC list_directory(path: OsPath, filter: GlobPattern) → Result<List<DirEntry>, OsError>
    FUNC watch_path(path: OsPath, events: Set<FsEvent>) → AsyncStream<FsEvent, OsError>

    // Process Management
    FUNC spawn_process(cmd: Command, options: SpawnOptions) → Result<ProcessHandle, OsError>
    FUNC kill_process(handle: ProcessHandle, signal: Signal) → Result<void, OsError>
    FUNC pipe_io(handle: ProcessHandle) → (AsyncWriter, AsyncReader, AsyncReader)

    // Network
    FUNC tcp_connect(addr: SocketAddr, timeout: Duration) → Result<TcpStream, OsError>
    FUNC tcp_listen(addr: SocketAddr, backlog: int) → Result<TcpListener, OsError>
    FUNC udp_socket(addr: SocketAddr) → Result<UdpSocket, OsError>
    FUNC unix_socket(path: OsPath) → Result<UnixStream, OsError>  // Unix-only

    // System Information
    FUNC hostname() → string
    FUNC environment_vars() → Map<string, string>
    FUNC current_user() → UserInfo
    FUNC os_type() → OsType  // LINUX | MACOS | WINDOWS | IOS | ANDROID | BROWSER | EMBEDDED

    // Clipboard / Desktop (when available)
    FUNC clipboard_read() → Result<ClipboardContent, OsError>
    FUNC clipboard_write(content: ClipboardContent) → Result<void, OsError>
    FUNC screen_capture(region: ScreenRegion?) → Result<ImageBuffer, OsError>
```

**Platform detection and capability matrix:**

| Capability | Linux | macOS | Windows | iOS | Android | Browser | Embedded |
|---|---|---|---|---|---|---|---|
| TCP sockets | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ (fetch only) | Partial |
| Unix domain sockets | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Process spawn | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| File system | ✓ | ✓ | ✓ | Sandboxed | Sandboxed | ✗ | Partial |
| Screen capture | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Clipboard | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (async) | ✗ |
| Named pipes | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |

The SDK queries capabilities at initialization and **degrades gracefully** when a capability is unavailable, rather than failing at compile time.

### 2.2 Transport Abstraction Layer (L1)

L1 normalizes all network communication into a **unified bidirectional byte-stream interface** with lifecycle management:

```
TYPE TransportStream:
    FUNC send(data: Bytes, deadline: Duration) → Result<int, TransportError>
    FUNC recv(max_bytes: int, deadline: Duration) → Result<Bytes, TransportError>
    FUNC close(reason: CloseReason) → Result<void, TransportError>
    FUNC is_alive() → bool
    PROPERTY local_addr: SocketAddr?
    PROPERTY remote_addr: SocketAddr?
    PROPERTY tls_info: TlsInfo?
    PROPERTY latency_estimate: Duration
```

**Connection Pool at L1:**

$$
P^* = \left\lceil \frac{\lambda \cdot \bar{s}}{1 - \epsilon} \right\rceil, \quad P_{\min} \leq |C_{\text{pool}}| \leq P_{\max}
$$

where $\lambda$ is request rate, $\bar{s}$ is mean service time, and $\epsilon$ is headroom. The pool is **per-endpoint**, with idle eviction at interval $T_{\text{idle}}$ and health probing at interval $T_{\text{health}}$.

---

## 3. L2: Universal Protocol Adapter Layer

### 3.1 Protocol Adapter Contract

Every protocol adapter implements the same typed interface, enabling the upper layers to be **protocol-agnostic**:

```
TYPE ProtocolAdapter:
    FUNC connect(endpoint: Endpoint, config: ProtocolConfig) → Result<ProtocolSession, ProtocolError>
    FUNC send_request(session: ProtocolSession, request: UniversalRequest) → Result<UniversalResponse, ProtocolError>
    FUNC send_streaming(session: ProtocolSession, request: UniversalRequest) → AsyncStream<UniversalResponseChunk, ProtocolError>
    FUNC listen(session: ProtocolSession, subscription: Subscription) → AsyncStream<UniversalEvent, ProtocolError>
    FUNC close(session: ProtocolSession) → Result<void, ProtocolError>
    FUNC capabilities() → ProtocolCapabilities

TYPE UniversalRequest:
    method: string                    // Unified method identifier
    headers: Map<string, string>      // Protocol-mapped headers/metadata
    body: Bytes                       // Serialized payload
    idempotency_key: string?          // For safe retries
    deadline: Duration                // Absolute deadline
    trace_context: TraceContext       // Distributed tracing propagation

TYPE UniversalResponse:
    status: UniversalStatus           // Normalized status
    headers: Map<string, string>
    body: Bytes
    latency: Duration
    protocol_metadata: ProtocolMetadata  // Protocol-specific details
```

### 3.2 Protocol Adapter Registry

| Protocol | Adapter ID | Transport | Serialization | Streaming | Bidirectional | Auth Methods |
|---|---|---|---|---|---|---|
| **HTTP/1.1** | `http1` | TCP + TLS | JSON, form, multipart | Chunked transfer | No | Bearer, Basic, OAuth2, API key |
| **HTTP/2** | `http2` | TCP + TLS | Binary frames | Server push, SSE | Limited | Same as HTTP/1.1 |
| **gRPC** | `grpc` | HTTP/2 | Protobuf | Server/client/bidi | Yes | mTLS, token, per-RPC credentials |
| **JSON-RPC 2.0** | `jsonrpc` | HTTP, WebSocket, stdio | JSON | Via transport | Via transport | Bearer, custom |
| **WebSocket** | `ws` | TCP + TLS | Binary/text frames | Native | Yes | Cookie, Bearer, ticket |
| **TCP raw** | `tcp` | TCP (+optional TLS) | Custom framing | Native | Yes | TLS client cert, application-level |
| **SSH/SCP** | `ssh` | TCP | SSH protocol | Channel multiplexing | Yes | Key-based, password, agent |
| **MCP (stdio)** | `mcp_stdio` | stdin/stdout pipes | JSON-RPC | No | Half-duplex | Process-level |
| **MCP (SSE)** | `mcp_sse` | HTTP/S | JSON-RPC + SSE | Server → client | No | Bearer |
| **MCP (Streamable HTTP)** | `mcp_http` | HTTP/S | JSON-RPC | Bidirectional | Yes | Bearer, OAuth2 |
| **Unix Socket** | `unix` | UDS | Custom | Native | Yes | Filesystem permissions |
| **Named Pipe** | `pipe` | OS pipe | Custom | Native | Yes | OS ACL |

### 3.3 Protocol Selection Algorithm

---

**PSEUDO-ALGORITHM 3.1: Protocol Selection**

```
PROCEDURE SelectProtocol(
    endpoint: Endpoint,
    requirements: ConnectivityRequirements,
    platform: OsPlatform
) → ProtocolAdapter:

    // Step 1: Determine available protocols for this endpoint
    available ← endpoint.advertised_protocols
    IF available IS EMPTY THEN
        available ← InferFromEndpoint(endpoint)
        // URL scheme → protocol: grpc:// → gRPC, ws:// → WebSocket
        // Port hints: 443 → HTTPS, 22 → SSH, etc.
    
    // Step 2: Filter by platform capabilities
    FOR EACH proto IN available:
        IF NOT platform.supports(proto.required_capabilities) THEN
            REMOVE proto FROM available
    
    // Step 3: Filter by requirements
    FOR EACH proto IN available:
        IF requirements.needs_streaming AND NOT proto.supports_streaming THEN
            REMOVE proto
        IF requirements.needs_bidirectional AND NOT proto.supports_bidirectional THEN
            REMOVE proto
        IF requirements.max_latency < proto.overhead_estimate THEN
            REMOVE proto
    
    // Step 4: Rank remaining by preference
    scored ← []
    FOR EACH proto IN available:
        score ← 0
        score += w_perf * proto.performance_score        // gRPC > HTTP/2 > HTTP/1.1
        score += w_type * proto.type_safety_score         // Protobuf > JSON Schema > raw JSON
        score += w_eco  * proto.ecosystem_support_score   // Tooling, debugging, monitoring
        score += w_comp * proto.compression_score         // Binary > text
        score -= w_over * proto.connection_overhead       // Handshake cost
        APPEND (proto, score) TO scored
    
    SortDescending(scored, key = score)
    
    IF scored IS EMPTY THEN
        RAISE Unavailable("No compatible protocol found for endpoint")
    
    RETURN scored[0].proto.create_adapter()
```

---

### 3.4 Protocol Bridging for Heterogeneous Environments

When the agent must bridge between two systems using incompatible protocols (e.g., invoking a gRPC service from a browser environment that only supports HTTP/fetch), the SDK deploys a **local protocol bridge**:

$$
\text{Client} \xrightarrow{\text{HTTP/JSON}} \text{Local Bridge} \xrightarrow{\text{gRPC/Protobuf}} \text{Target Service}
$$

The bridge runs as a sidecar process (on desktop/server) or a WebAssembly module (in browser). It handles protocol translation, serialization conversion, and TLS termination.

---

## 4. L3: Cross-Cutting Services

### 4.1 Universal Authentication Manager

The authentication manager supports **every authentication mechanism** encountered in the software ecosystem, unified behind a single credential resolution interface:

```
TYPE AuthManager:
    FUNC resolve_credential(
        target: Endpoint,
        scopes: Set<AuthScope>,
        auth_type: AuthType?
    ) → Result<Credential, AuthError>

    FUNC refresh(credential: Credential) → Result<Credential, AuthError>
    FUNC revoke(credential: Credential) → Result<void, AuthError>
    FUNC register_provider(provider: AuthProvider) → void
    FUNC list_providers() → List<AuthProviderInfo>

TYPE AuthType:
    OAUTH2_AUTHORIZATION_CODE    // Google, Microsoft, GitHub, etc.
    OAUTH2_CLIENT_CREDENTIALS    // Service-to-service
    OAUTH2_DEVICE_FLOW           // CLI/headless
    API_KEY                      // Static API keys
    BEARER_TOKEN                 // JWT, opaque tokens
    BASIC_AUTH                   // Username/password
    MTLS                         // Mutual TLS certificates
    SSH_KEY                      // RSA/Ed25519 key pairs
    KERBEROS                     // Enterprise SSO
    SAML                         // Enterprise federation
    BROWSER_SESSION              // Cookie-based via browser automation
    DESKTOP_KEYCHAIN             // OS credential store (Keychain, Credential Manager, Secret Service)
    CLOUD_IAM                    // AWS IAM, GCP IAM, Azure AD
    CUSTOM                       // Plugin-based
```

---

**PSEUDO-ALGORITHM 4.1: Universal Credential Resolution**

```
PROCEDURE ResolveCredential(
    target: Endpoint,
    scopes: Set<AuthScope>,
    auth_manager: AuthManager
) → Result<Credential, AuthError>:

    // Step 1: Check credential cache
    cached ← auth_manager.cache.get(target, scopes)
    IF cached IS NOT NULL AND NOT cached.is_expired(buffer = 60s) THEN
        RETURN Ok(cached)
    
    // Step 2: Determine auth type from target metadata
    auth_type ← target.auth_requirements.preferred_type
    IF auth_type IS NULL THEN
        auth_type ← auth_manager.infer_auth_type(target)
    
    // Step 3: Find matching provider
    provider ← auth_manager.find_provider(auth_type, target.domain)
    IF provider IS NULL THEN
        RETURN Err(AuthError("No auth provider for: " + auth_type))
    
    // Step 4: Acquire credential
    credential ← MATCH auth_type:
        OAUTH2_AUTHORIZATION_CODE →
            // Check for stored refresh token
            refresh_token ← auth_manager.secure_store.get_refresh_token(target.domain, scopes)
            IF refresh_token IS NOT NULL THEN
                provider.refresh(refresh_token)
            ELSE
                // Interactive flow: open browser, await callback
                auth_url ← provider.build_auth_url(scopes, redirect_uri, state, pkce_challenge)
                // Signal to agent orchestrator: human interaction required
                AWAIT agent.request_human_action(OpenBrowser(auth_url))
                code ← AWAIT auth_manager.callback_server.await_code(state, timeout = 120s)
                provider.exchange_code(code, pkce_verifier)
        
        OAUTH2_DEVICE_FLOW →
            device_code_response ← provider.request_device_code(scopes)
            AWAIT agent.request_human_action(DisplayCode(device_code_response.user_code, device_code_response.verification_uri))
            provider.poll_for_token(device_code_response, interval = 5s, timeout = 300s)
        
        API_KEY →
            auth_manager.secure_store.get_api_key(target.domain)
        
        SSH_KEY →
            key_path ← auth_manager.secure_store.get_ssh_key(target.host)
            SshCredential(key_path, passphrase = auth_manager.secure_store.get_passphrase(key_path))
        
        BROWSER_SESSION →
            // Delegate to browser automation layer to perform login
            session_cookies ← AWAIT browser_connector.perform_login(
                target.login_url,
                credentials = auth_manager.secure_store.get_login_creds(target.domain),
                mfa_handler = auth_manager.mfa_provider
            )
            BrowserSessionCredential(session_cookies)
        
        DESKTOP_KEYCHAIN →
            auth_manager.os_keychain.get(target.service_name, target.account)
        
        CLOUD_IAM →
            provider.assume_role(target.iam_role, session_name = agent.session_id)
    
    // Step 5: Validate credential
    IF credential IS Err THEN RETURN credential
    
    // Step 6: Scope verification — enforce least privilege
    IF NOT credential.scopes.is_subset_of(scopes) THEN
        LOG warning "Credential has excess scopes; applying scope ceiling"
        credential ← credential.narrow_to(scopes)
    
    // Step 7: Cache with TTL
    auth_manager.cache.put(target, scopes, credential, ttl = credential.expires_in * 0.8)
    
    // Step 8: Emit telemetry
    EMIT metric: sdk.auth.resolve {
        target_domain, auth_type, cached = false, latency_ms
    }
    
    RETURN Ok(credential)
```

---

### 4.2 Secure Credential Store

All credentials are stored in a **tiered secure storage** system:

| Storage Tier | Location | Encryption | Use Case |
|---|---|---|---|
| **In-Memory** | Process heap (guarded page) | Memory encryption where available | Active session tokens |
| **OS Keychain** | macOS Keychain, Windows Credential Manager, Linux Secret Service | OS-managed | Long-lived credentials (API keys, SSH keys) |
| **Encrypted File** | `~/.agent_sdk/credentials.enc` | AES-256-GCM with PBKDF2-derived key | Portable credential backup |
| **Hardware Security Module** | YubiKey, TPM, Secure Enclave | Hardware-bound | High-security deployments |
| **Vault/KMS** | HashiCorp Vault, AWS KMS, GCP KMS | Service-managed | Enterprise/cloud deployments |

**Credential lifecycle invariant:** Every credential has a **maximum lifetime** $T_{\max}$ and is automatically rotated or revoked upon expiry:

$$
\text{valid}(c, t) \iff t < c.t_{\text{issued}} + \min(c.T_{\text{expires}}, T_{\max}) \land c.\text{revoked} = \text{false}
$$

### 4.3 Unified Rate Limiting and Cost Governor

The SDK enforces **global and per-service rate limits** to prevent abuse, cost overruns, and dependency exhaustion:

```
TYPE CostGovernor:
    FUNC check_budget(operation: OperationSpec) → Result<BudgetApproval, BudgetError>
    FUNC record_cost(operation: OperationId, cost: CostRecord) → void
    FUNC get_usage(scope: CostScope, period: TimePeriod) → UsageReport
    FUNC set_limit(scope: CostScope, limit: CostLimit) → void

TYPE CostRecord:
    api_calls: int
    tokens_input: int
    tokens_output: int
    compute_seconds: float
    storage_bytes: int
    network_bytes: int
    monetary_cost_usd: float
```

**Cost enforcement at the SDK boundary:**

$$
C_{\text{accumulated}}(t) + C_{\text{estimated}}(\text{op}) \leq C_{\text{budget}}(\text{scope}, \text{period})
$$

If the inequality is violated, the operation is **rejected pre-flight** with a `ResourceExhausted` error containing the budget status and estimated time until budget replenishment.

---

## 5. L4: Interaction Modality Layer

### 5.1 Modality Taxonomy

The SDK must interact with external systems through **four distinct interaction modalities**, each requiring fundamentally different abstractions:

| Modality | Mechanism | Latency Profile | Observability | Reversibility |
|---|---|---|---|---|
| **API (Programmatic)** | HTTP/gRPC/SDK calls with typed schemas | Low (ms–s) | Full (request/response logged) | Depends on API design |
| **CLI (Terminal)** | Shell command execution, stdin/stdout/stderr | Low–Medium (ms–min) | Full (command + output captured) | Command-dependent |
| **GUI (Desktop)** | Screen reading, mouse/keyboard simulation | High (100ms–s per action) | Screenshot + action log | Undo where available |
| **Browser (Web)** | DOM manipulation, CDP, page navigation | Medium (50ms–s) | DOM snapshot + network log | Navigation back + undo |

### 5.2 API Interaction Engine

The API engine handles all **programmatic integrations** — REST, gRPC, GraphQL, SOAP, custom wire protocols:

```
TYPE ApiEngine:
    FUNC call(
        endpoint: ApiEndpoint,
        method: string,
        params: TypedParams,
        options: ApiCallOptions
    ) → Result<TypedResponse, ApiError>

    FUNC call_streaming(
        endpoint: ApiEndpoint,
        method: string,
        params: TypedParams
    ) → AsyncStream<TypedResponseChunk, ApiError>

    FUNC discover_api(base_url: URL) → Result<ApiSpec, ApiError>
    // Auto-detect: OpenAPI, gRPC reflection, GraphQL introspection

    FUNC generate_client(spec: ApiSpec) → DynamicApiClient
    // Runtime code generation for discovered APIs
```

**Dynamic API Client Generation:**

When the SDK encounters an undiscovered API, it attempts **automatic schema discovery**:

---

**PSEUDO-ALGORITHM 5.1: Automatic API Discovery**

```
PROCEDURE DiscoverApi(base_url: URL) → Result<ApiSpec, DiscoveryError>:

    // Strategy 1: OpenAPI/Swagger
    FOR EACH path IN ["/openapi.json", "/swagger.json", "/api-docs", "/.well-known/openapi"]:
        response ← http_get(base_url + path, timeout = 5s)
        IF response.status = 200 AND response.content_type contains "json" THEN
            spec ← parse_openapi(response.body)
            IF spec IS valid THEN RETURN Ok(ApiSpec.from_openapi(spec))
    
    // Strategy 2: gRPC reflection
    IF base_url.scheme IN ["grpc", "grpcs", "h2"] THEN
        reflection_client ← grpc_connect(base_url)
        services ← reflection_client.list_services()
        IF services IS NOT EMPTY THEN
            descriptors ← [reflection_client.file_descriptor(s) FOR s IN services]
            RETURN Ok(ApiSpec.from_grpc_reflection(descriptors))
    
    // Strategy 3: GraphQL introspection
    response ← http_post(base_url + "/graphql", body = INTROSPECTION_QUERY, timeout = 5s)
    IF response.status = 200 AND response.body contains "__schema" THEN
        schema ← parse_graphql_schema(response.body)
        RETURN Ok(ApiSpec.from_graphql(schema))
    
    // Strategy 4: MCP discovery
    IF endpoint.supports_mcp THEN
        mcp_session ← mcp_initialize(base_url)
        tools ← mcp_session.list_tools()
        resources ← mcp_session.list_resources()
        RETURN Ok(ApiSpec.from_mcp(tools, resources))
    
    // Strategy 5: HATEOAS / link traversal
    response ← http_get(base_url, accept = "application/json", timeout = 5s)
    IF response.body contains "_links" OR response.body contains "links" THEN
        links ← extract_hypermedia_links(response.body)
        spec ← crawl_api_surface(base_url, links, max_depth = 3, max_endpoints = 100)
        RETURN Ok(spec)
    
    RETURN Err(DiscoveryError("No discoverable API schema at: " + base_url))
```

---

### 5.3 Terminal Interaction Engine

The terminal engine provides **programmatic control over any terminal session** on any operating system:

```
TYPE TerminalEngine:
    // Session management
    FUNC create_session(config: TerminalConfig) → Result<TerminalSession, TerminalError>
    FUNC attach_session(session_id: SessionId) → Result<TerminalSession, TerminalError>
    FUNC list_sessions() → List<SessionInfo>

    // Command execution
    FUNC execute(session: TerminalSession, command: Command) → Result<CommandResult, TerminalError>
    FUNC execute_interactive(session: TerminalSession, command: Command) → InteractiveHandle
    FUNC execute_script(session: TerminalSession, script: Script, interpreter: Interpreter) → Result<ScriptResult, TerminalError>

    // Environment
    FUNC get_env(session: TerminalSession) → Map<string, string>
    FUNC set_env(session: TerminalSession, vars: Map<string, string>) → Result<void, TerminalError>
    FUNC get_working_dir(session: TerminalSession) → OsPath
    FUNC set_working_dir(session: TerminalSession, path: OsPath) → Result<void, TerminalError>

    // Observation
    FUNC read_output(session: TerminalSession, since: Cursor?) → TerminalOutput
    FUNC await_pattern(session: TerminalSession, pattern: RegexPattern, timeout: Duration) → Result<Match, TerminalError>
    FUNC get_shell_state(session: TerminalSession) → ShellState

TYPE TerminalConfig:
    shell: ShellType            // BASH | ZSH | FISH | POWERSHELL | CMD | SH | CUSTOM
    os: OsType                  // Auto-detected
    ssh_target: SshTarget?      // For remote terminals
    container: ContainerId?     // For containerized execution
    working_dir: OsPath?
    env_overrides: Map<string, string>?
    timeout_default: Duration
    capture_mode: CaptureMode   // FULL | STDOUT_ONLY | STRUCTURED
```

---

**PSEUDO-ALGORITHM 5.2: Safe Command Execution with Observation**

```
PROCEDURE ExecuteCommand(
    session: TerminalSession,
    command: Command,
    options: ExecutionOptions
) → Result<CommandResult, TerminalError>:

    // Step 1: Pre-execution validation
    IF options.safety_level ≥ CAUTIOUS THEN
        risk ← AssessCommandRisk(command)
        // Risk factors: rm -rf, DROP TABLE, shutdown, format, etc.
        IF risk.level = DESTRUCTIVE THEN
            IF NOT options.allow_destructive THEN
                RETURN Err(TerminalError("Destructive command blocked: " + risk.reason))
            // Require human approval
            approval ← AWAIT request_human_approval(
                action = "Execute destructive command",
                details = {command: command, risk: risk}
            )
            IF NOT approval.granted THEN
                RETURN Err(TerminalError("Command rejected by human reviewer"))
    
    // Step 2: Command preparation
    sanitized_command ← SanitizeCommand(command, session.shell)
    // Escape injection vectors, normalize path separators, set timeout wrapper
    
    wrapped_command ← WrapWithObservation(sanitized_command, session.shell)
    // Wraps command to capture: exit code, timing, stdout, stderr
    // e.g., on bash: `{ time command ; } 2>&1 ; echo "EXIT:$?"`
    
    // Step 3: Execute with timeout
    process ← session.spawn(wrapped_command, timeout = options.timeout)
    
    stdout_buffer ← AsyncBuffer()
    stderr_buffer ← AsyncBuffer()
    
    // Stream output with backpressure
    CONCURRENT:
        TASK stream_stdout:
            FOR EACH chunk IN process.stdout:
                stdout_buffer.write(chunk)
                IF stdout_buffer.size > options.max_output_bytes THEN
                    process.signal(SIGTERM)
                    RETURN Err(TerminalError("Output exceeded maximum size"))
        
        TASK stream_stderr:
            FOR EACH chunk IN process.stderr:
                stderr_buffer.write(chunk)
    
    exit_code ← AWAIT process.wait(timeout = options.timeout)
    
    // Step 4: Parse result
    result ← CommandResult(
        exit_code = exit_code,
        stdout = stdout_buffer.to_string(),
        stderr = stderr_buffer.to_string(),
        duration = process.elapsed(),
        working_dir = session.cwd(),
        command = command  // For provenance
    )
    
    // Step 5: Post-execution observation
    IF options.observe_filesystem_changes THEN
        result.fs_changes ← session.diff_filesystem(pre_snapshot, post_snapshot)
    
    // Step 6: Emit telemetry
    EMIT span: sdk.terminal.execute {
        shell = session.shell, exit_code, duration_ms, output_bytes = stdout_buffer.size
    }
    
    RETURN Ok(result)
```

---

### 5.4 Browser Interaction Engine

The browser engine provides **full control over web browsers** for navigating, inspecting, and interacting with web applications:

```
TYPE BrowserEngine:
    // Browser lifecycle
    FUNC launch(config: BrowserConfig) → Result<BrowserSession, BrowserError>
    FUNC connect(endpoint: CdpEndpoint) → Result<BrowserSession, BrowserError>
    FUNC close(session: BrowserSession) → Result<void, BrowserError>

    // Navigation
    FUNC navigate(session: BrowserSession, url: URL, wait: WaitCondition) → Result<NavigationResult, BrowserError>
    FUNC back(session: BrowserSession) → Result<void, BrowserError>
    FUNC forward(session: BrowserSession) → Result<void, BrowserError>
    FUNC reload(session: BrowserSession) → Result<void, BrowserError>

    // DOM Interaction
    FUNC query_selector(session: BrowserSession, selector: CssSelector) → Result<Element?, BrowserError>
    FUNC query_all(session: BrowserSession, selector: CssSelector) → Result<List<Element>, BrowserError>
    FUNC click(session: BrowserSession, target: ElementTarget) → Result<void, BrowserError>
    FUNC type_text(session: BrowserSession, target: ElementTarget, text: string) → Result<void, BrowserError>
    FUNC select_option(session: BrowserSession, target: ElementTarget, value: string) → Result<void, BrowserError>
    FUNC scroll(session: BrowserSession, direction: ScrollDirection, amount: int) → Result<void, BrowserError>

    // Page Inspection
    FUNC get_page_content(session: BrowserSession) → Result<PageContent, BrowserError>
    FUNC screenshot(session: BrowserSession, region: ScreenRegion?) → Result<ImageBuffer, BrowserError>
    FUNC get_accessibility_tree(session: BrowserSession) → Result<AccessibilityTree, BrowserError>
    FUNC evaluate_js(session: BrowserSession, expression: JsExpression) → Result<JsValue, BrowserError>

    // Network Observation
    FUNC intercept_requests(session: BrowserSession, filter: RequestFilter) → AsyncStream<NetworkRequest, BrowserError>
    FUNC get_cookies(session: BrowserSession, domain: string?) → Result<List<Cookie>, BrowserError>
    FUNC set_cookie(session: BrowserSession, cookie: Cookie) → Result<void, BrowserError>

    // Authentication via Browser
    FUNC perform_login(session: BrowserSession, login_spec: LoginSpec) → Result<AuthResult, BrowserError>

TYPE BrowserConfig:
    browser: BrowserType           // CHROMIUM | FIREFOX | WEBKIT
    headless: bool                 // true for server, false for desktop
    viewport: ViewportSize
    user_agent: string?
    proxy: ProxyConfig?
    extensions: List<ExtensionPath>?
    profile_dir: OsPath?          // Persistent browser profile
    cdp_port: int?                // Custom CDP port
```

---

**PSEUDO-ALGORITHM 5.3: Intelligent Page Interaction**

```
PROCEDURE InteractWithPage(
    browser: BrowserSession,
    action: PageAction,
    engine: BrowserEngine
) → Result<ActionResult, BrowserError>:

    // Step 1: Capture pre-action state
    pre_screenshot ← engine.screenshot(browser)
    pre_dom ← engine.get_page_content(browser)
    pre_a11y ← engine.get_accessibility_tree(browser)
    
    // Step 2: Resolve target element
    element ← MATCH action.target:
        CSS_SELECTOR(sel) →
            engine.query_selector(browser, sel)
        
        ACCESSIBILITY_LABEL(label) →
            FindByAccessibilityLabel(pre_a11y, label)
        
        VISUAL_DESCRIPTION(desc) →
            // Use vision model to locate element from screenshot
            coordinates ← vision_model.locate_element(pre_screenshot, desc)
            engine.element_at_point(browser, coordinates)
        
        XPATH(xpath) →
            engine.evaluate_js(browser,
                "document.evaluate('" + xpath + "', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue"
            )
    
    IF element IS NULL THEN
        RETURN Err(BrowserError("Target element not found: " + action.target))
    
    // Step 3: Verify element is actionable
    IF NOT element.is_visible THEN
        engine.scroll_into_view(browser, element)
        AWAIT stabilize(50ms)
    IF NOT element.is_enabled THEN
        RETURN Err(BrowserError("Target element is disabled"))
    
    // Step 4: Execute action
    MATCH action.type:
        CLICK →
            engine.click(browser, element)
        TYPE_TEXT(text) →
            engine.click(browser, element)  // Focus
            engine.type_text(browser, element, text)
        SELECT(value) →
            engine.select_option(browser, element, value)
        UPLOAD_FILE(path) →
            engine.set_input_files(browser, element, [path])
    
    // Step 5: Wait for page stabilization
    AWAIT engine.wait_for_stable(browser, 
        condition = action.wait_condition ?? WaitCondition.NETWORK_IDLE,
        timeout = action.timeout ?? 10s
    )
    
    // Step 6: Capture post-action state
    post_screenshot ← engine.screenshot(browser)
    post_dom ← engine.get_page_content(browser)
    
    // Step 7: Verify action effect
    changes ← DiffDom(pre_dom, post_dom)
    navigation_occurred ← browser.current_url != pre_dom.url
    
    result ← ActionResult(
        success = true,
        pre_screenshot = pre_screenshot,
        post_screenshot = post_screenshot,
        dom_changes = changes,
        navigated = navigation_occurred,
        new_url = browser.current_url,
        provenance = ActionProvenance(
            action = action,
            element_selector = element.unique_selector,
            timestamp = NOW()
        )
    )
    
    RETURN Ok(result)
```

---

### 5.5 Desktop GUI Interaction Engine

For native desktop applications not accessible via API or browser:

```
TYPE DesktopEngine:
    // Window management
    FUNC list_windows() → List<WindowInfo>
    FUNC focus_window(window: WindowId) → Result<void, DesktopError>
    FUNC get_window_bounds(window: WindowId) → Result<Rect, DesktopError>
    FUNC resize_window(window: WindowId, bounds: Rect) → Result<void, DesktopError>

    // Screen reading
    FUNC screenshot(region: ScreenRegion?) → Result<ImageBuffer, DesktopError>
    FUNC get_accessibility_tree(window: WindowId) → Result<AccessibilityTree, DesktopError>
    FUNC ocr_region(region: ScreenRegion) → Result<OcrResult, DesktopError>

    // Input simulation
    FUNC mouse_move(x: int, y: int) → Result<void, DesktopError>
    FUNC mouse_click(x: int, y: int, button: MouseButton) → Result<void, DesktopError>
    FUNC mouse_double_click(x: int, y: int) → Result<void, DesktopError>
    FUNC mouse_drag(from: Point, to: Point) → Result<void, DesktopError>
    FUNC keyboard_type(text: string) → Result<void, DesktopError>
    FUNC keyboard_hotkey(keys: List<Key>) → Result<void, DesktopError>

    // Clipboard
    FUNC clipboard_get() → Result<ClipboardContent, DesktopError>
    FUNC clipboard_set(content: ClipboardContent) → Result<void, DesktopError>
```

**Interaction strategy selection:**

$$
\text{Strategy}(\text{app}) = \begin{cases}
\text{API} & \text{if app exposes programmatic interface (preferred)} \\
\text{Accessibility Tree} & \text{if app supports a11y (reliable, structured)} \\
\text{Vision + OCR} & \text{if no a11y (fallback, less reliable)} \\
\text{Keyboard/Mouse} & \text{last resort (fragile, position-dependent)}
\end{cases}
$$

The SDK always prefers higher-fidelity interaction strategies over lower-fidelity ones.

---

## 6. L5: Domain Connector Layer

### 6.1 Connector Architecture

Each domain connector encapsulates **all knowledge required to interact with a specific software system**, including authentication, API versions, rate limits, data models, and error semantics.

```
TYPE DomainConnector:
    // Identity
    PROPERTY connector_id: ConnectorId
    PROPERTY version: SemanticVersion
    PROPERTY supported_operations: List<OperationDescriptor>

    // Lifecycle
    FUNC initialize(config: ConnectorConfig) → Result<void, ConnectorError>
    FUNC health_check() → Result<HealthStatus, ConnectorError>
    FUNC shutdown() → Result<void, ConnectorError>

    // MCP Tool Surface
    FUNC as_mcp_tools() → List<McpToolDescriptor>
    // Exposes all operations as MCP-compatible tool definitions

    // Capability Discovery
    FUNC capabilities() → ConnectorCapabilities
    FUNC operations(filter: OperationFilter?) → List<OperationDescriptor>
```

### 6.2 Google Workspace Connector

---

**PSEUDO-ALGORITHM 6.1: Google Workspace Connector Operations**

```
TYPE GoogleWorkspaceConnector IMPLEMENTS DomainConnector:

    // ═══════════════════════════════════════════
    // GMAIL
    // ═══════════════════════════════════════════
    FUNC gmail_list_messages(query: GmailQuery, page: PageToken?) → Result<Page<EmailSummary>, GmailError>
    FUNC gmail_get_message(id: MessageId, format: MessageFormat) → Result<Email, GmailError>
    FUNC gmail_send(draft: EmailDraft) → Result<SendReceipt, GmailError>
    FUNC gmail_reply(thread_id: ThreadId, body: EmailBody) → Result<SendReceipt, GmailError>
    FUNC gmail_search(query: string, max_results: int) → Result<List<EmailSummary>, GmailError>
    FUNC gmail_create_label(name: string) → Result<Label, GmailError>
    FUNC gmail_apply_label(message_id: MessageId, label: LabelId) → Result<void, GmailError>
    FUNC gmail_create_filter(criteria: FilterCriteria, action: FilterAction) → Result<Filter, GmailError>

    // ═══════════════════════════════════════════
    // GOOGLE DRIVE
    // ═══════════════════════════════════════════
    FUNC drive_list(folder: FolderId?, query: DriveQuery?, page: PageToken?) → Result<Page<DriveItem>, DriveError>
    FUNC drive_get_metadata(file_id: FileId) → Result<FileMetadata, DriveError>
    FUNC drive_download(file_id: FileId, export_format: MimeType?) → Result<FileContent, DriveError>
    FUNC drive_upload(content: FileContent, metadata: UploadMetadata) → Result<FileId, DriveError>
    FUNC drive_update(file_id: FileId, content: FileContent?, metadata: MetadataPatch?) → Result<FileMetadata, DriveError>
    FUNC drive_create_folder(name: string, parent: FolderId?) → Result<FolderId, DriveError>
    FUNC drive_share(file_id: FileId, permission: PermissionSpec) → Result<void, DriveError>
    FUNC drive_search(query: string, mime_types: List<MimeType>?) → Result<List<DriveItem>, DriveError>

    // ═══════════════════════════════════════════
    // GOOGLE DOCS
    // ═══════════════════════════════════════════
    FUNC docs_get(doc_id: DocId) → Result<Document, DocsError>
    FUNC docs_create(title: string, content: DocContent?) → Result<DocId, DocsError>
    FUNC docs_batch_update(doc_id: DocId, requests: List<DocUpdateRequest>) → Result<void, DocsError>
    FUNC docs_insert_text(doc_id: DocId, location: InsertionPoint, text: string) → Result<void, DocsError>
    FUNC docs_replace_text(doc_id: DocId, find: string, replace: string) → Result<int, DocsError>
    FUNC docs_export(doc_id: DocId, format: ExportFormat) → Result<Bytes, DocsError>
    // ExportFormat: PDF | DOCX | TXT | HTML | ODT | EPUB

    // ═══════════════════════════════════════════
    // GOOGLE SHEETS
    // ═══════════════════════════════════════════
    FUNC sheets_get(spreadsheet_id: SpreadsheetId, range: A1Range?) → Result<SheetData, SheetsError>
    FUNC sheets_update(spreadsheet_id: SpreadsheetId, range: A1Range, values: CellMatrix) → Result<UpdateResult, SheetsError>
    FUNC sheets_append(spreadsheet_id: SpreadsheetId, range: A1Range, values: CellMatrix) → Result<AppendResult, SheetsError>
    FUNC sheets_create(title: string, sheets: List<SheetSpec>?) → Result<SpreadsheetId, SheetsError>
    FUNC sheets_add_sheet(spreadsheet_id: SpreadsheetId, title: string) → Result<SheetId, SheetsError>

    // ═══════════════════════════════════════════
    // GOOGLE SLIDES
    // ═══════════════════════════════════════════
    FUNC slides_get(presentation_id: PresentationId) → Result<Presentation, SlidesError>
    FUNC slides_create(title: string) → Result<PresentationId, SlidesError>
    FUNC slides_add_slide(presentation_id: PresentationId, layout: SlideLayout) → Result<SlideId, SlidesError>
    FUNC slides_batch_update(presentation_id: PresentationId, requests: List<SlideUpdateRequest>) → Result<void, SlidesError>
    FUNC slides_export(presentation_id: PresentationId, format: SlideExportFormat) → Result<Bytes, SlidesError>

    // ═══════════════════════════════════════════
    // GOOGLE CALENDAR
    // ═══════════════════════════════════════════
    FUNC calendar_list_events(calendar_id: CalendarId, time_range: TimeRange) → Result<List<CalendarEvent>, CalendarError>
    FUNC calendar_create_event(calendar_id: CalendarId, event: EventSpec) → Result<EventId, CalendarError>
    FUNC calendar_update_event(calendar_id: CalendarId, event_id: EventId, patch: EventPatch) → Result<void, CalendarError>
    FUNC calendar_delete_event(calendar_id: CalendarId, event_id: EventId) → Result<void, CalendarError>

    // ═══════════════════════════════════════════
    // GOOGLE MEET
    // ═══════════════════════════════════════════
    FUNC meet_create_space(config: MeetConfig) → Result<MeetSpace, MeetError>
    FUNC meet_get_recording(meeting_id: MeetingId) → Result<RecordingInfo, MeetError>

    // ═══════════════════════════════════════════
    // INTERNAL: Auth and rate limiting
    // ═══════════════════════════════════════════
    INTERNAL auth_type ← OAUTH2_AUTHORIZATION_CODE
    INTERNAL scopes_map: Map<OperationType, Set<OAuthScope>>
    INTERNAL rate_limits: Map<ApiName, RateLimitConfig>
    // Gmail: 250 quota units/s (user), 1B quota units/day (domain)
    // Drive: 20000 queries/100s (user)
    // Docs: 300 requests/60s (user)
    // etc.
```

---

### 6.3 IDE Connector

The IDE connector provides **programmatic access to any IDE** through multiple interaction strategies:

```
TYPE IdeConnector:
    // Discovery
    FUNC detect_running_ides() → List<IdeInstance>
    FUNC connect(instance: IdeInstance) → Result<IdeSession, IdeError>

    // File Operations
    FUNC open_file(session: IdeSession, path: FilePath) → Result<void, IdeError>
    FUNC get_active_file(session: IdeSession) → Result<FileInfo, IdeError>
    FUNC get_file_content(session: IdeSession, path: FilePath) → Result<string, IdeError>
    FUNC edit_file(session: IdeSession, path: FilePath, edits: List<TextEdit>) → Result<void, IdeError>
    FUNC save_file(session: IdeSession, path: FilePath?) → Result<void, IdeError>

    // Project Operations
    FUNC get_project_structure(session: IdeSession) → Result<ProjectTree, IdeError>
    FUNC search_symbol(session: IdeSession, query: string) → Result<List<SymbolInfo>, IdeError>
    FUNC go_to_definition(session: IdeSession, symbol: SymbolRef) → Result<Location, IdeError>
    FUNC find_references(session: IdeSession, symbol: SymbolRef) → Result<List<Location>, IdeError>
    FUNC get_diagnostics(session: IdeSession, path: FilePath?) → Result<List<Diagnostic>, IdeError>
    FUNC run_code_action(session: IdeSession, action: CodeAction) → Result<void, IdeError>

    // Terminal Integration
    FUNC get_integrated_terminal(session: IdeSession) → Result<TerminalSession, IdeError>
    FUNC run_task(session: IdeSession, task: IdeTask) → Result<TaskResult, IdeError>

    // Debugging
    FUNC set_breakpoint(session: IdeSession, location: Location) → Result<BreakpointId, IdeError>
    FUNC start_debug(session: IdeSession, config: DebugConfig) → Result<DebugSession, IdeError>
    FUNC inspect_variable(debug: DebugSession, name: string) → Result<VariableInfo, IdeError>

    // Build & Test
    FUNC build(session: IdeSession, config: BuildConfig?) → Result<BuildResult, IdeError>
    FUNC run_tests(session: IdeSession, filter: TestFilter?) → Result<TestResult, IdeError>
    FUNC get_test_coverage(session: IdeSession) → Result<CoverageReport, IdeError>

TYPE IdeInstance:
    ide_type: IdeType    // VSCODE | JETBRAINS | VIM | NEOVIM | EMACS | SUBLIME | XCODE | ANDROID_STUDIO
    process_id: int
    version: string
    workspace_path: OsPath
    connection_method: IdeConnectionMethod
```

**IDE connection strategy matrix:**

| IDE | Connection Method | Protocol | Capabilities |
|---|---|---|---|
| **VS Code** | Extension API + LSP + Built-in terminal | JSON-RPC over stdio/pipe | Full: files, terminal, debug, extensions |
| **JetBrains (IntelliJ, PyCharm, etc.)** | Gateway/Remote Dev API + PSI | HTTP REST + WebSocket | Full: PSI model, refactoring, debug |
| **Vim/Neovim** | Remote plugin API (msgpack-rpc) | MessagePack-RPC over socket | Files, commands, buffers, LSP |
| **Emacs** | emacsclient / EPC | Lisp S-expressions over socket | Files, commands, elisp eval |
| **Sublime Text** | Plugin API (Python) | Internal Python API | Files, commands, limited debug |
| **Xcode** | xcrun + simctl + XCTest | CLI + device protocols | Build, test, simulator |
| **Android Studio** | ADB + Gradle + JetBrains API | CLI + HTTP | Build, test, device control |

### 6.4 Connector Registration and MCP Surface Projection

Every domain connector **automatically exposes its operations as MCP tools**, enabling standardized discovery and invocation:

---

**PSEUDO-ALGORITHM 6.2: MCP Surface Projection**

```
PROCEDURE ProjectConnectorToMcp(connector: DomainConnector) → List<McpToolDescriptor>:

    tools ← []
    
    FOR EACH op IN connector.operations():
        // Generate MCP tool descriptor
        tool ← McpToolDescriptor(
            name = connector.connector_id + "." + op.name,
            // e.g., "google_workspace.gmail_send"
            
            description = op.description,
            
            input_schema = JsonSchema.from_typed(op.input_type),
            // Converts typed Protobuf/struct → JSON Schema for MCP
            
            output_schema = JsonSchema.from_typed(op.output_type),
            // Structured output where available
            
            annotations = McpAnnotations(
                idempotent = op.is_idempotent,
                destructive = op.is_destructive,
                requires_approval = op.requires_human_approval,
                timeout_class = op.timeout_class,
                cost_class = op.cost_class,
                auth_scopes = op.required_scopes
            )
        )
        
        APPEND tool TO tools
    
    RETURN tools
```

---

### 6.5 Complete Connector Registry

| Connector Domain | Key Operations | Protocol(s) | Auth |
|---|---|---|---|
| **Google Workspace** | Gmail, Drive, Docs, Sheets, Slides, Calendar, Meet | REST (HTTP/2) | OAuth2 |
| **Microsoft 365** | Outlook, OneDrive, Word, Excel, PowerPoint, Teams | Microsoft Graph (REST) | OAuth2/MSAL |
| **GitHub/GitLab** | Repos, PRs, Issues, CI/CD, Code Search, Actions | REST + GraphQL | OAuth2/PAT |
| **Slack/Discord** | Messages, channels, threads, reactions, files | REST + WebSocket (RTM) | OAuth2/Bot token |
| **Jira/Linear/Asana** | Issues, projects, sprints, assignments | REST | OAuth2/API key |
| **AWS** | S3, Lambda, EC2, RDS, CloudWatch, IAM | REST (Sigv4) | IAM/STS |
| **GCP** | GCS, Compute, BigQuery, Cloud Run | REST + gRPC | Service Account/OAuth2 |
| **Azure** | Blob, VM, SQL, Functions, Monitor | REST | Azure AD/MSI |
| **Databases** | Query, schema inspect, migrate | Native protocol (Postgres, MySQL, etc.) | Username/password/IAM |
| **Docker/K8s** | Container lifecycle, pod management, logs | REST (Docker API/K8s API) | TLS client cert/token |
| **CI/CD** | Pipeline trigger, status, logs, artifacts | REST (Jenkins/CircleCI/GH Actions) | API token |
| **Monitoring** | Metrics query, alert management, dashboards | REST/gRPC (Prometheus/Datadog/Grafana) | API key |
| **File Systems** | SFTP, SCP, SMB, NFS, local FS | SSH/SFTP, SMB, NFS, POSIX | SSH key/password/Kerberos |
| **Email (Generic)** | IMAP/SMTP for non-Google mail | IMAP + SMTP | Username/password/OAuth2 |
| **Browsers** | Page control, DOM, network, screenshots | CDP/BiDi (WebSocket) | N/A (local) |
| **Terminals** | Command execution on all OS | SSH (remote) / PTY (local) | SSH key/password/local |
| **IDEs** | File editing, debugging, building, testing | IDE-specific (see §6.3) | Local/extension token |

---

## 7. L6: Agent Orchestration Layer — Unified Loop with Universal Routing

### 7.1 The Universal Agent Loop

The orchestration layer executes the canonical bounded agent loop, routing each action to the appropriate connector, modality, and protocol:

---

**PSEUDO-ALGORITHM 7.1: Universal Agent Loop with Connector Routing**

```
PROCEDURE UniversalAgentLoop(
    sdk: AgentSdk,
    task: TaskSpec,
    options: LoopOptions
) → Result<TaskResult, AgentError>:

    // Phase 1: PLAN
    plan ← sdk.llm.execute(PlanRequest(
        objective = task.objective,
        available_connectors = sdk.connector_registry.list_capabilities(),
        constraints = task.constraints,
        context = sdk.context.compile()
    ))
    
    steps ← Decompose(plan, max_steps = options.max_steps)
    committed_results ← []
    
    // Phase 2: Execution Loop
    FOR EACH step IN steps:
        iteration ← 0
        step_complete ← false
        
        WHILE NOT step_complete AND iteration < options.max_step_iterations:
            
            // RETRIEVE — gather evidence for current step
            retrieval_plan ← GenerateRetrievalPlan(step, sdk.retrieval_engine)
            evidence ← ExecuteRetrievalPlan(retrieval_plan, sdk.retrieval_engine)
            
            // ROUTE — determine which connector and modality to use
            route ← RouteAction(step, sdk.connector_registry, sdk.modality_selector)
            // RouteAction returns: {connector, modality, protocol, auth_requirements}
            
            // AUTHENTICATE — resolve credentials for target
            credential ← sdk.auth.resolve_credential(
                route.connector.endpoint,
                route.auth_requirements.scopes
            )
            
            // ACT — execute the action via the routed connector
            action_result ← MATCH route.modality:
                API →
                    route.connector.invoke(
                        step.operation,
                        step.params,
                        credential,
                        options.action_options
                    )
                
                TERMINAL →
                    sdk.terminal.execute(
                        route.connector.session,
                        step.command,
                        options.terminal_options
                    )
                
                BROWSER →
                    sdk.browser.interact(
                        route.connector.session,
                        step.page_actions,
                        options.browser_options
                    )
                
                DESKTOP →
                    sdk.desktop.interact(
                        route.connector.session,
                        step.gui_actions,
                        options.desktop_options
                    )
            
            IF action_result IS Err THEN
                // Error handling with retry/escalate/skip
                error_decision ← HandleActionError(
                    action_result.error, step, iteration, options
                )
                MATCH error_decision:
                    RETRY → iteration += 1; CONTINUE
                    ESCALATE → AWAIT request_human_intervention(step, action_result.error)
                    SKIP → BREAK
                    ABORT → RETURN Err(AgentError("Step failed: " + step.id))
            
            // VERIFY — check postconditions
            verification ← VerifyStepResult(
                step,
                action_result.value,
                evidence,
                sdk.llm
            )
            
            IF verification.passed THEN
                // CRITIQUE — quality gate
                critique ← CritiqueResult(
                    step, action_result.value, sdk.llm, options.critique_config
                )
                
                IF critique.severity < options.critique_threshold THEN
                    // COMMIT
                    APPEND CommittedResult(step, action_result.value, verification, critique) TO committed_results
                    
                    // Update memory with learned outcomes
                    sdk.memory.write(MemoryWriteRequest(
                        content = ExtractLearning(step, action_result.value, critique),
                        tier = SESSION,
                        provenance = StepProvenance(step.id, task.id)
                    ))
                    
                    step_complete ← true
                ELSE
                    // REPAIR
                    repair_plan ← GenerateRepairPlan(step, critique, sdk.llm)
                    step ← MergeRepairIntoStep(step, repair_plan)
                    iteration += 1
            ELSE
                // Verification failed — retry with adjusted approach
                step ← AdjustApproach(step, verification.failure_reason, sdk.llm)
                iteration += 1
        
        IF NOT step_complete THEN
            // Step exhausted iteration budget
            PersistFailureState(step, task.id, sdk.memory)
            IF options.on_step_failure = ABORT THEN
                RETURN Err(AgentError("Step exceeded max iterations: " + step.id))
    
    RETURN Ok(TaskResult(committed_results))
```

---

### 7.2 Action Routing Algorithm

The routing algorithm selects the optimal connector, interaction modality, and protocol for each action:

---

**PSEUDO-ALGORITHM 7.2: Action Routing**

```
PROCEDURE RouteAction(
    step: TaskStep,
    registry: ConnectorRegistry,
    modality_selector: ModalitySelector
) → ActionRoute:

    // Step 1: Identify candidate connectors
    candidates ← registry.find_connectors(
        target_system = step.target_system,
        operation = step.operation_type
    )
    
    IF candidates IS EMPTY THEN
        // Fallback: attempt dynamic discovery
        candidates ← registry.discover_connector(step.target_system)
        IF candidates IS EMPTY THEN
            RAISE Unavailable("No connector available for: " + step.target_system)
    
    // Step 2: Select optimal modality
    FOR EACH connector IN candidates:
        modalities ← connector.supported_modalities()
        
        FOR EACH modality IN modalities:
            score ← 0
            score += w_reliability * modality.reliability_score
            // API > CLI > Browser > Desktop GUI (generally)
            score += w_speed * (1.0 / modality.avg_latency)
            score += w_observability * modality.observability_score
            score += w_reversibility * modality.reversibility_score
            score -= w_fragility * modality.fragility_score
            
            connector.modality_scores[modality] ← score
    
    // Step 3: Select best (connector, modality) pair
    best ← ArgMax(
        [(c, m, c.modality_scores[m]) FOR c IN candidates FOR m IN c.supported_modalities()],
        key = score
    )
    
    // Step 4: Select protocol
    protocol ← best.connector.preferred_protocol(best.modality)
    
    // Step 5: Determine auth requirements
    auth_req ← best.connector.auth_requirements(best.modality, step.operation)
    
    RETURN ActionRoute(
        connector = best.connector,
        modality = best.modality,
        protocol = protocol,
        auth_requirements = auth_req,
        estimated_latency = best.modality.avg_latency,
        estimated_cost = best.connector.cost_estimate(step.operation)
    )
```

---

### 7.3 Await-for-Requirement: Long-Running Operation Support

Certain operations (CI/CD pipelines, file processing, human approvals, large data transfers) require the agent to **await asynchronously** for an indeterminate duration:

```
TYPE LongRunningOperation:
    FUNC start(params: OperationParams) → Result<OperationHandle, OperationError>
    FUNC poll_status(handle: OperationHandle) → Result<OperationStatus, OperationError>
    FUNC await_completion(handle: OperationHandle, deadline: Duration) → Result<OperationResult, OperationError>
    FUNC cancel(handle: OperationHandle) → Result<void, OperationError>

TYPE OperationStatus:
    state: OperationState    // PENDING | RUNNING | SUCCEEDED | FAILED | CANCELLED
    progress: float?         // [0, 1] if available
    message: string?
    eta: Duration?
    intermediate_result: Any?
```

---

**PSEUDO-ALGORITHM 7.3: Intelligent Await with Backoff**

```
PROCEDURE AwaitOperation(
    handle: OperationHandle,
    connector: DomainConnector,
    deadline: Duration,
    options: AwaitOptions
) → Result<OperationResult, OperationError>:

    start_time ← NOW()
    poll_interval ← options.initial_poll_interval   // e.g., 1s
    max_interval ← options.max_poll_interval          // e.g., 60s
    poll_count ← 0
    
    WHILE elapsed(start_time) < deadline:
        status ← connector.poll_status(handle)
        
        IF status IS Err THEN
            IF status.error.class = TransientNetwork THEN
                // Transient error; continue polling
                LOG warning "Poll failed transiently: " + status.error
            ELSE
                RETURN Err(status.error)
        
        MATCH status.value.state:
            SUCCEEDED →
                RETURN Ok(status.value.result)
            FAILED →
                RETURN Err(OperationError("Operation failed: " + status.value.message))
            CANCELLED →
                RETURN Err(OperationError("Operation was cancelled"))
            PENDING | RUNNING →
                // Emit progress if available
                IF status.value.progress IS NOT NULL THEN
                    EMIT event: operation.progress {
                        handle, progress = status.value.progress, eta = status.value.eta
                    }
                
                // Check if agent can perform other work while waiting
                IF options.allow_interleave AND sdk.task_queue.has_pending() THEN
                    // Yield to orchestrator for parallel work
                    sdk.yield_with_callback(handle, on_complete = ResumeHere)
                    RETURN  // Will be resumed by callback
                
                // Adaptive polling interval
                poll_interval ← MIN(
                    max_interval,
                    poll_interval * options.backoff_multiplier
                )
                
                // If we have ETA, align poll to just after ETA
                IF status.value.eta IS NOT NULL AND status.value.eta > poll_interval THEN
                    poll_interval ← status.value.eta + jitter(1s)
                
                SLEEP(poll_interval)
                poll_count += 1
    
    // Deadline exceeded
    IF options.cancel_on_timeout THEN
        connector.cancel(handle)
    
    RETURN Err(DeadlineExceeded("Operation did not complete within deadline"))
```

---

## 8. Inspection and Observation Framework

### 8.1 Deep System Inspection

The SDK provides **comprehensive inspection capabilities** that allow the agent to observe and diagnose the state of any connected system:

```
TYPE InspectionEngine:
    // Application State
    FUNC inspect_page(browser: BrowserSession) → PageInspection
    FUNC inspect_terminal(terminal: TerminalSession) → TerminalInspection
    FUNC inspect_process(pid: ProcessId) → ProcessInspection
    FUNC inspect_network(interface: NetworkInterface?) → NetworkInspection
    FUNC inspect_filesystem(path: OsPath, depth: int) → FilesystemInspection

    // Service State
    FUNC inspect_api_health(endpoint: Endpoint) → ApiHealthInspection
    FUNC inspect_database_state(connection: DbConnection, query: InspectionQuery) → DbInspection
    FUNC inspect_container(container_id: ContainerId) → ContainerInspection
    FUNC inspect_logs(source: LogSource, filter: LogFilter, window: TimeWindow) → LogInspection
    FUNC inspect_metrics(source: MetricSource, queries: List<MetricQuery>) → MetricInspection
    FUNC inspect_traces(trace_id: TraceId) → TraceInspection

    // Repository State
    FUNC inspect_repo(path: OsPath) → RepoInspection
    FUNC inspect_diff(path: OsPath, base: GitRef, head: GitRef) → DiffInspection
    FUNC inspect_ci_status(repo: RepoRef) → CiInspection
    FUNC inspect_dependencies(path: OsPath) → DependencyInspection

TYPE PageInspection:
    url: URL
    title: string
    dom_summary: DomSummary          // Structural summary, not full DOM
    accessibility_tree: A11yTree
    screenshot: ImageBuffer
    console_errors: List<ConsoleError>
    network_requests: List<NetworkRequestSummary>
    performance_metrics: WebVitals
    form_state: List<FormFieldState>
    local_storage_keys: List<string>
    cookies: List<CookieSummary>
```

### 8.2 Universal Page Inspection for Web Applications

---

**PSEUDO-ALGORITHM 8.1: Deep Page Inspection**

```
PROCEDURE InspectPage(
    browser: BrowserSession,
    engine: BrowserEngine,
    options: InspectionOptions
) → PageInspection:

    inspection ← PageInspection()
    
    // Basic page info
    inspection.url ← engine.evaluate_js(browser, "window.location.href")
    inspection.title ← engine.evaluate_js(browser, "document.title")
    
    // DOM summary (not full DOM — too large for context window)
    IF options.include_dom THEN
        full_dom ← engine.get_page_content(browser)
        inspection.dom_summary ← SummarizeDom(full_dom, 
            max_depth = options.dom_depth ?? 5,
            include_text = options.include_text_content,
            include_attributes = ["id", "class", "name", "role", "aria-label", "href", "src", "type", "value"],
            max_elements = options.max_dom_elements ?? 500
        )
    
    // Accessibility tree — often more useful than DOM for agent interaction
    IF options.include_a11y THEN
        inspection.accessibility_tree ← engine.get_accessibility_tree(browser)
    
    // Visual capture
    IF options.include_screenshot THEN
        inspection.screenshot ← engine.screenshot(browser, region = options.screenshot_region)
    
    // Console errors
    inspection.console_errors ← engine.evaluate_js(browser, """
        window.__agent_console_errors || []
    """)
    // Note: console error capture requires prior injection of error listener
    
    // Network activity
    IF options.include_network THEN
        inspection.network_requests ← engine.get_network_log(browser, 
            since = options.network_window_start,
            include_bodies = false  // Bodies are too large; include only metadata
        )
    
    // Form state — useful for login and data entry
    IF options.include_forms THEN
        inspection.form_state ← engine.evaluate_js(browser, """
            Array.from(document.querySelectorAll('form')).map(form => ({
                id: form.id,
                action: form.action,
                method: form.method,
                fields: Array.from(form.elements).map(el => ({
                    name: el.name,
                    type: el.type,
                    value: el.type === 'password' ? '[REDACTED]' : el.value,
                    required: el.required,
                    disabled: el.disabled
                }))
            }))
        """)
    
    // Performance metrics
    IF options.include_performance THEN
        inspection.performance_metrics ← engine.evaluate_js(browser, """
            ({
                lcp: performance.getEntriesByType('largest-contentful-paint').pop()?.startTime,
                fid: performance.getEntriesByType('first-input').pop()?.processingStart,
                cls: performance.getEntriesByType('layout-shift').reduce((a, b) => a + b.value, 0),
                ttfb: performance.getEntriesByType('navigation').pop()?.responseStart,
                load_time: performance.timing.loadEventEnd - performance.timing.navigationStart
            })
        """)
    
    RETURN inspection
```

---

### 8.3 Login Flow Automation

---

**PSEUDO-ALGORITHM 8.2: Universal Login Flow**

```
PROCEDURE PerformLogin(
    browser: BrowserSession,
    engine: BrowserEngine,
    login_spec: LoginSpec,
    credential_store: SecureStore
) → Result<AuthResult, LoginError>:

    // Step 1: Navigate to login page
    engine.navigate(browser, login_spec.login_url, wait = NETWORK_IDLE)
    
    // Step 2: Detect login form
    page_inspection ← InspectPage(browser, engine, InspectionOptions(include_forms = true, include_a11y = true))
    
    login_form ← DetectLoginForm(page_inspection)
    // Heuristics: form with password field, or a11y landmarks with "login"/"sign in" role
    
    IF login_form IS NULL THEN
        // Try: look for SSO buttons (Google, Microsoft, GitHub, etc.)
        sso_button ← DetectSsoButton(page_inspection, login_spec.preferred_sso)
        IF sso_button IS NOT NULL THEN
            engine.click(browser, sso_button)
            AWAIT engine.wait_for_navigation(browser, timeout = 10s)
            // Recurse into SSO provider login
            RETURN PerformLogin(browser, engine, login_spec.sso_spec, credential_store)
        RETURN Err(LoginError("Cannot detect login form"))
    
    // Step 3: Fill credentials
    credentials ← credential_store.get_login_creds(login_spec.domain)
    
    // Username/email field
    username_field ← login_form.find_field(type = ["email", "text"], name_hint = ["user", "email", "login"])
    engine.click(browser, username_field)
    engine.type_text(browser, username_field, credentials.username)
    
    // Password field (may be on separate page)
    password_field ← login_form.find_field(type = "password")
    IF password_field IS NULL THEN
        // Multi-step login: submit username first, then password
        submit_button ← login_form.find_submit()
        engine.click(browser, submit_button)
        AWAIT engine.wait_for_stable(browser, timeout = 5s)
        page_inspection ← InspectPage(browser, engine, InspectionOptions(include_forms = true))
        password_field ← DetectPasswordField(page_inspection)
    
    engine.click(browser, password_field)
    engine.type_text(browser, password_field, credentials.password)
    
    // Step 4: Submit
    submit_button ← login_form.find_submit()
    engine.click(browser, submit_button)
    AWAIT engine.wait_for_stable(browser, timeout = 10s)
    
    // Step 5: Handle MFA if required
    mfa_challenge ← DetectMfaChallenge(browser, engine)
    IF mfa_challenge IS NOT NULL THEN
        MATCH mfa_challenge.type:
            TOTP →
                totp_code ← credential_store.generate_totp(login_spec.domain)
                engine.type_text(browser, mfa_challenge.input_field, totp_code)
                engine.click(browser, mfa_challenge.submit_button)
            
            SMS | EMAIL →
                // Signal to agent: human action required
                code ← AWAIT request_human_input(
                    "Enter MFA code sent to " + mfa_challenge.delivery_hint
                )
                engine.type_text(browser, mfa_challenge.input_field, code)
                engine.click(browser, mfa_challenge.submit_button)
            
            PUSH_NOTIFICATION →
                AWAIT request_human_action("Approve push notification on your device")
                AWAIT engine.wait_for_navigation(browser, timeout = 60s)
            
            SECURITY_KEY →
                // Hardware key — requires physical interaction
                AWAIT request_human_action("Touch your security key")
                AWAIT engine.wait_for_navigation(browser, timeout = 30s)
        
        AWAIT engine.wait_for_stable(browser, timeout = 10s)
    
    // Step 6: Verify login success
    post_login_url ← engine.evaluate_js(browser, "window.location.href")
    cookies ← engine.get_cookies(browser, login_spec.domain)
    
    IF IsLoginSuccessful(post_login_url, cookies, login_spec.success_indicators) THEN
        // Extract session credentials
        session_cookies ← FilterSessionCookies(cookies)
        auth_tokens ← ExtractAuthTokens(browser, engine)
        
        RETURN Ok(AuthResult(
            session_cookies = session_cookies,
            auth_tokens = auth_tokens,
            expires_at = EstimateSessionExpiry(session_cookies)
        ))
    ELSE
        error_message ← DetectLoginError(browser, engine)
        RETURN Err(LoginError("Login failed: " + error_message))
```

---

## 9. MCP Integration: Universal Tool Surface

### 9.1 MCP as the Universal Tool Protocol

All connectors, modalities, and system interactions are **projected onto a unified MCP tool surface**:

$$
\text{MCP Surface} = \bigcup_{c \in \text{Connectors}} \text{ProjectToMcp}(c) \cup \bigcup_{m \in \text{Modalities}} \text{ProjectToMcp}(m)
$$

This enables any LLM with MCP support to discover and invoke any operation across the entire software ecosystem through a single, standardized protocol.

### 9.2 Tool Manifest Structure

```
TYPE UniversalToolManifest:
    manifest_version: SemanticVersion
    generated_at: ISO8601
    
    // Categories
    categories: [
        Category("google_workspace", [gmail_tools, drive_tools, docs_tools, ...]),
        Category("terminal", [execute_command, create_session, ...]),
        Category("browser", [navigate, click, type_text, screenshot, inspect_page, ...]),
        Category("ide", [open_file, edit_file, run_tests, debug, ...]),
        Category("git", [clone, commit, push, create_pr, ...]),
        Category("database", [query, inspect_schema, migrate, ...]),
        Category("cloud", [aws_tools, gcp_tools, azure_tools, ...]),
        Category("file_system", [read, write, search, watch, ...]),
        Category("network", [http_request, ssh_connect, scp_transfer, ...]),
        Category("monitoring", [query_metrics, get_logs, get_traces, ...])
    ]
    
    // Total tool count
    total_tools: int
    
    // Lazy loading metadata
    per_category_schema_urls: Map<CategoryId, URL>
    // Tools are loaded lazily by category to minimize context cost
```

### 9.3 Lazy Tool Loading Strategy

Loading all tools into the LLM context is **prohibitively expensive** (thousands of tools × hundreds of tokens per schema = context window exhaustion). The SDK implements **lazy, demand-driven tool loading**:

$$
\text{LoadedTools}(t) = \text{TopK}\Big(\text{Rank}\big(\text{AllTools},\; \text{task}(t),\; \text{step}(t)\big),\; k = \text{budget} / \text{avg\_schema\_tokens}\Big)
$$

---

**PSEUDO-ALGORITHM 9.1: Lazy Tool Loading**

```
PROCEDURE SelectToolsForContext(
    task_step: TaskStep,
    tool_manifest: UniversalToolManifest,
    token_budget: int
) → List<McpToolDescriptor>:

    // Step 1: Determine relevant categories
    relevant_categories ← InferRelevantCategories(task_step)
    // Uses task description, target system, and step type to filter categories
    // e.g., "Send email to team about deployment" → ["google_workspace", "terminal"]
    
    // Step 2: Load schemas for relevant categories only
    candidate_tools ← []
    FOR EACH category IN relevant_categories:
        tools ← tool_manifest.load_category(category)
        FOR EACH tool IN tools:
            tool.relevance ← ComputeToolRelevance(tool, task_step)
            APPEND tool TO candidate_tools
    
    // Step 3: Rank by relevance
    SortDescending(candidate_tools, key = relevance)
    
    // Step 4: Select within token budget
    selected ← []
    budget_remaining ← token_budget
    FOR EACH tool IN candidate_tools:
        schema_tokens ← TokenCount(tool.to_mcp_json())
        IF budget_remaining >= schema_tokens THEN
            APPEND tool TO selected
            budget_remaining -= schema_tokens
        ELSE
            BREAK  // Budget exhausted
    
    // Step 5: Always include core tools (navigate, screenshot, execute_command)
    FOR EACH core_tool IN CORE_TOOL_SET:
        IF core_tool NOT IN selected THEN
            IF budget_remaining >= TokenCount(core_tool.to_mcp_json()) THEN
                PREPEND core_tool TO selected
                budget_remaining -= TokenCount(core_tool.to_mcp_json())
    
    RETURN selected
```

---

## 10. Reliability, Security, and Production Invariants

### 10.1 End-to-End Reliability Architecture

| Failure Domain | Mechanism | Recovery |
|---|---|---|
| **Network partition** | Circuit breaker per connector; cached fallback | Retry with jitter; degrade to cached data |
| **Auth token expiry** | Proactive refresh at 80% TTL | Auto-refresh; re-auth flow; credential rotation |
| **API rate limit** | Per-connector token bucket; 429 backoff parsing | Queue with priority; batch operations; request coalescence |
| **Tool execution timeout** | Per-tool timeout class; deadline propagation | Cancel + compensate; retry with extended deadline |
| **Browser crash** | Process isolation; health monitoring | Restart browser session; replay from last checkpoint |
| **Terminal hang** | PTY timeout; SIGTERM → SIGKILL escalation | Kill process; reset session; report hung command |
| **IDE disconnection** | Heartbeat monitoring; reconnection with state recovery | Reconnect; re-attach to workspace; verify file state |
| **Memory exhaustion** | Resource governor; bounded buffers | Evict low-priority context; compress; failsafe mode |
| **LLM inference failure** | Model fallback chain; retry with backoff | Primary → secondary model; cached response if available |
| **Data corruption** | Checksums on all persisted state; WAL for durability | Restore from checkpoint; replay WAL |

### 10.2 Security Enforcement

| Invariant | Enforcement |
|---|---|
| **Least-privilege credentials** | Scope ceiling on every credential resolution; no ambient authority |
| **Credential isolation** | Per-connector credential scope; no cross-connector credential sharing |
| **Input sanitization** | Command injection prevention for terminal; XSS prevention for browser eval |
| **Audit trail** | Every state-mutating operation logged with timestamp, actor, target, and result |
| **Human approval gates** | Destructive operations require explicit human confirmation |
| **Secret hygiene** | No secrets in logs, traces, or context; redaction at emission |
| **TLS everywhere** | All external connections use TLS 1.3; certificate pinning for high-security endpoints |
| **Sandbox isolation** | Browser and terminal sessions run in isolated processes/containers |

### 10.3 Observability Stack

Every layer of the SDK emits **correlated telemetry**:

```
Trace: sdk.agent.task (root)
  ├── sdk.plan
  ├── sdk.step[0]
  │     ├── sdk.retrieval (evidence gathering)
  │     ├── sdk.route (connector selection)
  │     ├── sdk.auth.resolve (credential acquisition)
  │     ├── sdk.connector.google_workspace.gmail_send (action)
  │     │     ├── sdk.protocol.http2.request
  │     │     ├── sdk.middleware.retry (if retry occurred)
  │     │     └── sdk.middleware.rate_limit
  │     ├── sdk.verify (postcondition check)
  │     └── sdk.commit (result persistence)
  ├── sdk.step[1]
  │     ├── sdk.route → sdk.modality.terminal
  │     ├── sdk.connector.terminal.execute
  │     │     ├── sdk.terminal.spawn
  │     │     └── sdk.terminal.await_output
  │     └── ...
  └── sdk.step[N]
```

All spans carry: `trace_id`, `span_id`, `parent_span_id`, `tenant_id`, `task_id`, `step_id`, `connector_id`, `modality`, `protocol`, `latency_ms`, `tokens_used`, `cost_usd`, `error_class` (if any).

---

## 11. Formal Quality Invariants

| ID | Invariant | Enforcement |
|---|---|---|
| **SDK-INV-01** | Every external system interaction passes through a typed connector | L5 + L4 type system; no raw HTTP/TCP at L6 |
| **SDK-INV-02** | Every state-mutating tool call carries an idempotency key | SDK middleware injection at L3 |
| **SDK-INV-03** | Every credential is scoped to minimum required permissions | AuthManager scope ceiling at resolution time |
| **SDK-INV-04** | Every destructive operation requires human approval (configurable) | Approval gate in connector invocation path |
| **SDK-INV-05** | Every operation completes within its deadline or fails explicitly | Deadline interceptor propagation through all layers |
| **SDK-INV-06** | No secret appears in logs, traces, context, or error messages | Redaction pipeline in telemetry export path |
| **SDK-INV-07** | Tool schemas are loaded lazily; context budget is never exceeded by tool affordances | Lazy loading with token budget cap at L6 |
| **SDK-INV-08** | Every browser/terminal session is isolated; no cross-session state leakage | Process isolation at L4 |
| **SDK-INV-09** | Offline operations are durably queued and synced on reconnect | WAL + sync-on-reconnect at L3 |
| **SDK-INV-10** | Protocol selection is automatic and optimal for the target system and environment | Protocol selection algorithm at L2 |
| **SDK-INV-11** | Every connector exposes health, latency, and error-rate metrics | Middleware chain mandatory injection |
| **SDK-INV-12** | The agent loop is bounded: max steps, max iterations, max cost, max time | Resource governor enforcement at L6 |
| **SDK-INV-13** | Every action result includes provenance: what was done, to what, by which connector, at what time | Provenance tagging in connector invocation path |
| **SDK-INV-14** | Cross-language SDK behavioral parity verified by shared conformance test suite | CI pipeline with shared test vectors |

---

## 12. Summary of Architectural Constructs

| Construct | Section | Purpose |
|---|---|---|
| Seven-layer universal stack (L0–L6) | §1.3 | Orthogonal decomposition of connectivity concerns |
| OS abstraction with capability matrix | §2.1 | Platform-neutral system access |
| Universal protocol adapter with canonical request/response | §3.1–3.2 | Protocol-agnostic communication |
| Protocol selection algorithm (Algorithm 3.1) | §3.3 | Automatic optimal protocol routing |
| Universal authentication manager (Algorithm 4.1) | §4.1 | Support for all auth mechanisms in the ecosystem |
| Secure credential store with tiered storage | §4.2 | Defense-in-depth credential management |
| Cost governor with pre-flight budget enforcement | §4.3 | Cost control across all operations |
| API interaction engine with auto-discovery (Algorithm 5.1) | §5.2 | Programmatic integration with any API |
| Terminal engine with safe execution (Algorithm 5.2) | §5.3 | Command execution on any OS |
| Browser engine with intelligent interaction (Algorithm 5.3) | §5.4 | Full web application control |
| Desktop GUI engine with modality fallback | §5.5 | Native application automation |
| Google Workspace connector (Algorithm 6.1) | §6.2 | Complete Google ecosystem integration |
| IDE connector with multi-IDE strategy | §6.3 | Programmatic IDE control |
| MCP surface projection (Algorithm 6.2) | §6.4 | Universal tool discoverability |
| Universal agent loop with connector routing (Algorithm 7.1) | §7.1 | End-to-end orchestrated execution |
| Action routing algorithm (Algorithm 7.2) | §7.2 | Optimal connector/modality selection |
| Intelligent await with interleaving (Algorithm 7.3) | §7.3 | Long-running operation support |
| Deep page inspection (Algorithm 8.1) | §8.2 | Comprehensive web application observation |
| Universal login automation (Algorithm 8.2) | §8.3 | Authentication across any web application |
| Lazy tool loading (Algorithm 9.1) | §9.3 | Token-efficient tool context management |
| 14 mechanical invariants | §11 | Non-negotiable production quality enforcement |

---

This architecture transforms the Agent SDK from a library of API wrappers into a **universal, typed, protocol-agnostic connector fabric** that enables agentic systems to interact with any software system — programmatic APIs, command-line tools, web applications, desktop GUIs, IDEs, databases, cloud services, file systems, and network services — through a single, coherent, observable, secure, and bounded execution framework. Every interaction is routed through typed contracts, authenticated with least-privilege credentials, observed with correlated telemetry, bounded by resource governors, and recoverable through explicit failure semantics. The result is an SDK that does not merely connect to the software ecosystem but **operates within it as a first-class, reliable, auditable participant**.