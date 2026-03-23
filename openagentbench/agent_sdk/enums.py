"""Enumerations for the universal agent SDK module."""

from __future__ import annotations

from enum import Enum


class OsType(str, Enum):
    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"
    IOS = "ios"
    ANDROID = "android"
    BROWSER = "browser"
    EMBEDDED = "embedded"
    UNKNOWN = "unknown"


class InteractionModality(str, Enum):
    API = "api"
    CLI = "cli"
    GUI = "gui"
    BROWSER = "browser"


class ProtocolName(str, Enum):
    HTTP = "http"
    HTTP2 = "http2"
    GRPC = "grpc"
    JSONRPC = "jsonrpc"
    WEBSOCKET = "websocket"
    TCP = "tcp"
    SSH = "ssh"
    MCP = "mcp"
    IPC = "ipc"
    PIPE = "pipe"


class AuthType(str, Enum):
    OAUTH2_AUTHORIZATION_CODE = "oauth2_authorization_code"
    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"
    OAUTH2_DEVICE_FLOW = "oauth2_device_flow"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    MTLS = "mtls"
    SSH_KEY = "ssh_key"
    KERBEROS = "kerberos"
    SAML = "saml"
    BROWSER_SESSION = "browser_session"
    DESKTOP_KEYCHAIN = "desktop_keychain"
    CLOUD_IAM = "cloud_iam"
    CUSTOM = "custom"
    NONE = "none"


class ConnectorHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class OperationState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResourceScope(str, Enum):
    REQUEST = "request"
    SESSION = "session"
    USER = "user"
    TENANT = "tenant"
    GLOBAL = "global"


class SafetyLevel(str, Enum):
    TRUSTED = "trusted"
    CAUTIOUS = "cautious"
    RESTRICTIVE = "restrictive"


class ConnectorDomain(str, Enum):
    CORE = "core"
    MEMORY = "memory"
    CONTEXT = "context"
    RETRIEVAL = "retrieval"
    TOOLS = "tools"
    BROWSER = "browser"
    TERMINAL = "terminal"
    VISION = "vision"
    A2A = "a2a"
    MCP = "mcp"
    JSONRPC = "jsonrpc"
    FUNCTION = "function"
    CUSTOM = "custom"


class ProviderTarget(str, Enum):
    OPENAI = "openai"
    VLLM = "vllm"


__all__ = [
    "AuthType",
    "ConnectorDomain",
    "ConnectorHealth",
    "InteractionModality",
    "OperationState",
    "OsType",
    "ProtocolName",
    "ProviderTarget",
    "ResourceScope",
    "SafetyLevel",
]
