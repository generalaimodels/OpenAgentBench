"""Environment-backed configuration for realtime integration tests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(slots=True, frozen=True)
class LiveTestConfig:
    run_live_realtime_tests: bool
    database_url: str | None
    live_test_vendor: str
    max_budget_usd: float
    openai_api_key: str | None
    gemini_api_key: str | None
    openai_model: str
    gemini_model: str
    run_id: str
    cleanup_rows: bool

    @classmethod
    def from_env(cls) -> "LiveTestConfig":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return cls(
            run_live_realtime_tests=os.getenv("RUN_LIVE_REALTIME_TESTS", "0") == "1",
            database_url=os.getenv("TEST_DATABASE_URL"),
            live_test_vendor=os.getenv("LIVE_TEST_VENDOR", "all").strip().lower() or "all",
            max_budget_usd=float(os.getenv("LIVE_TEST_MAX_BUDGET_USD", "3.0")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            openai_model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-mini"),
            gemini_model=os.getenv(
                "GEMINI_LIVE_MODEL",
                "gemini-2.5-flash-native-audio-preview-12-2025",
            ),
            run_id=os.getenv("LIVE_TEST_RUN_ID", timestamp),
            cleanup_rows=os.getenv("LIVE_TEST_CLEANUP", "0") == "1",
        )

    def vendor_enabled(self, vendor: str) -> bool:
        selected = self.live_test_vendor
        return selected in {"all", vendor}

    def has_vendor_key(self, vendor: str) -> bool:
        if vendor == "openai":
            return bool(self.openai_api_key)
        if vendor == "gemini":
            return bool(self.gemini_api_key)
        return False

    def live_skip_reason(self, vendor: str) -> str | None:
        if not self.run_live_realtime_tests:
            return "live realtime tests are disabled; set RUN_LIVE_REALTIME_TESTS=1"
        if not self.vendor_enabled(vendor):
            return f"vendor {vendor!r} is not enabled by LIVE_TEST_VENDOR={self.live_test_vendor!r}"
        if not self.has_vendor_key(vendor):
            return f"missing API key for {vendor}; check environment"
        if not self.database_url:
            return "TEST_DATABASE_URL is required for live realtime persistence tests"
        return None

    def postgres_skip_reason(self) -> str | None:
        if not self.database_url:
            return "TEST_DATABASE_URL is not set"
        return None
