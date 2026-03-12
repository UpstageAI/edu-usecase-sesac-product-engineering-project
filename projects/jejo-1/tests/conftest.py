"""
Test configuration for stable local/CI execution.
"""
from __future__ import annotations

import asyncio
import inspect
import os


# Disable external tracing uploads during tests.
os.environ.setdefault("LANGSMITH_TRACING", "false")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("AUTO_INGEST_ON_STARTUP", "false")
os.environ.setdefault("VECTOR_DB_DIR", "/tmp/trade-onboarding-agent-test-vectorstore")


def pytest_pyfunc_call(pyfuncitem):
    """
    Minimal asyncio runner to support async test functions without extra plugins.
    """
    testfunction = pyfuncitem.obj
    if not inspect.iscoroutinefunction(testfunction):
        return None

    kwargs = {
        argname: pyfuncitem.funcargs[argname]
        for argname in pyfuncitem._fixtureinfo.argnames
    }
    asyncio.run(testfunction(**kwargs))
    return True
