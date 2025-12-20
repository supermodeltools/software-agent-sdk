"""Integration test to verify the agent server works with wsproto.

Fixes:
- Port selection TOCTOU race by pre-binding a socket and passing its FD to uvicorn
- Replace heavy readiness probe (/docs) with lightweight /alive
"""

import asyncio
import json
import multiprocessing
import os
import socket
import time
from collections.abc import Generator

import pytest
import requests
import websockets


def _prebind_inet_socket() -> tuple[socket.socket, int]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Avoid lingering TIME_WAIT issues in CI; child will adopt the same FD
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    s.listen(128)
    return s, s.getsockname()[1]


def run_agent_server_with_fd(fd: int, api_key: str) -> None:
    # Configure API key for auth dependency (support both new and legacy envs)
    os.environ["OH_SESSION_API_KEYS"] = f'["{api_key}"]'
    os.environ["SESSION_API_KEY"] = api_key
    # Start uvicorn directly so we can adopt the pre-bound socket via fd
    # Keep configuration aligned with __main__.py
    import uvicorn

    from openhands.agent_server.logging_config import LOGGING_CONFIG

    uvicorn.run(
        "openhands.agent_server.api:api",
        fd=fd,
        log_config=LOGGING_CONFIG,
        ws="wsproto",
        # Host/port are ignored when fd is provided; keep defaults implicit
    )


@pytest.fixture(scope="session")
def agent_server() -> Generator[dict[str, object], None, None]:
    sock, port = _prebind_inet_socket()
    api_key = "test-wsproto-key"

    # Fork a child that adopts the listening socket via its file descriptor
    process = multiprocessing.Process(
        target=run_agent_server_with_fd, args=(sock.fileno(), api_key)
    )
    process.start()

    # Parent no longer needs its copy of the FD; child inherited it on fork
    try:
        sock.close()
    except OSError:
        pass

    # Lightweight readiness loop using /alive, with a slightly more forgiving timeout
    start = time.time()
    last_err: Exception | None = None
    for _ in range(60):
        try:
            resp = requests.get(f"http://127.0.0.1:{port}/alive", timeout=3)
            if resp.status_code == 200:
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = e
        time.sleep(0.25)
    else:
        process.terminate()
        process.join()
        reason = f"last error: {last_err!r}" if last_err else "no response"
        pytest.fail(
            "Agent server failed to start on port "
            f"{port} within {int(time.time() - start)}s ({reason})"
        )

    yield {"port": port, "api_key": api_key}

    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join()


def test_agent_server_starts_with_wsproto(agent_server: dict[str, object]) -> None:
    response = requests.get(f"http://127.0.0.1:{agent_server['port']}/docs")
    assert response.status_code == 200
    assert (
        "OpenHands Agent Server" in response.text or "swagger" in response.text.lower()
    )


@pytest.mark.asyncio
async def test_agent_server_websocket_with_wsproto(
    agent_server: dict[str, object],
) -> None:
    port = agent_server["port"]
    api_key = agent_server["api_key"]

    response = requests.post(
        f"http://127.0.0.1:{port}/api/conversations",
        headers={"X-Session-API-Key": str(api_key)},
        json={
            "agent": {
                "llm": {
                    "usage_id": "test-llm",
                    "model": "test-provider/test-model",
                    "api_key": "test-key",
                },
                "tools": [],
            },
            "workspace": {"working_dir": "/tmp/test-workspace"},
        },
    )
    assert response.status_code in [200, 201]
    conversation_id = response.json()["id"]

    ws_url = (
        f"ws://127.0.0.1:{port}/sockets/events/{conversation_id}"
        f"?session_api_key={api_key}&resend_all=true"
    )

    async with websockets.connect(ws_url, open_timeout=5) as ws:
        try:
            first = await asyncio.wait_for(ws.recv(), timeout=2)
            assert first is not None
        except TimeoutError:
            pass

        await ws.send(
            json.dumps({"role": "user", "content": "Hello from wsproto test"})
        )
