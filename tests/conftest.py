"""Offline test configuration.

Blocks outbound sockets so a regression cannot pass by calling Yahoo Finance.
yfinance's HTTP stack is not used by these tests; entry points are mocked.
"""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Refuse TCP connects for the duration of each test."""

    def _blocked(self, address):
        raise RuntimeError(f"Network access is disabled in offline tests ({address})")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
