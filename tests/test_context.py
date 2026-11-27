import pytest

from coincurve.context import Context


class TestContext:
    def test_reseed(self):
        context = Context()
        context.reseed()
        context.reseed(b"\x01" * 32)

    def test_seed_invalid_length(self):
        with pytest.raises(ValueError, match=r"Seed must be 32 bytes long\."):
            Context(seed=b"\x01" * 16)
        with pytest.raises(ValueError, match=r"Seed must be 32 bytes long\."):
            Context().reseed(b"")
        with pytest.raises(ValueError, match=r"Seed must be 32 bytes long\."):
            Context().reseed(b"\x01" * 33)
