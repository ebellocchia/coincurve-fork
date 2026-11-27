from contextlib import suppress
from hashlib import sha512
from os import urandom

import pytest

from coincurve.der import OBJECT_IDENTIFIER_TAG, SECP256K1_OID
from coincurve.ecdsa import deserialize_recoverable, recover
from coincurve.keys import PrivateKey, PublicKey, PublicKeyXOnly
from coincurve.utils import GROUP_ORDER_INT, bytes_to_int, int_to_bytes_padded, verify_signature

G = PublicKey(
    b"\x04y\xbef~\xf9\xdc\xbb\xacU\xa0b\x95\xce\x87\x0b\x07\x02\x9b"
    b"\xfc\xdb-\xce(\xd9Y\xf2\x81[\x16\xf8\x17\x98H:\xdaw&\xa3\xc4e"
    b"]\xa4\xfb\xfc\x0e\x11\x08\xa8\xfd\x17\xb4H\xa6\x85T\x19\x9cG"
    b"\xd0\x8f\xfb\x10\xd4\xb8"
)
n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


class TestPrivateKey:
    def test_public_key(self, samples):
        assert PrivateKey(samples["PRIVATE_KEY_BYTES"]).public_key.format() == samples["PUBLIC_KEY_COMPRESSED"]

    def test_xonly_pubkey(self, samples):
        assert (
            PrivateKey(samples["PRIVATE_KEY_BYTES"]).public_key_xonly.format() == samples["PUBLIC_KEY_COMPRESSED"][1:]
        )

    def test_signature_correct(self):
        private_key = PrivateKey()
        public_key = private_key.public_key

        message = urandom(200)
        signature = private_key.sign(message)

        assert verify_signature(signature, message, public_key.format(compressed=True))
        assert verify_signature(signature, message, public_key.format(compressed=False))

    def test_signature_deterministic(self, samples):
        assert PrivateKey(samples["PRIVATE_KEY_BYTES"]).sign(samples["MESSAGE"]) == samples["SIGNATURE"]

    def test_signature_invalid_hasher(self, samples):
        with pytest.raises(ValueError, match=r"Message hash must be 32 bytes long\."):
            PrivateKey().sign(samples["MESSAGE"], lambda x: sha512(x).digest())

    def test_signature_recoverable(self, samples):
        private_key = PrivateKey(samples["PRIVATE_KEY_BYTES"])
        assert (
            private_key.public_key.format()
            == PublicKey(
                recover(
                    samples["MESSAGE"],
                    deserialize_recoverable(private_key.sign_recoverable(samples["MESSAGE"])),
                )
            ).format()
        )

    def test_schnorr_signature(self):
        private_key = PrivateKey()
        message = urandom(32)

        # Message must be 32 bytes
        with pytest.raises(ValueError, match=r"Message must be 32 bytes long\."):
            private_key.sign_schnorr(message + b"\x01")

        # We can provide supplementary randomness
        sig = private_key.sign_schnorr(message, urandom(32))
        assert private_key.public_key_xonly.verify(sig, message)

        # Or not
        sig = private_key.sign_schnorr(message)
        assert private_key.public_key_xonly.verify(sig, message)

    def test_to_hex(self, samples):
        assert PrivateKey(samples["PRIVATE_KEY_BYTES"]).to_hex() == samples["PRIVATE_KEY_HEX"]

    def test_to_int(self, samples):
        assert PrivateKey(samples["PRIVATE_KEY_BYTES"]).to_int() == samples["PRIVATE_KEY_NUM"]

    def test_to_pem(self, samples):
        assert PrivateKey(samples["PRIVATE_KEY_BYTES"]).to_pem() == samples["PRIVATE_KEY_PEM"]

    def test_to_der(self, samples):
        assert PrivateKey(samples["PRIVATE_KEY_BYTES"]).to_der() == samples["PRIVATE_KEY_DER"]

    def test_from_hex(self, samples):
        assert PrivateKey.from_hex(samples["PRIVATE_KEY_HEX"]).secret == samples["PRIVATE_KEY_BYTES"]

    def test_from_int(self, samples):
        assert PrivateKey.from_int(samples["PRIVATE_KEY_NUM"]).secret == samples["PRIVATE_KEY_BYTES"]

    def test_from_pem(self, samples):
        assert PrivateKey.from_pem(samples["PRIVATE_KEY_PEM"]).secret == samples["PRIVATE_KEY_BYTES"]

    def test_from_der(self, samples):
        assert PrivateKey.from_der(samples["PRIVATE_KEY_DER"]).secret == samples["PRIVATE_KEY_BYTES"]

    def test_from_der_malformed(self, samples):
        der = samples["PRIVATE_KEY_DER"]

        # Long-form length pointing past the end of the data
        with pytest.raises(ValueError, match="Invalid DER"):
            PrivateKey.from_der(b"\x30\xfe" + b"\x00" * 64)

        # Every truncation must be rejected
        for i in range(len(der)):
            with pytest.raises(ValueError, match="Invalid DER"):
                PrivateKey.from_der(der[:i])

        # Replacing any byte with a length/tag edge case must either decode or raise ValueError
        for i in range(len(der)):
            for b in (0x00, 0x01, 0x02, 0x7F, 0x80, 0x81, 0x82, 0x84, 0xFF):
                with suppress(ValueError):
                    PrivateKey.from_der(der[:i] + bytes([b]) + der[i + 1 :])

    def test_from_der_wrong_curve(self, samples):
        der = samples["PRIVATE_KEY_DER"]
        curve_oid = bytes([OBJECT_IDENTIFIER_TAG, len(SECP256K1_OID)]) + SECP256K1_OID
        assert der.count(curve_oid) == 1

        # Same structure, but for secp384r1 (1.3.132.0.34)
        with pytest.raises(ValueError, match=r"Not a secp256k1 private key"):
            PrivateKey.from_der(der.replace(SECP256K1_OID, bytes([0x2B, 0x81, 0x04, 0x00, 0x22])))

        # Algorithm identifier without the curve OID
        no_curve = bytearray(der.replace(curve_oid, b""))
        no_curve[2] -= len(curve_oid)  # Outer SEQUENCE length
        no_curve[7] -= len(curve_oid)  # Algorithm identifier SEQUENCE length
        with pytest.raises(ValueError, match=r"expected OBJECT IDENTIFIER tag for curve"):
            PrivateKey.from_der(bytes(no_curve))

    def test_ecdh(self):
        a = PrivateKey()
        b = PrivateKey()

        assert a.ecdh(b.public_key.format()) == b.ecdh(a.public_key.format())

    def test_add(self):
        assert PrivateKey(b"\x01").add(b"\x09").to_int() == 10

    def test_scalar_too_long(self, samples):
        # libsecp256k1 would silently use only the first 32 bytes
        with pytest.raises(ValueError, match=r"Scalar must be at most 32 bytes long\."):
            PrivateKey(b"\x00" + samples["PRIVATE_KEY_BYTES"])
        with pytest.raises(ValueError, match=r"Scalar must be at most 32 bytes long\."):
            PrivateKey(b"\x01").add(b"\x00" * 32 + b"\x01")

    def test_add_update(self):
        private_key = PrivateKey(b"\x01")
        new_private_key = private_key.add(b"\x09", update=True)

        assert new_private_key.to_int() == 10
        assert private_key is new_private_key

    def test_multiply(self):
        assert PrivateKey(b"\x05").multiply(b"\x05").to_int() == 25

    def test_multiply_update(self):
        private_key = PrivateKey(b"\x05")
        new_private_key = private_key.multiply(b"\x05", update=True)

        assert new_private_key.to_int() == 25
        assert private_key is new_private_key


class TestPublicKey:
    def test_from_secret(self, samples):
        assert PublicKey.from_secret(samples["PRIVATE_KEY_BYTES"]).format() == samples["PUBLIC_KEY_COMPRESSED"]

    def test_from_point(self, samples):
        assert PublicKey.from_point(samples["PUBLIC_KEY_X"], samples["PUBLIC_KEY_Y"]).format() == samples.get(
            "PUBLIC_KEY_COMPRESSED"
        )

    def test_from_signature_and_message(self, samples):
        assert (
            PublicKey.from_secret(samples["PRIVATE_KEY_BYTES"]).format()
            == PublicKey.from_signature_and_message(samples["RECOVERABLE_SIGNATURE"], samples["MESSAGE"]).format()
        )

    def test_format(self, samples):
        assert PublicKey(samples["PUBLIC_KEY_UNCOMPRESSED"]).format(compressed=True) == samples.get(
            "PUBLIC_KEY_COMPRESSED"
        )
        assert PublicKey(samples["PUBLIC_KEY_COMPRESSED"]).format(compressed=False) == samples.get(
            "PUBLIC_KEY_UNCOMPRESSED"
        )

    def test_point(self, samples):
        assert PublicKey(samples["PUBLIC_KEY_COMPRESSED"]).point() == (
            samples["PUBLIC_KEY_X"],
            samples["PUBLIC_KEY_Y"],
        )

    def test_verify(self, samples):
        public_key = PublicKey(samples["PUBLIC_KEY_COMPRESSED"])
        assert public_key.verify(samples["SIGNATURE"], samples["MESSAGE"])

    def test_transform(self):
        x = urandom(32)
        k = urandom(32)
        point = G.multiply(x)

        assert point.add(k) == G.multiply(int_to_bytes_padded((bytes_to_int(x) + bytes_to_int(k)) % n))

    def test_add_scalar_too_long(self):
        with pytest.raises(ValueError, match=r"Scalar must be at most 32 bytes long\."):
            G.add(b"\x00" * 32 + b"\x01")

    def test_combine(self):
        a = PrivateKey().public_key
        b = PrivateKey().public_key

        assert PublicKey.combine_keys([a, b]) == a.combine([b])

    def test_combine_empty(self):
        with pytest.raises(ValueError, match=r"At least one public key is required\."):
            PublicKey.combine_keys([])


class TestXonlyPubKey:
    def test_parse_invalid(self, samples):
        # Must be 32 bytes
        with pytest.raises(ValueError, match=f"Secret scalar must be greater than 0 and less than {GROUP_ORDER_INT}"):
            PublicKeyXOnly.from_secret(bytes(33))

        # Must be an x coordinate for a valid point
        with pytest.raises(ValueError, match=r"The public key could not be parsed or is invalid\."):
            PublicKeyXOnly(samples["X_ONLY_PUBKEY_INVALID"])

        # Must be 32 bytes
        with pytest.raises(ValueError, match=r"X-only public key must be 32 bytes long\."):
            PublicKeyXOnly(samples["X_ONLY_PUBKEY"][:31])
        with pytest.raises(ValueError, match=r"X-only public key must be 32 bytes long\."):
            PublicKeyXOnly(samples["X_ONLY_PUBKEY"] + b"\x00")

    def test_roundtrip(self, samples):
        assert PublicKeyXOnly(samples["X_ONLY_PUBKEY"]).format() == samples["X_ONLY_PUBKEY"]
        assert PublicKeyXOnly(samples["PUBLIC_KEY_COMPRESSED"][1:]).format() == samples["PUBLIC_KEY_COMPRESSED"][1:]

        # Test __eq__
        assert PublicKeyXOnly(samples["X_ONLY_PUBKEY"]) == PublicKeyXOnly(samples["X_ONLY_PUBKEY"])

    def test_tweak(self):
        # Taken from BIP341 test vectors.
        # See github.com/bitcoin/bips/blob/6545b81022212a9f1c814f6ce1673e84bc02c910/bip-0341/wallet-test-vectors.json
        pubkey = PublicKeyXOnly(bytes.fromhex("d6889cb081036e0faefa3a35157ad71086b123b2b144b649798b494c300a961d"))
        pubkey.tweak_add(bytes.fromhex("b86e7be8f39bab32a6f2c0443abbc210f0edac0e2c53d501b36b64437d9c6c70"))
        assert pubkey.format() == bytes.fromhex("53a1f6e454df1aa2776a2814a721372d6258050de330b3c6d10ee8f4e0dda343")

    def test_tweak_scalar_too_long(self, samples):
        with pytest.raises(ValueError, match=r"Scalar must be at most 32 bytes long\."):
            PublicKeyXOnly(samples["X_ONLY_PUBKEY"]).tweak_add(b"\x00" * 32 + b"\x01")

    def test_parity(self):
        # Taken from BIP341 test vectors.
        # See github.com/bitcoin/bips/blob/6545b81022212a9f1c814f6ce1673e84bc02c910/bip-0341/wallet-test-vectors.json
        pubkey = PublicKeyXOnly(bytes.fromhex("187791b6f712a8ea41c8ecdd0ee77fab3e85263b37e1ec18a3651926b3a6cf27"))
        pubkey.tweak_add(bytes.fromhex("cbd8679ba636c1110ea247542cfbd964131a6be84f873f7f3b62a777528ed001"))
        assert pubkey.format() == bytes.fromhex("147c9c57132f6e7ecddba9800bb0c4449251c92a1e60371ee77557b6620f3ea3")
        assert pubkey.parity

        pubkey = PublicKeyXOnly(bytes.fromhex("93478e9488f956df2396be2ce6c5cced75f900dfa18e7dabd2428aae78451820"))
        pubkey.tweak_add(bytes.fromhex("6af9e28dbf9d6aaf027696e2598a5b3d056f5fd2355a7fd5a37a0e5008132d30"))
        assert pubkey.format() == bytes.fromhex("e4d810fd50586274face62b8a807eb9719cef49c04177cc6b76a9a4251d5450e")
        assert not pubkey.parity


if __name__ == "__main__":
    pytest.main(["-v", __file__])
