"""Core identity management using cryptographic keys for node authentication."""

import uuid
import base64
import nacl.signing
import nacl.encoding

__all__ = ['Identity', 'verify_signature']


class Identity:
    """Identity management with Ed25519 cryptographic keys.
    
    This class provides cryptographic identity functionality including
    key generation, message signing, and decentralized identifier (DID)
    generation for secure node authentication.
    
    Attributes:
        signing_key (nacl.signing.SigningKey): Private signing key.
        verify_key (nacl.signing.VerifyKey): Public verification key.
        did (str): Decentralized identifier generated from public key.
    """
    
    def __init__(self):
        """Initialize identity with Ed25519 key pair.
        
        Generates a new Ed25519 key pair and creates a corresponding
        decentralized identifier (DID).
        """
        # Create Ed25519 key pair
        self.signing_key = nacl.signing.SigningKey.generate()
        self.verify_key = self.signing_key.verify_key
        self.did = self.generate_did()
    
    def generate_did(self) -> str:
        """Generate Decentralized Identifier (DID) from public key.
        
        Creates a DID using the first 20 characters of the base64-encoded
        public key as a unique identifier.
        
        Returns:
            str: Generated DID string in format 'did:ecn:...'.
        """
        # Use base64 encoded public key as part of DID
        pubkey_b64 = self.verify_key.encode(
            encoder=nacl.encoding.Base64Encoder
        ).decode("utf-8")
        return f"did:ecn:{pubkey_b64[:20]}"
    
    def sign(self, message: str) -> str:
        """Sign a message with the private key.
        
        Args:
            message (str): Message to sign.
            
        Returns:
            str: Base64 encoded signature.
        """
        signed = self.signing_key.sign(message.encode("utf-8"))
        return base64.b64encode(signed.signature).decode("utf-8")
    
    def get_public_key(self) -> str:
        """Get the public key as base64 string.
        
        Returns:
            str: Base64 encoded public key.
        """
        return self.verify_key.encode(
            encoder=nacl.encoding.Base64Encoder
        ).decode("utf-8")
    
    def __repr__(self):
        """Return string representation of the identity.
        
        Returns:
            str: Formatted string showing the DID.
        """
        return f"<Identity DID={self.did}>"


def verify_signature(public_key_b64: str, message: str, signature_b64: str) -> bool:
    """Verify a signature using public key.
    
    Public utility function for verifying message signatures without
    requiring an Identity instance.
    
    Args:
        public_key_b64 (str): Base64 encoded public key.
        message (str): Original message that was signed.
        signature_b64 (str): Base64 encoded signature to verify.
        
    Returns:
        bool: True if signature is valid, False otherwise.
    """
    try:
        verify_key = nacl.signing.VerifyKey(
            public_key_b64.encode("utf-8"),
            encoder=nacl.encoding.Base64Encoder
        )
        signature = base64.b64decode(signature_b64)
        verify_key.verify(message.encode("utf-8"), signature)
        return True
    except Exception as e:
        print(f"[Verify] Signature invalid: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    identity = Identity()
    test_message = "task_abc_xyz"
    signature = identity.sign(test_message)
    print("Signature:", signature)
    print("Verify:", verify_signature(
        identity.get_public_key(), test_message, signature))
