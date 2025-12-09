"""
Post-Quantum Signature Module (PQ Signatures)
Implements Dilithium (ML-DSA) and SPHINCS+ for quantum-resistant authentication
Part of hybrid SKG+PQC system for thesis work
"""

import os
import time
import hashlib
from typing import Tuple, Optional, Dict, Any

try:
    # Try to import pqcrypto dilithium implementation
    from pqcrypto.sign.dilithium2 import generate_keypair, sign, verify
    DILITHIUM_AVAILABLE = True
    DILITHIUM_LEVEL = "Dilithium2"
except ImportError:
    try:
        from pqcrypto.sign.dilithium3 import generate_keypair, sign, verify
        DILITHIUM_AVAILABLE = True
        DILITHIUM_LEVEL = "Dilithium3"
    except ImportError:
        DILITHIUM_AVAILABLE = False
        print("Warning: pqcrypto dilithium not available. Using simulated signatures.")


class PQCDilithiumSignature:
    """
    Post-Quantum Signature using Dilithium (ML-DSA)
    Provides quantum-resistant authentication for hybrid SKG+PQC
    
    Dilithium is NIST's standardized lattice-based signature scheme (ML-DSA)
    Balanced performance for transcript authentication
    """
    
    def __init__(self, security_level: str = "dilithium3"):
        """
        Initialize PQ signature scheme
        
        Args:
            security_level: 'dilithium2', 'dilithium3', or 'dilithium5'
        """
        self.security_level = security_level
        self.dilithium_available = DILITHIUM_AVAILABLE
        self.public_key = None
        self.secret_key = None
        self.metrics = {
            'keygen_time': 0.0,
            'sign_time': 0.0,
            'verify_time': 0.0,
            'pk_size': 0,
            'sk_size': 0,
            'sig_size': 0
        }
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate a Dilithium signing keypair
        
        Returns:
            (public_key, secret_key) tuple
        """
        start_time = time.time()
        
        if self.dilithium_available:
            self.public_key, self.secret_key = generate_keypair()
        else:
            # Simulated keypair (NOT SECURE - for testing only)
            self.public_key = os.urandom(1312)  # Approximate Dilithium2 pk size
            self.secret_key = os.urandom(2528)  # Approximate Dilithium2 sk size
        
        self.metrics['keygen_time'] = time.time() - start_time
        self.metrics['pk_size'] = len(self.public_key)
        self.metrics['sk_size'] = len(self.secret_key)
        
        return self.public_key, self.secret_key
    
    def sign_transcript(self, 
                       transcript: bytes,
                       secret_key: Optional[bytes] = None) -> bytes:
        """
        Sign a transcript with Dilithium
        
        Args:
            transcript: Data to sign (full protocol transcript)
            secret_key: Signing key (uses instance key if not provided)
            
        Returns:
            Digital signature
        """
        start_time = time.time()
        
        if secret_key is None:
            secret_key = self.secret_key
            
        if self.dilithium_available:
            signature = sign(secret_key, transcript)
        else:
            # Simulated signature (NOT SECURE - for testing only)
            # Hash transcript and secret key to create deterministic signature
            sig_input = transcript + secret_key
            signature = hashlib.sha512(sig_input).digest() + os.urandom(2420)
        
        self.metrics['sign_time'] = time.time() - start_time
        self.metrics['sig_size'] = len(signature)
        
        return signature
    
    def verify_signature(self,
                        transcript: bytes,
                        signature: bytes,
                        public_key: Optional[bytes] = None) -> bool:
        """
        Verify a Dilithium signature
        
        Args:
            transcript: Signed data
            signature: Signature to verify
            public_key: Verification key (uses instance key if not provided)
            
        Returns:
            True if signature is valid, False otherwise
        """
        start_time = time.time()
        
        if public_key is None:
            public_key = self.public_key
        
        try:
            if self.dilithium_available:
                # pqcrypto verify returns None on success, raises on failure
                verify(public_key, transcript, signature)
                valid = True
            else:
                # Simulated verification (always returns True for demo)
                valid = True
        except Exception as e:
            valid = False
        
        self.metrics['verify_time'] = time.time() - start_time
        
        return valid
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get signature performance metrics"""
        return self.metrics.copy()
    
    def get_overhead_stats(self) -> Dict[str, Any]:
        """
        Calculate overhead statistics for thesis evaluation
        
        Returns:
            Dictionary with overhead analysis
        """
        return {
            'public_key_bytes': self.metrics['pk_size'],
            'secret_key_bytes': self.metrics['sk_size'],
            'signature_bytes': self.metrics['sig_size'],
            'keygen_latency_ms': self.metrics['keygen_time'] * 1000,
            'sign_latency_ms': self.metrics['sign_time'] * 1000,
            'verify_latency_ms': self.metrics['verify_time'] * 1000,
            'security_level': self.security_level,
            'implementation': 'pqcrypto' if self.dilithium_available else 'simulated'
        }


class SPHINCSPlusSignature:
    """
    SPHINCS+ - Conservative hash-based signature
    
    Stateless and based only on hash functions (conservative assumptions)
    Larger signatures and slower than Dilithium
    """
    
    def __init__(self, variant: str = "sphincs-shake-128f"):
        """
        Initialize SPHINCS+
        
        Args:
            variant: SPHINCS+ variant (e.g., 'sphincs-shake-128f' for fast)
        """
        self.variant = variant
        self.metrics = {
            'keygen_time': 0.0,
            'sign_time': 0.0,
            'verify_time': 0.0,
            'pk_size': 32,
            'sig_size': 17088  # SPHINCS+-128f approximate
        }
        print("Note: SPHINCS+ is conservative hash-based. Implementation placeholder.")
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate SPHINCS+ keypair (simulated)"""
        start_time = time.time()
        public_key = os.urandom(32)  # Small public key
        secret_key = os.urandom(64)  # Small secret key
        self.metrics['keygen_time'] = time.time() - start_time
        return public_key, secret_key
    
    def sign_transcript(self, transcript: bytes, secret_key: bytes) -> bytes:
        """Sign with SPHINCS+ (simulated)"""
        start_time = time.time()
        signature = os.urandom(17088)  # Large signature
        self.metrics['sign_time'] = time.time() - start_time
        return signature
    
    def verify_signature(self, transcript: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify SPHINCS+ signature (simulated)"""
        start_time = time.time()
        valid = True  # Simulated always valid
        self.metrics['verify_time'] = time.time() - start_time
        return valid


class HybridSignature:
    """
    Hybrid signature combining classical ECDSA/Ed25519 with Dilithium
    
    Useful during transition period for interoperability
    Provides both classical and post-quantum security
    """
    
    def __init__(self):
        """Initialize hybrid signature scheme"""
        self.dilithium = PQCDilithiumSignature()
        self.metrics = {
            'keygen_time': 0.0,
            'sign_time': 0.0,
            'verify_time': 0.0,
            'total_sig_size': 0
        }
        print("Hybrid signature: Classical + Dilithium")
    
    def generate_keypair(self) -> Tuple[Dict[str, bytes], Dict[str, bytes]]:
        """
        Generate hybrid keypairs
        
        Returns:
            (public_keys_dict, secret_keys_dict)
        """
        start_time = time.time()
        
        # Generate classical Ed25519 keypair (simulated)
        from cryptography.hazmat.primitives.asymmetric import ed25519
        classical_sk = ed25519.Ed25519PrivateKey.generate()
        classical_pk = classical_sk.public_key()
        
        # Generate PQ Dilithium keypair
        pq_pk, pq_sk = self.dilithium.generate_keypair()
        
        self.metrics['keygen_time'] = time.time() - start_time
        
        public_keys = {
            'classical': classical_pk.public_bytes_raw(),
            'pq': pq_pk
        }
        
        secret_keys = {
            'classical': classical_sk.private_bytes_raw(),
            'pq': pq_sk,
            'classical_key_obj': classical_sk  # Keep for signing
        }
        
        return public_keys, secret_keys
    
    def sign_transcript(self, transcript: bytes, secret_keys: Dict[str, Any]) -> Dict[str, bytes]:
        """
        Create hybrid signature (both classical and PQ)
        
        Args:
            transcript: Data to sign
            secret_keys: Dictionary with both classical and PQ secret keys
            
        Returns:
            Dictionary with both signatures
        """
        start_time = time.time()
        
        # Classical Ed25519 signature
        classical_sig = secret_keys['classical_key_obj'].sign(transcript)
        
        # PQ Dilithium signature
        pq_sig = self.dilithium.sign_transcript(transcript, secret_keys['pq'])
        
        self.metrics['sign_time'] = time.time() - start_time
        self.metrics['total_sig_size'] = len(classical_sig) + len(pq_sig)
        
        return {
            'classical': classical_sig,
            'pq': pq_sig
        }
    
    def verify_signature(self,
                        transcript: bytes,
                        signatures: Dict[str, bytes],
                        public_keys: Dict[str, bytes]) -> Dict[str, bool]:
        """
        Verify hybrid signature
        
        Args:
            transcript: Signed data
            signatures: Dictionary with both signatures
            public_keys: Dictionary with both public keys
            
        Returns:
            Dictionary with verification results for each signature
        """
        start_time = time.time()
        
        # Verify classical signature
        from cryptography.hazmat.primitives.asymmetric import ed25519
        classical_pk = ed25519.Ed25519PublicKey.from_public_bytes(public_keys['classical'])
        try:
            classical_pk.verify(signatures['classical'], transcript)
            classical_valid = True
        except:
            classical_valid = False
        
        # Verify PQ signature
        pq_valid = self.dilithium.verify_signature(transcript, signatures['pq'], public_keys['pq'])
        
        self.metrics['verify_time'] = time.time() - start_time
        
        return {
            'classical': classical_valid,
            'pq': pq_valid,
            'both': classical_valid and pq_valid
        }


class TranscriptBuilder:
    """
    Helper class to build authenticated transcripts for signing
    
    Includes all protocol messages, channel commitments, and context
    """
    
    def __init__(self):
        """Initialize transcript builder"""
        self.elements = []
    
    def add_element(self, label: str, data: bytes):
        """Add an element to the transcript"""
        self.elements.append((label, data))
    
    def add_context(self, context: Dict[str, Any]):
        """Add context information to transcript"""
        for key, value in context.items():
            self.add_element(f"context.{key}", str(value).encode())
    
    def add_channel_commitment(self, commitment: bytes):
        """Add channel state commitment H(CSI||pilot||nonce)"""
        self.add_element("channel_commitment", commitment)
    
    def add_pq_kem_ciphertext(self, ciphertext: bytes):
        """Add PQ KEM ciphertext to transcript"""
        self.add_element("pq_kem_ciphertext", ciphertext)
    
    def add_reconciliation_data(self, syndrome: bytes):
        """Add reconciliation helper data to transcript"""
        self.add_element("reconciliation_syndrome", syndrome)
    
    def finalize(self) -> bytes:
        """
        Finalize and return the complete transcript for signing
        
        Returns:
            Serialized transcript ready for signing
        """
        transcript_parts = []
        for label, data in self.elements:
            # Format: length(label) || label || length(data) || data
            label_bytes = label.encode()
            transcript_parts.append(len(label_bytes).to_bytes(2, 'big'))
            transcript_parts.append(label_bytes)
            transcript_parts.append(len(data).to_bytes(4, 'big'))
            transcript_parts.append(data)
        
        return b"".join(transcript_parts)
    
    def hash_transcript(self) -> bytes:
        """Get hash of transcript for compact commitment"""
        return hashlib.sha384(self.finalize()).digest()


def demo_pq_signatures():
    """Demonstrate PQ signature operation"""
    print("=== PQ Signature Demonstration ===\n")
    
    # 1. Dilithium signature
    print("1. Dilithium (ML-DSA) Signature")
    print("   " + "="*50)
    dilithium = PQCDilithiumSignature(security_level="dilithium3")
    
    print("   Generating keypair...")
    pk, sk = dilithium.generate_keypair()
    print(f"   Public key: {len(pk)} bytes")
    print(f"   Secret key: {len(sk)} bytes")
    print(f"   Keygen time: {dilithium.metrics['keygen_time']*1000:.3f} ms")
    
    # Build a transcript
    print("\n   Building protocol transcript...")
    transcript_builder = TranscriptBuilder()
    transcript_builder.add_element("protocol", b"SKG+PQC-v1.0")
    transcript_builder.add_element("role", b"eNodeB")
    transcript_builder.add_context({'cell_id': 'CELL_001', 'timestamp': int(time.time())})
    transcript_builder.add_channel_commitment(os.urandom(48))
    transcript_builder.add_pq_kem_ciphertext(os.urandom(768))
    transcript = transcript_builder.finalize()
    print(f"   Transcript size: {len(transcript)} bytes")
    
    # Sign transcript
    print("\n   Signing transcript...")
    signature = dilithium.sign_transcript(transcript, sk)
    print(f"   Signature size: {len(signature)} bytes")
    print(f"   Sign time: {dilithium.metrics['sign_time']*1000:.3f} ms")
    
    # Verify signature
    print("\n   Verifying signature...")
    valid = dilithium.verify_signature(transcript, signature, pk)
    print(f"   Valid: {valid}")
    print(f"   Verify time: {dilithium.metrics['verify_time']*1000:.3f} ms")
    
    # Overhead statistics
    print("\n   Overhead Statistics:")
    stats = dilithium.get_overhead_stats()
    for key, value in stats.items():
        print(f"     {key}: {value}")
    
    # 2. Hybrid signature demo
    print("\n\n2. Hybrid Signature (Ed25519 + Dilithium)")
    print("   " + "="*50)
    hybrid = HybridSignature()
    
    print("   Generating hybrid keypairs...")
    pub_keys, sec_keys = hybrid.generate_keypair()
    print(f"   Classical PK: {len(pub_keys['classical'])} bytes")
    print(f"   PQ PK: {len(pub_keys['pq'])} bytes")
    
    print("\n   Signing with hybrid scheme...")
    hybrid_sigs = hybrid.sign_transcript(transcript, sec_keys)
    print(f"   Classical sig: {len(hybrid_sigs['classical'])} bytes")
    print(f"   PQ sig: {len(hybrid_sigs['pq'])} bytes")
    print(f"   Total: {len(hybrid_sigs['classical']) + len(hybrid_sigs['pq'])} bytes")
    
    print("\n   Verifying hybrid signatures...")
    verify_results = hybrid.verify_signature(transcript, hybrid_sigs, pub_keys)
    print(f"   Classical valid: {verify_results['classical']}")
    print(f"   PQ valid: {verify_results['pq']}")
    print(f"   Both valid: {verify_results['both']}")


if __name__ == "__main__":
    demo_pq_signatures()
