"""
Post-Quantum Key Encapsulation Module (PQ KEM)
Implements Kyber (ML-KEM) for quantum-resistant key exchange
Part of hybrid SKG+PQC system for thesis work
"""

import os
import time
from typing import Tuple, Optional
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

try:
    # Try to import pqcrypto kyber implementation
    from pqcrypto.kem.kyber512 import generate_keypair, encrypt, decrypt
    KYBER_AVAILABLE = True
    KYBER_LEVEL = "Kyber512"
except ImportError:
    try:
        from pqcrypto.kem.kyber768 import generate_keypair, encrypt, decrypt
        KYBER_AVAILABLE = True
        KYBER_LEVEL = "Kyber768"
    except ImportError:
        KYBER_AVAILABLE = False
        print("Warning: pqcrypto not available. Using simulated Kyber for demonstration.")


class PQCKyberKEM:
    """
    Post-Quantum KEM using Kyber (ML-KEM)
    Provides quantum-resistant key encapsulation for hybrid SKG+PQC
    """
    
    def __init__(self, security_level: str = "kyber768"):
        """
        Initialize PQ KEM with specified security level
        
        Args:
            security_level: 'kyber512', 'kyber768', or 'kyber1024'
        """
        self.security_level = security_level
        self.kyber_available = KYBER_AVAILABLE
        self.public_key = None
        self.secret_key = None
        self.metrics = {
            'keygen_time': 0.0,
            'encaps_time': 0.0,
            'decaps_time': 0.0,
            'pk_size': 0,
            'ct_size': 0,
            'ss_size': 0
        }
        
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate a Kyber keypair
        
        Returns:
            (public_key, secret_key) tuple
        """
        start_time = time.time()
        
        if self.kyber_available:
            self.public_key, self.secret_key = generate_keypair()
        else:
            # Simulated keypair for demonstration (NOT SECURE - for testing only)
            self.public_key = os.urandom(800)  # Approximate Kyber512 pk size
            self.secret_key = os.urandom(1632)  # Approximate Kyber512 sk size
            
        self.metrics['keygen_time'] = time.time() - start_time
        self.metrics['pk_size'] = len(self.public_key)
        
        return self.public_key, self.secret_key
    
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulate a shared secret using receiver's public key
        
        Args:
            public_key: Receiver's Kyber public key
            
        Returns:
            (ciphertext, shared_secret) tuple
        """
        start_time = time.time()
        
        if self.kyber_available:
            ciphertext, shared_secret = encrypt(public_key)
        else:
            # Simulated encapsulation (NOT SECURE - for testing only)
            ciphertext = os.urandom(768)  # Approximate Kyber512 ct size
            # Use HKDF to derive a consistent shared secret from ciphertext
            kdf = HKDF(
                algorithm=hashes.SHA384(),
                length=32,
                salt=b'simulated_kyber_salt',
                info=b'simulated_kyber_encaps'
            )
            shared_secret = kdf.derive(ciphertext + public_key)
            
        self.metrics['encaps_time'] = time.time() - start_time
        self.metrics['ct_size'] = len(ciphertext)
        self.metrics['ss_size'] = len(shared_secret)
        
        return ciphertext, shared_secret
    
    def decapsulate(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """
        Decapsulate shared secret using secret key
        
        Args:
            ciphertext: Kyber ciphertext from encapsulation
            secret_key: Receiver's Kyber secret key
            
        Returns:
            shared_secret: The encapsulated shared secret
        """
        start_time = time.time()
        
        if self.kyber_available:
            shared_secret = decrypt(secret_key, ciphertext)
        else:
            # Simulated decapsulation (NOT SECURE - for testing only)
            # Must produce same result as encapsulation for consistency
            # In real implementation, this would use the secret key
            public_key = self.public_key if self.public_key else os.urandom(800)
            kdf = HKDF(
                algorithm=hashes.SHA384(),
                length=32,
                salt=b'simulated_kyber_salt',
                info=b'simulated_kyber_encaps'
            )
            shared_secret = kdf.derive(ciphertext + public_key)
            
        self.metrics['decaps_time'] = time.time() - start_time
        
        return shared_secret
    
    def get_metrics(self) -> dict:
        """
        Get performance metrics for the KEM operations
        
        Returns:
            Dictionary with timing and size metrics
        """
        return self.metrics.copy()
    
    def get_overhead_stats(self) -> dict:
        """
        Calculate overhead statistics for thesis evaluation
        
        Returns:
            Dictionary with overhead analysis
        """
        return {
            'public_key_bytes': self.metrics['pk_size'],
            'ciphertext_bytes': self.metrics['ct_size'],
            'shared_secret_bytes': self.metrics['ss_size'],
            'total_transmission_bytes': self.metrics['pk_size'] + self.metrics['ct_size'],
            'keygen_latency_ms': self.metrics['keygen_time'] * 1000,
            'encaps_latency_ms': self.metrics['encaps_time'] * 1000,
            'decaps_latency_ms': self.metrics['decaps_time'] * 1000,
            'total_latency_ms': (self.metrics['keygen_time'] + 
                                self.metrics['encaps_time'] + 
                                self.metrics['decaps_time']) * 1000,
            'security_level': self.security_level,
            'implementation': 'pqcrypto' if self.kyber_available else 'simulated'
        }


class FrodoKEM:
    """
    Conservative lattice-agnostic KEM alternative (FrodoKEM)
    Larger overhead but different security assumptions than Kyber
    """
    
    def __init__(self):
        """Initialize FrodoKEM (placeholder for conservative option)"""
        self.metrics = {
            'keygen_time': 0.0,
            'encaps_time': 0.0,
            'decaps_time': 0.0,
            'pk_size': 0,
            'ct_size': 0
        }
        print("Note: FrodoKEM is a conservative alternative. Implementation placeholder.")
        
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate FrodoKEM keypair (simulated)"""
        start_time = time.time()
        # FrodoKEM has larger keys (e.g., FrodoKEM-640: ~9KB pk, ~16KB ct)
        public_key = os.urandom(9616)  # FrodoKEM-640 approximate
        secret_key = os.urandom(19888)
        self.metrics['keygen_time'] = time.time() - start_time
        self.metrics['pk_size'] = len(public_key)
        return public_key, secret_key
        
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Encapsulate using FrodoKEM (simulated)"""
        start_time = time.time()
        ciphertext = os.urandom(9720)  # FrodoKEM-640 approximate
        shared_secret = os.urandom(16)  # 128-bit shared secret
        self.metrics['encaps_time'] = time.time() - start_time
        self.metrics['ct_size'] = len(ciphertext)
        return ciphertext, shared_secret
        
    def decapsulate(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """Decapsulate using FrodoKEM (simulated)"""
        start_time = time.time()
        shared_secret = os.urandom(16)
        self.metrics['decaps_time'] = time.time() - start_time
        return shared_secret


def demo_kem():
    """Demonstrate PQ KEM operation"""
    print("=== PQ KEM Demonstration ===\n")
    
    # Create KEM instance
    kem = PQCKyberKEM(security_level="kyber768")
    
    # Scenario: Alice generates keypair
    print("1. Alice generates Kyber keypair...")
    pk_alice, sk_alice = kem.generate_keypair()
    print(f"   Public key size: {len(pk_alice)} bytes")
    print(f"   Secret key size: {len(sk_alice)} bytes")
    print(f"   Keygen time: {kem.metrics['keygen_time']*1000:.3f} ms\n")
    
    # Scenario: Bob encapsulates a shared secret
    print("2. Bob encapsulates shared secret using Alice's public key...")
    ct, ss_bob = kem.encapsulate(pk_alice)
    print(f"   Ciphertext size: {len(ct)} bytes")
    print(f"   Shared secret size: {len(ss_bob)} bytes")
    print(f"   Encapsulation time: {kem.metrics['encaps_time']*1000:.3f} ms\n")
    
    # Scenario: Alice decapsulates to recover shared secret
    print("3. Alice decapsulates to recover shared secret...")
    ss_alice = kem.decapsulate(ct, sk_alice)
    print(f"   Decapsulation time: {kem.metrics['decaps_time']*1000:.3f} ms\n")
    
    # Verify shared secrets match
    if kem.kyber_available:
        match = ss_alice == ss_bob
        print(f"4. Shared secret match: {match}")
    else:
        print("4. Using simulated Kyber (shared secret consistency varies)")
    
    # Display overhead statistics
    print("\n=== Overhead Statistics ===")
    stats = kem.get_overhead_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    demo_kem()
