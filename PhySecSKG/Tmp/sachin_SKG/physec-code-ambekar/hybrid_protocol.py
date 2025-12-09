"""
Hybrid SKG+PQC Protocol Orchestrator
Coordinates the complete protocol flow: SKG → PQ KEM → Joint KDF → Authentication
Implements the post-quantum augmented physical-layer key generation system
"""

import os
import time
import hashlib
from typing import Tuple, Optional, Dict, Any, List
from enum import Enum

# Import our PQC modules
from pqc_kem import PQCKyberKEM
from hybrid_kdf import HybridKDF
from pqc_signatures import PQCDilithiumSignature, TranscriptBuilder


class CiphersuiteMode(Enum):
    """Ciphersuite options for crypto-agility"""
    CLASSICAL_ONLY = "classical"  # SKG only (legacy)
    HYBRID = "hybrid"  # SKG + PQ KEM (recommended)
    PQ_ONLY = "pq_only"  # PQ KEM only (fallback if SKG fails)


class SecurityLevel(Enum):
    """Security level markers"""
    HIGH = "high"  # Hybrid with good SKG entropy
    MEDIUM = "medium"  # Hybrid with reduced SKG or PQ-only
    LOW = "low"  # Classical only or degraded


class HybridSKGProtocol:
    """
    Main protocol orchestrator for hybrid SKG+PQC system
    
    Protocol Flow:
    1. SKG: Channel probing → quantization → reconciliation → SKG candidate key
    2. PQ KEM: Encapsulation/decapsulation → ss_pqkem
    3. Joint KDF: HKDF(ss_skg ∥ ss_pqkem) → session key
    4. Authentication: PQ signature over full transcript
    """
    
    def __init__(self, 
                 role: str,
                 ciphersuite: CiphersuiteMode = CiphersuiteMode.HYBRID,
                 use_hybrid_sig: bool = False):
        """
        Initialize hybrid protocol
        
        Args:
            role: 'alice' or 'bob' (or 'enb'/'ue' in LTE context)
            ciphersuite: Operating mode (classical/hybrid/pq_only)
            use_hybrid_sig: Whether to use hybrid classical+PQ signatures
        """
        self.role = role
        self.ciphersuite = ciphersuite
        self.use_hybrid_sig = use_hybrid_sig
        
        # Initialize PQC components
        self.kem = PQCKyberKEM(security_level="kyber768")
        self.kdf = HybridKDF(hash_algorithm="SHA384")
        self.signature = PQCDilithiumSignature(security_level="dilithium3")
        
        # Protocol state
        self.public_key = None
        self.secret_key = None
        self.sig_public_key = None
        self.sig_secret_key = None
        self.session_key = None
        self.security_level = SecurityLevel.HIGH
        
        # Metrics
        self.metrics = {
            'total_time': 0.0,
            'skg_time': 0.0,
            'pqkem_time': 0.0,
            'kdf_time': 0.0,
            'sig_time': 0.0,
            'total_overhead_bytes': 0,
            'entropy_bits': 0
        }
        
    def initialize(self):
        """
        Initialize cryptographic keys for this session
        Called once at setup
        """
        print(f"[{self.role.upper()}] Initializing hybrid SKG+PQC protocol...")
        print(f"  Ciphersuite: {self.ciphersuite.value}")
        
        if self.ciphersuite in [CiphersuiteMode.HYBRID, CiphersuiteMode.PQ_ONLY]:
            # Generate KEM keypair
            print("  Generating PQ KEM keypair...")
            self.public_key, self.secret_key = self.kem.generate_keypair()
            self.metrics['total_overhead_bytes'] += len(self.public_key)
            
        # Generate signature keypair
        print("  Generating signature keypair...")
        self.sig_public_key, self.sig_secret_key = self.signature.generate_keypair()
        self.metrics['total_overhead_bytes'] += len(self.sig_public_key)
        
        print("  Initialization complete.\n")
        
    def execute_skg_phase(self,
                         profile_list: List[float],
                         quantizer_params: Optional[Dict[str, Any]] = None) -> Tuple[bytes, Dict[str, Any]]:
        """
        Execute SKG phase: quantization of channel measurements
        
        Args:
            profile_list: Channel profile measurements (CSI/RSSI)
            quantizer_params: Parameters for quantization (deviation, method, etc.)
            
        Returns:
            (skg_secret, skg_metrics) tuple
        """
        start_time = time.time()
        print(f"[{self.role.upper()}] Phase 1: SKG Key Generation")
        print(f"  Channel measurements: {len(profile_list)} samples")
        
        # Import quantizer (using existing implementation)
        from quantizer import Quantizer
        
        if quantizer_params is None:
            quantizer_params = {'deviation': 0.5, 'method': 'var'}
        
        # Quantize channel profile
        quantizer = Quantizer(profile_list, quantizer_params.get('deviation', 0.5))
        
        # Use specified quantization method
        method = quantizer_params.get('method', 'var')
        if method == 'var':
            prelim_key = quantizer.quantize_var()
        elif method == 'stddev':
            prelim_key = quantizer.quantize_stddev()
        elif method == 'median':
            prelim_key = quantizer.quantize_median()
        else:
            prelim_key = quantizer.quantize_mean()
        
        # Convert to bytes
        skg_secret = bytes(prelim_key)
        
        skg_time = time.time() - start_time
        self.metrics['skg_time'] = skg_time
        
        skg_metrics = {
            'bits_generated': len(prelim_key),
            'time_ms': skg_time * 1000,
            'method': method,
            'samples_used': len(profile_list)
        }
        
        print(f"  Generated {len(prelim_key)} SKG bits")
        print(f"  Time: {skg_time*1000:.3f} ms")
        
        return skg_secret, skg_metrics
    
    def execute_pqkem_phase(self,
                           peer_public_key: Optional[bytes] = None,
                           is_initiator: bool = True) -> Tuple[Optional[bytes], bytes, Dict[str, Any]]:
        """
        Execute PQ KEM phase: encapsulation or decapsulation
        
        Args:
            peer_public_key: Peer's public key (for initiator)
            is_initiator: True if initiating (encapsulating), False if responding (decapsulating)
            
        Returns:
            (ciphertext, shared_secret, pqkem_metrics) tuple
        """
        start_time = time.time()
        print(f"\n[{self.role.upper()}] Phase 2: PQ KEM Exchange")
        
        if self.ciphersuite == CiphersuiteMode.CLASSICAL_ONLY:
            print("  Skipped (classical-only mode)")
            return None, b"", {'time_ms': 0}
        
        ciphertext = None
        shared_secret = None
        
        if is_initiator:
            print("  Role: Initiator (encapsulating)")
            ciphertext, shared_secret = self.kem.encapsulate(peer_public_key)
            self.metrics['total_overhead_bytes'] += len(ciphertext)
            print(f"  Ciphertext: {len(ciphertext)} bytes")
        else:
            print("  Role: Responder (decapsulating)")
            # Peer will provide ciphertext, we decrypt with our secret key
            # (This would be provided in real protocol)
            pass
        
        print(f"  Shared secret: {len(shared_secret) if shared_secret else 0} bytes")
        
        pqkem_time = time.time() - start_time
        self.metrics['pqkem_time'] = pqkem_time
        
        pqkem_metrics = {
            'time_ms': pqkem_time * 1000,
            'ciphertext_bytes': len(ciphertext) if ciphertext else 0,
            'shared_secret_bits': len(shared_secret) * 8 if shared_secret else 0
        }
        
        print(f"  Time: {pqkem_time*1000:.3f} ms")
        
        return ciphertext, shared_secret, pqkem_metrics
    
    def decapsulate_pqkem(self, ciphertext: bytes) -> bytes:
        """
        Decapsulate received PQ KEM ciphertext
        
        Args:
            ciphertext: Received ciphertext from peer
            
        Returns:
            Shared secret
        """
        print(f"\n[{self.role.upper()}] Phase 2b: PQ KEM Decapsulation")
        shared_secret = self.kem.decapsulate(ciphertext, self.secret_key)
        print(f"  Decapsulated shared secret: {len(shared_secret)} bytes")
        return shared_secret
    
    def execute_kdf_phase(self,
                         skg_secret: bytes,
                         pqkem_secret: bytes,
                         context: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Execute joint KDF phase: derive session key from both secrets
        
        Args:
            skg_secret: SKG shared secret
            pqkem_secret: PQ KEM shared secret
            context: Protocol context (IDs, nonces, timestamps, etc.)
            
        Returns:
            (session_key, kdf_metrics) tuple
        """
        start_time = time.time()
        print(f"\n[{self.role.upper()}] Phase 3: Joint Key Derivation")
        
        # Derive session key using hybrid KDF
        if self.ciphersuite == CiphersuiteMode.CLASSICAL_ONLY:
            print("  Mode: Classical (SKG only)")
            # Use only SKG secret with KDF for privacy amplification
            session_key = self.kdf.derive_session_key(
                ss_skg=skg_secret,
                ss_pqkem=b"",  # Empty PQ component
                context=context,
                key_length=32
            )
        elif self.ciphersuite == CiphersuiteMode.PQ_ONLY:
            print("  Mode: PQ only (SKG fallback)")
            session_key = self.kdf.derive_session_key(
                ss_skg=b"",  # Empty SKG component
                ss_pqkem=pqkem_secret,
                context=context,
                key_length=32
            )
            self.security_level = SecurityLevel.MEDIUM
        else:  # HYBRID
            print("  Mode: Hybrid (SKG + PQ KEM)")
            session_key = self.kdf.derive_session_key(
                ss_skg=skg_secret,
                ss_pqkem=pqkem_secret,
                context=context,
                key_length=32
            )
        
        self.session_key = session_key
        
        kdf_time = time.time() - start_time
        self.metrics['kdf_time'] = kdf_time
        
        # Calculate entropy estimate
        skg_bits = len(skg_secret) * 8
        pqkem_bits = len(pqkem_secret) * 8
        
        entropy_estimate = self.kdf.get_entropy_estimate(
            skg_bits=skg_bits,
            skg_min_entropy_per_bit=0.85,  # Typical after reconciliation
            reconciliation_leakage_bits=int(skg_bits * 0.15),  # Typical leakage
            pqkem_bits=pqkem_bits
        )
        
        self.metrics['entropy_bits'] = entropy_estimate['total_entropy_bits']
        
        kdf_metrics = {
            'time_ms': kdf_time * 1000,
            'session_key_bits': len(session_key) * 8,
            'entropy_estimate': entropy_estimate
        }
        
        print(f"  Session key: {len(session_key)} bytes ({len(session_key)*8} bits)")
        print(f"  Estimated entropy: {entropy_estimate['total_entropy_bits']:.1f} bits")
        print(f"  Time: {kdf_time*1000:.3f} ms")
        
        return session_key, kdf_metrics
    
    def execute_authentication_phase(self,
                                    transcript_data: Dict[str, bytes],
                                    is_signer: bool = True) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """
        Execute authentication phase: sign or verify transcript
        
        Args:
            transcript_data: Dictionary with all transcript elements
            is_signer: True if signing, False if verifying
            
        Returns:
            (signature, auth_metrics) tuple
        """
        start_time = time.time()
        print(f"\n[{self.role.upper()}] Phase 4: Transcript Authentication")
        
        # Build transcript
        transcript_builder = TranscriptBuilder()
        transcript_builder.add_element("protocol", b"SKG+PQC-Hybrid-v1.0")
        transcript_builder.add_element("ciphersuite", self.ciphersuite.value.encode())
        transcript_builder.add_element("role", self.role.encode())
        
        # Add all transcript data
        for key, value in transcript_data.items():
            if value is not None:
                transcript_builder.add_element(key, value if isinstance(value, bytes) else str(value).encode())
        
        transcript = transcript_builder.finalize()
        
        signature = None
        if is_signer:
            print("  Signing transcript...")
            signature = self.signature.sign_transcript(transcript, self.sig_secret_key)
            self.metrics['total_overhead_bytes'] += len(signature)
            print(f"  Signature: {len(signature)} bytes")
        else:
            print("  Verifying transcript...")
            # Verification would be done with peer's signature and public key
        
        auth_time = time.time() - start_time
        self.metrics['sig_time'] = auth_time
        
        auth_metrics = {
            'time_ms': auth_time * 1000,
            'transcript_bytes': len(transcript),
            'signature_bytes': len(signature) if signature else 0
        }
        
        print(f"  Time: {auth_time*1000:.3f} ms")
        
        return signature, auth_metrics
    
    def verify_peer_signature(self,
                             transcript_data: Dict[str, bytes],
                             signature: bytes,
                             peer_sig_public_key: bytes) -> bool:
        """
        Verify peer's signature on transcript
        
        Args:
            transcript_data: Transcript elements
            signature: Peer's signature
            peer_sig_public_key: Peer's signature public key
            
        Returns:
            True if valid, False otherwise
        """
        print(f"\n[{self.role.upper()}] Verifying peer signature...")
        
        # Rebuild transcript (must match peer's construction)
        transcript_builder = TranscriptBuilder()
        transcript_builder.add_element("protocol", b"SKG+PQC-Hybrid-v1.0")
        transcript_builder.add_element("ciphersuite", self.ciphersuite.value.encode())
        
        for key, value in transcript_data.items():
            if value is not None:
                transcript_builder.add_element(key, value if isinstance(value, bytes) else str(value).encode())
        
        transcript = transcript_builder.finalize()
        
        valid = self.signature.verify_signature(transcript, signature, peer_sig_public_key)
        print(f"  Verification result: {'VALID' if valid else 'INVALID'}")
        
        return valid
    
    def get_protocol_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive protocol metrics for thesis evaluation
        
        Returns:
            Dictionary with all performance and overhead metrics
        """
        self.metrics['total_time'] = (self.metrics['skg_time'] + 
                                     self.metrics['pqkem_time'] + 
                                     self.metrics['kdf_time'] + 
                                     self.metrics['sig_time'])
        
        return {
            'timing': {
                'total_ms': self.metrics['total_time'] * 1000,
                'skg_ms': self.metrics['skg_time'] * 1000,
                'pqkem_ms': self.metrics['pqkem_time'] * 1000,
                'kdf_ms': self.metrics['kdf_time'] * 1000,
                'signature_ms': self.metrics['sig_time'] * 1000
            },
            'overhead': {
                'total_bytes': self.metrics['total_overhead_bytes'],
                'kem_pk_bytes': len(self.public_key) if self.public_key else 0,
                'sig_pk_bytes': len(self.sig_public_key) if self.sig_public_key else 0
            },
            'security': {
                'ciphersuite': self.ciphersuite.value,
                'security_level': self.security_level.value,
                'entropy_bits': self.metrics['entropy_bits']
            },
            'components': {
                'kem': self.kem.get_metrics(),
                'kdf': self.kdf.get_metrics(),
                'signature': self.signature.get_metrics()
            }
        }
    
    def adaptive_ciphersuite_selection(self,
                                      skg_entropy: float,
                                      skg_threshold: float = 100.0,
                                      pqkem_available: bool = True) -> CiphersuiteMode:
        """
        Adaptively select ciphersuite based on conditions
        
        Args:
            skg_entropy: Estimated SKG entropy in bits
            skg_threshold: Minimum entropy threshold for hybrid mode
            pqkem_available: Whether PQ KEM is available
            
        Returns:
            Selected ciphersuite mode
        """
        print("\n[Protocol] Adaptive Ciphersuite Selection")
        print(f"  SKG entropy: {skg_entropy:.1f} bits (threshold: {skg_threshold:.1f})")
        print(f"  PQ KEM available: {pqkem_available}")
        
        if skg_entropy >= skg_threshold and pqkem_available:
            selected = CiphersuiteMode.HYBRID
            self.security_level = SecurityLevel.HIGH
            print(f"  Selected: HYBRID (high security)")
        elif pqkem_available:
            selected = CiphersuiteMode.PQ_ONLY
            self.security_level = SecurityLevel.MEDIUM
            print(f"  Selected: PQ_ONLY (medium security, low SKG entropy)")
        else:
            selected = CiphersuiteMode.CLASSICAL_ONLY
            self.security_level = SecurityLevel.LOW
            print(f"  Selected: CLASSICAL_ONLY (low security, PQ unavailable)")
        
        self.ciphersuite = selected
        return selected


def demo_full_protocol():
    """
    Demonstrate complete hybrid SKG+PQC protocol execution
    """
    print("="*70)
    print("HYBRID SKG+PQC PROTOCOL DEMONSTRATION")
    print("Complete Flow: SKG → PQ KEM → Joint KDF → Authentication")
    print("="*70)
    
    # Setup: Two parties (Alice as eNodeB, Bob as UE)
    print("\n### SETUP PHASE ###\n")
    
    alice = HybridSKGProtocol(role="alice", ciphersuite=CiphersuiteMode.HYBRID)
    bob = HybridSKGProtocol(role="bob", ciphersuite=CiphersuiteMode.HYBRID)
    
    alice.initialize()
    bob.initialize()
    
    # Simulate channel measurements (same channel, different noise)
    print("\n### PROTOCOL EXECUTION ###\n")
    
    # Phase 1: SKG
    import numpy as np
    base_channel = np.random.randn(100) * 10 + 50
    alice_measurements = (base_channel + np.random.randn(100) * 0.5).tolist()
    bob_measurements = (base_channel + np.random.randn(100) * 0.5).tolist()
    
    alice_skg_secret, alice_skg_metrics = alice.execute_skg_phase(
        alice_measurements,
        {'deviation': 0.5, 'method': 'var'}
    )
    
    bob_skg_secret, bob_skg_metrics = bob.execute_skg_phase(
        bob_measurements,
        {'deviation': 0.5, 'method': 'var'}
    )
    
    # Phase 2: PQ KEM
    # Alice initiates, Bob responds
    pqkem_ct, alice_pqkem_secret, alice_pqkem_metrics = alice.execute_pqkem_phase(
        peer_public_key=bob.public_key,
        is_initiator=True
    )
    
    bob_pqkem_secret = bob.decapsulate_pqkem(pqkem_ct)
    
    # Phase 3: Joint KDF
    context = {
        'role': 'alice',
        'timestamp': int(time.time()),
        'cell_id': 'CELL_001',
        'sector_id': 'SECTOR_A',
        'snr_bin': 15,
        'nonce': os.urandom(16).hex()
    }
    
    alice_session_key, alice_kdf_metrics = alice.execute_kdf_phase(
        alice_skg_secret,
        alice_pqkem_secret,
        context
    )
    
    bob_context = context.copy()
    bob_context['role'] = 'bob'
    
    bob_session_key, bob_kdf_metrics = bob.execute_kdf_phase(
        bob_skg_secret,
        bob_pqkem_secret,
        bob_context
    )
    
    # Phase 4: Authentication
    transcript_data = {
        'channel_commitment': alice.kdf.compute_channel_commitment(
            csi_data=bytes(alice_skg_secret),
            pilot_seq=b"PILOT_001",
            nonce=context['nonce'].encode()
        ),
        'pqkem_ciphertext': pqkem_ct,
        'context': str(context).encode()
    }
    
    alice_signature, alice_auth_metrics = alice.execute_authentication_phase(
        transcript_data,
        is_signer=True
    )
    
    # Verification
    bob_verifies_alice = bob.verify_peer_signature(
        transcript_data,
        alice_signature,
        alice.sig_public_key
    )
    
    # Results Summary
    print("\n")
    print("="*70)
    print("PROTOCOL EXECUTION SUMMARY")
    print("="*70)
    
    print("\n### Key Agreement ###")
    print(f"Alice session key: {alice_session_key.hex()[:32]}...")
    print(f"Bob session key:   {bob_session_key.hex()[:32]}...")
    
    # Note: Keys will differ slightly due to SKG disagreement rate
    # In practice, error correction/reconciliation ensures agreement
    
    print("\n### Performance Metrics (Alice) ###")
    alice_metrics = alice.get_protocol_metrics()
    print(f"Total time: {alice_metrics['timing']['total_ms']:.3f} ms")
    print(f"  - SKG: {alice_metrics['timing']['skg_ms']:.3f} ms")
    print(f"  - PQ KEM: {alice_metrics['timing']['pqkem_ms']:.3f} ms")
    print(f"  - KDF: {alice_metrics['timing']['kdf_ms']:.3f} ms")
    print(f"  - Signature: {alice_metrics['timing']['signature_ms']:.3f} ms")
    
    print(f"\nTotal overhead: {alice_metrics['overhead']['total_bytes']} bytes")
    print(f"Security level: {alice_metrics['security']['security_level']}")
    print(f"Entropy: {alice_metrics['security']['entropy_bits']:.1f} bits")
    
    print(f"\nAuthentication: {'SUCCESS' if bob_verifies_alice else 'FAILED'}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    demo_full_protocol()
