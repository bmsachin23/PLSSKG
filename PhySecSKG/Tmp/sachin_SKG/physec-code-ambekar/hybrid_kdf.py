"""
Hybrid Key Derivation Function (KDF) Module
Combines SKG shared secret with PQ KEM shared secret for quantum-resilient keys
Uses HKDF-SHA-384/512 with context binding
"""

import hashlib
import hmac
import time
from typing import Optional, Dict, Any
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


class HybridKDF:
    """
    Hybrid Key Derivation Function combining SKG and PQ KEM secrets
    
    Implements: k_session = HKDF(salt, info, ss_skg ∥ ss_pqkem)
    Using HKDF-SHA-384 or HKDF-SHA-512 for quantum-resistant hashing
    """
    
    def __init__(self, hash_algorithm: str = "SHA384"):
        """
        Initialize hybrid KDF
        
        Args:
            hash_algorithm: 'SHA384' or 'SHA512' (quantum-resistant)
        """
        self.hash_algorithm = hash_algorithm
        if hash_algorithm == "SHA384":
            self.hash_algo = hashes.SHA384()
        elif hash_algorithm == "SHA512":
            self.hash_algo = hashes.SHA512()
        else:
            raise ValueError("Use SHA384 or SHA512 for quantum resistance")
            
        self.metrics = {
            'derive_time': 0.0,
            'input_entropy_bits': 0,
            'output_key_bits': 0
        }
    
    def derive_session_key(self,
                          ss_skg: bytes,
                          ss_pqkem: bytes,
                          context: Optional[Dict[str, Any]] = None,
                          salt: Optional[bytes] = None,
                          key_length: int = 32) -> bytes:
        """
        Derive session key from SKG and PQ KEM shared secrets
        
        Args:
            ss_skg: SKG shared secret (from channel measurements)
            ss_pqkem: PQ KEM shared secret (from Kyber/FrodoKEM)
            context: Dictionary with context info (IDs, nonces, time, SNR, etc.)
            salt: Optional salt for HKDF-Extract (default: hash of transcript)
            key_length: Desired key length in bytes (default: 32 for AES-256)
            
        Returns:
            Derived session key
        """
        start_time = time.time()
        
        # Concatenate both shared secrets
        combined_secret = ss_skg + ss_pqkem
        
        # Prepare context/info string for HKDF
        info = self._build_info_string(context)
        
        # Use provided salt or derive from context
        if salt is None:
            salt = self._derive_salt_from_context(context, info)
        
        # HKDF-Extract and Expand
        kdf = HKDF(
            algorithm=self.hash_algo,
            length=key_length,
            salt=salt,
            info=info
        )
        
        session_key = kdf.derive(combined_secret)
        
        self.metrics['derive_time'] = time.time() - start_time
        self.metrics['input_entropy_bits'] = len(combined_secret) * 8
        self.metrics['output_key_bits'] = len(session_key) * 8
        
        return session_key
    
    def _build_info_string(self, context: Optional[Dict[str, Any]]) -> bytes:
        """
        Build info/context string for HKDF binding
        
        Includes: role, timestamp, cell/sector ID, SNR bin, nonces, pilot config
        """
        if context is None:
            context = {}
        
        info_parts = []
        
        # Protocol identifier
        info_parts.append(b"SKG+PQC-Hybrid-v1.0")
        
        # Add context fields in canonical order
        if 'role' in context:
            info_parts.append(f"role={context['role']}".encode())
        if 'timestamp' in context:
            info_parts.append(f"time={context['timestamp']}".encode())
        if 'cell_id' in context:
            info_parts.append(f"cell={context['cell_id']}".encode())
        if 'sector_id' in context:
            info_parts.append(f"sector={context['sector_id']}".encode())
        if 'snr_bin' in context:
            info_parts.append(f"snr={context['snr_bin']}".encode())
        if 'nonce' in context:
            info_parts.append(f"nonce={context['nonce']}".encode())
        if 'frame_index' in context:
            info_parts.append(f"frame={context['frame_index']}".encode())
        if 'slot_index' in context:
            info_parts.append(f"slot={context['slot_index']}".encode())
        if 'pilot_seq' in context:
            info_parts.append(f"pilot={context['pilot_seq']}".encode())
            
        # Join with separator
        return b"|".join(info_parts)
    
    def _derive_salt_from_context(self, context: Optional[Dict[str, Any]], info: bytes) -> bytes:
        """
        Derive salt from transcript/context for HKDF-Extract
        
        Uses hash of context including CSI/RSSI measurements if available
        """
        if context is None:
            context = {}
        
        # Build transcript for hashing
        transcript_parts = [info]
        
        # Include channel measurements in transcript
        if 'csi_hash' in context:
            transcript_parts.append(context['csi_hash'])
        if 'rssi_measurements' in context:
            transcript_parts.append(str(context['rssi_measurements']).encode())
        if 'pilot_sequence' in context:
            transcript_parts.append(context['pilot_sequence'])
            
        transcript = b"||".join(transcript_parts)
        
        # Hash transcript to create salt
        if self.hash_algorithm == "SHA384":
            salt = hashlib.sha384(transcript).digest()
        else:
            salt = hashlib.sha512(transcript).digest()
            
        return salt
    
    def compute_channel_commitment(self, 
                                   csi_data: Optional[bytes] = None,
                                   pilot_seq: Optional[bytes] = None,
                                   nonce: Optional[bytes] = None) -> bytes:
        """
        Compute H(CSI||pilot-seq||nonce) for channel binding
        
        This commitment is included in the signed transcript to prevent
        transcript forgery and bind PQ messages to measured channel state
        
        Args:
            csi_data: Channel State Information measurements
            pilot_seq: Pilot sequence used for measurement
            nonce: Fresh nonce for this session
            
        Returns:
            Hash commitment binding PQ exchange to channel
        """
        commitment_input = b""
        
        if csi_data is not None:
            commitment_input += csi_data
        if pilot_seq is not None:
            commitment_input += pilot_seq
        if nonce is not None:
            commitment_input += nonce
        
        if self.hash_algorithm == "SHA384":
            return hashlib.sha384(commitment_input).digest()
        else:
            return hashlib.sha512(commitment_input).digest()
    
    def ratchet_key(self, 
                    current_key: bytes,
                    new_skg_secret: Optional[bytes] = None,
                    new_pqkem_secret: Optional[bytes] = None,
                    ratchet_context: Optional[Dict[str, Any]] = None) -> bytes:
        """
        Perform key ratcheting for forward secrecy
        
        Useful for post-compromise security and frequent re-keying
        synchronized with channel coherence or handover events
        
        Args:
            current_key: Current session key
            new_skg_secret: Optional new SKG measurement
            new_pqkem_secret: Optional new PQ KEM encapsulation
            ratchet_context: Context for this ratchet step
            
        Returns:
            New ratcheted key
        """
        ratchet_input = current_key
        
        if new_skg_secret is not None:
            ratchet_input += new_skg_secret
        if new_pqkem_secret is not None:
            ratchet_input += new_pqkem_secret
            
        # Add ratchet counter/context
        info = self._build_info_string(ratchet_context) if ratchet_context else b"ratchet"
        
        kdf = HKDF(
            algorithm=self.hash_algo,
            length=32,
            salt=b"ratchet_salt",
            info=info
        )
        
        return kdf.derive(ratchet_input)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get KDF performance metrics"""
        return self.metrics.copy()
    
    def get_entropy_estimate(self, 
                            skg_bits: int,
                            skg_min_entropy_per_bit: float,
                            reconciliation_leakage_bits: int,
                            pqkem_bits: int) -> Dict[str, float]:
        """
        Estimate total entropy in derived key
        
        Args:
            skg_bits: Number of SKG bits
            skg_min_entropy_per_bit: Min-entropy per SKG bit (e.g., 0.9)
            reconciliation_leakage_bits: Information leaked during reconciliation
            pqkem_bits: PQ KEM shared secret bits (typically 256)
            
        Returns:
            Dictionary with entropy analysis
        """
        skg_gross_entropy = skg_bits * skg_min_entropy_per_bit
        skg_net_entropy = max(0, skg_gross_entropy - reconciliation_leakage_bits)
        pqkem_entropy = pqkem_bits  # Assuming full entropy from KEM
        total_entropy = skg_net_entropy + pqkem_entropy
        
        return {
            'skg_gross_entropy_bits': skg_gross_entropy,
            'skg_net_entropy_bits': skg_net_entropy,
            'reconciliation_leakage_bits': reconciliation_leakage_bits,
            'pqkem_entropy_bits': pqkem_entropy,
            'total_entropy_bits': total_entropy,
            'output_key_bits': self.metrics['output_key_bits'],
            'entropy_margin_bits': total_entropy - self.metrics['output_key_bits']
        }


def demo_hybrid_kdf():
    """Demonstrate hybrid KDF operation"""
    print("=== Hybrid KDF Demonstration ===\n")
    
    # Create hybrid KDF instance
    kdf = HybridKDF(hash_algorithm="SHA384")
    
    # Simulate SKG shared secret (from quantized channel measurements)
    print("1. Simulating SKG shared secret...")
    skg_key_bits = [1, 0, 1, 1, 0, 0, 1, 1] * 16  # 128 bits
    ss_skg = bytes(skg_key_bits)
    print(f"   SKG secret size: {len(ss_skg)} bytes ({len(ss_skg)*8} bits)")
    
    # Simulate PQ KEM shared secret (from Kyber)
    print("2. Simulating PQ KEM shared secret...")
    import os
    ss_pqkem = os.urandom(32)  # 256 bits from Kyber
    print(f"   PQ KEM secret size: {len(ss_pqkem)} bytes ({len(ss_pqkem)*8} bits)")
    
    # Prepare context for key derivation
    print("3. Preparing context information...")
    import time as time_module
    context = {
        'role': 'eNodeB',
        'timestamp': int(time_module.time()),
        'cell_id': 'CELL_001',
        'sector_id': 'SECTOR_A',
        'snr_bin': 15,
        'nonce': os.urandom(16).hex(),
        'frame_index': 1234,
        'slot_index': 56
    }
    print(f"   Context: {context}")
    
    # Compute channel commitment
    print("\n4. Computing channel commitment...")
    csi_data = os.urandom(64)  # Simulated CSI measurements
    pilot_seq = b"PILOT_SEQ_001"
    nonce = os.urandom(16)
    commitment = kdf.compute_channel_commitment(csi_data, pilot_seq, nonce)
    print(f"   Commitment: {commitment.hex()[:32]}...")
    
    # Derive session key
    print("\n5. Deriving session key using HKDF...")
    session_key = kdf.derive_session_key(
        ss_skg=ss_skg,
        ss_pqkem=ss_pqkem,
        context=context,
        key_length=32  # 256-bit key for AES-256-GCM
    )
    print(f"   Session key: {session_key.hex()[:32]}...")
    print(f"   Key length: {len(session_key)} bytes ({len(session_key)*8} bits)")
    print(f"   Derivation time: {kdf.metrics['derive_time']*1000:.3f} ms")
    
    # Entropy analysis
    print("\n6. Entropy analysis...")
    entropy_stats = kdf.get_entropy_estimate(
        skg_bits=128,
        skg_min_entropy_per_bit=0.85,
        reconciliation_leakage_bits=20,
        pqkem_bits=256
    )
    print("   Entropy estimates:")
    for key, value in entropy_stats.items():
        print(f"     {key}: {value:.2f}")
    
    # Demonstrate key ratcheting
    print("\n7. Demonstrating key ratcheting for forward secrecy...")
    new_skg = os.urandom(16)
    ratcheted_key = kdf.ratchet_key(
        current_key=session_key,
        new_skg_secret=new_skg,
        ratchet_context={'ratchet_epoch': 2}
    )
    print(f"   Ratcheted key: {ratcheted_key.hex()[:32]}...")


if __name__ == "__main__":
    demo_hybrid_kdf()
