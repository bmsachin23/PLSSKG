"""
Configuration and Crypto-Agility Module
Manages ciphersuite registry, negotiation, and fallback mechanisms
"""

import json
from typing import Dict, List, Optional, Any
from enum import Enum


class CiphersuiteRegistry:
    """
    Registry of available cipher suites with capabilities and preferences
    """
    
    # Ciphersuite definitions
    SUITES = {
        "HYBRID_KYBER768_DILITHIUM3_SHA384": {
            "id": 0x01,
            "name": "HYBRID_KYBER768_DILITHIUM3_SHA384",
            "kem": "Kyber768",
            "signature": "Dilithium3",
            "kdf": "HKDF-SHA384",
            "skg": True,
            "pqkem": True,
            "security_level": "high",
            "preferred": True,
            "description": "Hybrid SKG+Kyber768 with Dilithium3 signatures"
        },
        "HYBRID_KYBER512_DILITHIUM2_SHA384": {
            "id": 0x02,
            "name": "HYBRID_KYBER512_DILITHIUM2_SHA384",
            "kem": "Kyber512",
            "signature": "Dilithium2",
            "kdf": "HKDF-SHA384",
            "skg": True,
            "pqkem": True,
            "security_level": "medium-high",
            "preferred": False,
            "description": "Hybrid SKG+Kyber512 with Dilithium2 (lower overhead)"
        },
        "PQ_ONLY_KYBER768_DILITHIUM3": {
            "id": 0x03,
            "name": "PQ_ONLY_KYBER768_DILITHIUM3",
            "kem": "Kyber768",
            "signature": "Dilithium3",
            "kdf": "HKDF-SHA384",
            "skg": False,
            "pqkem": True,
            "security_level": "medium",
            "preferred": False,
            "description": "PQ-only fallback (no SKG)"
        },
        "CLASSICAL_SKG_SHA384": {
            "id": 0x04,
            "name": "CLASSICAL_SKG_SHA384",
            "kem": None,
            "signature": "Ed25519",
            "kdf": "HKDF-SHA384",
            "skg": True,
            "pqkem": False,
            "security_level": "low",
            "preferred": False,
            "description": "Classical SKG-only (legacy)"
        },
        "HYBRID_FRODO640_SPHINCS_SHA512": {
            "id": 0x05,
            "name": "HYBRID_FRODO640_SPHINCS_SHA512",
            "kem": "FrodoKEM640",
            "signature": "SPHINCS+",
            "kdf": "HKDF-SHA512",
            "skg": True,
            "pqkem": True,
            "security_level": "high",
            "preferred": False,
            "description": "Conservative: FrodoKEM+SPHINCS+ (higher overhead)"
        }
    }
    
    @classmethod
    def get_suite(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get ciphersuite by name"""
        return cls.SUITES.get(name)
    
    @classmethod
    def get_suite_by_id(cls, suite_id: int) -> Optional[Dict[str, Any]]:
        """Get ciphersuite by ID"""
        for suite in cls.SUITES.values():
            if suite['id'] == suite_id:
                return suite
        return None
    
    @classmethod
    def get_preferred_suite(cls) -> Dict[str, Any]:
        """Get the preferred ciphersuite"""
        for suite in cls.SUITES.values():
            if suite.get('preferred', False):
                return suite
        return list(cls.SUITES.values())[0]
    
    @classmethod
    def list_suites(cls, 
                   require_skg: Optional[bool] = None,
                   require_pqkem: Optional[bool] = None,
                   min_security_level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List ciphersuites matching criteria
        
        Args:
            require_skg: Filter by SKG requirement
            require_pqkem: Filter by PQ KEM requirement
            min_security_level: Minimum security level
            
        Returns:
            List of matching ciphersuites
        """
        results = []
        security_order = ["low", "medium", "medium-high", "high"]
        
        for suite in cls.SUITES.values():
            # Apply filters
            if require_skg is not None and suite['skg'] != require_skg:
                continue
            if require_pqkem is not None and suite['pqkem'] != require_pqkem:
                continue
            if min_security_level is not None:
                suite_level_idx = security_order.index(suite['security_level'])
                min_level_idx = security_order.index(min_security_level)
                if suite_level_idx < min_level_idx:
                    continue
            
            results.append(suite)
        
        return results


class CiphersuiteNegotiator:
    """
    Handles ciphersuite negotiation between parties
    """
    
    def __init__(self, 
                 supported_suites: Optional[List[str]] = None,
                 preferences: Optional[List[str]] = None):
        """
        Initialize negotiator
        
        Args:
            supported_suites: List of supported ciphersuite names
            preferences: Ordered list of preferred ciphersuites
        """
        self.supported_suites = supported_suites or list(CiphersuiteRegistry.SUITES.keys())
        self.preferences = preferences or [
            "HYBRID_KYBER768_DILITHIUM3_SHA384",
            "HYBRID_KYBER512_DILITHIUM2_SHA384",
            "PQ_ONLY_KYBER768_DILITHIUM3",
            "CLASSICAL_SKG_SHA384"
        ]
    
    def negotiate(self, 
                 peer_supported: List[str],
                 peer_preferences: Optional[List[str]] = None) -> Optional[str]:
        """
        Negotiate ciphersuite with peer
        
        Args:
            peer_supported: Peer's supported ciphersuites
            peer_preferences: Peer's preferences (optional)
            
        Returns:
            Selected ciphersuite name, or None if no match
        """
        # Find intersection of supported suites
        common_suites = set(self.supported_suites) & set(peer_supported)
        
        if not common_suites:
            return None
        
        # Select based on preferences
        for suite_name in self.preferences:
            if suite_name in common_suites:
                return suite_name
        
        # Fallback: use peer's preference if provided
        if peer_preferences:
            for suite_name in peer_preferences:
                if suite_name in common_suites:
                    return suite_name
        
        # Last resort: any common suite
        return list(common_suites)[0]
    
    def create_client_hello(self) -> Dict[str, Any]:
        """
        Create ClientHello-style negotiation message
        
        Returns:
            Dictionary with supported suites and preferences
        """
        return {
            'supported_suites': self.supported_suites,
            'preferences': self.preferences,
            'capabilities': {
                'skg': True,
                'pqkem': True,
                'hybrid_sig': True
            }
        }
    
    def process_client_hello(self, client_hello: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process ClientHello and create ServerHello
        
        Args:
            client_hello: Client's hello message
            
        Returns:
            ServerHello with selected ciphersuite
        """
        selected_suite = self.negotiate(
            client_hello['supported_suites'],
            client_hello.get('preferences')
        )
        
        if selected_suite is None:
            return {
                'status': 'error',
                'message': 'No common ciphersuite'
            }
        
        return {
            'status': 'ok',
            'selected_suite': selected_suite,
            'suite_info': CiphersuiteRegistry.get_suite(selected_suite)
        }


class FallbackManager:
    """
    Manages graceful fallback when ciphersuites fail
    """
    
    def __init__(self):
        """Initialize fallback manager"""
        self.fallback_chain = [
            ("HYBRID_KYBER768_DILITHIUM3_SHA384", "high"),
            ("HYBRID_KYBER512_DILITHIUM2_SHA384", "medium-high"),
            ("PQ_ONLY_KYBER768_DILITHIUM3", "medium"),
            ("CLASSICAL_SKG_SHA384", "low")
        ]
        self.current_suite = None
        self.security_level = None
    
    def attempt_fallback(self,
                        failed_suite: str,
                        reason: str,
                        skg_entropy: Optional[float] = None,
                        pqkem_available: bool = True) -> Optional[tuple]:
        """
        Attempt to fall back to a working ciphersuite
        
        Args:
            failed_suite: Suite that failed
            reason: Failure reason
            skg_entropy: Available SKG entropy (if known)
            pqkem_available: Whether PQ KEM is available
            
        Returns:
            (fallback_suite, security_level, warning) tuple or None
        """
        print(f"\n[Fallback] Suite '{failed_suite}' failed: {reason}")
        
        # Find position in fallback chain
        failed_idx = -1
        for idx, (suite, level) in enumerate(self.fallback_chain):
            if suite == failed_suite:
                failed_idx = idx
                break
        
        # Try fallbacks
        for idx in range(failed_idx + 1, len(self.fallback_chain)):
            candidate_suite, candidate_level = self.fallback_chain[idx]
            suite_info = CiphersuiteRegistry.get_suite(candidate_suite)
            
            # Check if candidate is viable
            if suite_info['pqkem'] and not pqkem_available:
                continue  # Skip if PQ KEM required but unavailable
            
            if suite_info['skg'] and skg_entropy is not None and skg_entropy < 50:
                continue  # Skip if SKG required but entropy too low
            
            warning = f"Security downgraded to '{candidate_level}'"
            print(f"[Fallback] Falling back to '{candidate_suite}' (security: {candidate_level})")
            
            self.current_suite = candidate_suite
            self.security_level = candidate_level
            
            return (candidate_suite, candidate_level, warning)
        
        print("[Fallback] No viable fallback found")
        return None
    
    def log_fallback(self, log_file: str = "fallback_log.json"):
        """Log fallback events for analysis"""
        import json
        import time
        
        log_entry = {
            'timestamp': time.time(),
            'suite': self.current_suite,
            'security_level': self.security_level
        }
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except:
            pass


class ConfigManager:
    """
    Manages protocol configuration and policy
    """
    
    DEFAULT_CONFIG = {
        'protocol_version': '1.0',
        'default_ciphersuite': 'HYBRID_KYBER768_DILITHIUM3_SHA384',
        'min_security_level': 'medium',
        'allow_fallback': True,
        'skg_params': {
            'min_entropy_bits': 80,
            'quantization_method': 'var',
            'deviation': 0.5,
            'reconciliation_code': 'LDPC'
        },
        'pqkem_params': {
            'preferred_kem': 'Kyber768',
            'fallback_kem': 'Kyber512'
        },
        'signature_params': {
            'preferred_sig': 'Dilithium3',
            'fallback_sig': 'Dilithium2'
        },
        'kdf_params': {
            'hash_algorithm': 'SHA384',
            'key_length': 32,
            'rekeying_interval_sec': 3600
        },
        'timeouts': {
            'handshake_timeout_sec': 10,
            'signature_timeout_sec': 5
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager
        
        Args:
            config: Custom configuration (merges with defaults)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self._merge_config(config)
    
    def _merge_config(self, custom_config: Dict[str, Any]):
        """Recursively merge custom config with defaults"""
        for key, value in custom_config.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by path
        
        Args:
            key_path: Dot-separated path (e.g., 'skg_params.min_entropy_bits')
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Set configuration value by path
        
        Args:
            key_path: Dot-separated path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load(self, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            loaded_config = json.load(f)
            self._merge_config(loaded_config)
    
    def validate(self) -> List[str]:
        """
        Validate configuration
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check ciphersuite exists
        suite_name = self.config['default_ciphersuite']
        if not CiphersuiteRegistry.get_suite(suite_name):
            errors.append(f"Unknown ciphersuite: {suite_name}")
        
        # Check min entropy
        min_entropy = self.config['skg_params']['min_entropy_bits']
        if min_entropy < 80:
            errors.append(f"min_entropy_bits too low: {min_entropy} (should be ≥80)")
        
        # Check key length
        key_length = self.config['kdf_params']['key_length']
        if key_length < 16:
            errors.append(f"key_length too short: {key_length} (should be ≥16)")
        
        return errors


def demo_config_and_negotiation():
    """Demonstrate configuration and negotiation"""
    print("="*70)
    print("CONFIGURATION AND CRYPTO-AGILITY DEMONSTRATION")
    print("="*70)
    
    # 1. Ciphersuite registry
    print("\n### 1. Ciphersuite Registry ###\n")
    print("Available ciphersuites:")
    for suite in CiphersuiteRegistry.SUITES.values():
        print(f"  [{suite['id']:02x}] {suite['name']}")
        print(f"      Security: {suite['security_level']}, SKG: {suite['skg']}, PQ: {suite['pqkem']}")
    
    print("\nPreferred suite:")
    preferred = CiphersuiteRegistry.get_preferred_suite()
    print(f"  {preferred['name']} - {preferred['description']}")
    
    # 2. Negotiation
    print("\n### 2. Ciphersuite Negotiation ###\n")
    
    alice_negotiator = CiphersuiteNegotiator()
    bob_negotiator = CiphersuiteNegotiator(
        supported_suites=[
            "HYBRID_KYBER512_DILITHIUM2_SHA384",
            "PQ_ONLY_KYBER768_DILITHIUM3",
            "CLASSICAL_SKG_SHA384"
        ]
    )
    
    print("Alice supported:", alice_negotiator.supported_suites)
    print("Bob supported:", bob_negotiator.supported_suites)
    
    client_hello = alice_negotiator.create_client_hello()
    print("\nAlice sends ClientHello...")
    
    server_hello = bob_negotiator.process_client_hello(client_hello)
    print("Bob responds with ServerHello...")
    
    if server_hello['status'] == 'ok':
        print(f"\nNegotiated suite: {server_hello['selected_suite']}")
        print(f"Security level: {server_hello['suite_info']['security_level']}")
    else:
        print(f"\nNegotiation failed: {server_hello['message']}")
    
    # 3. Fallback mechanism
    print("\n### 3. Fallback Mechanism ###\n")
    
    fallback_mgr = FallbackManager()
    
    # Simulate failure
    result = fallback_mgr.attempt_fallback(
        failed_suite="HYBRID_KYBER768_DILITHIUM3_SHA384",
        reason="PQ KEM size exceeds MTU",
        skg_entropy=120.0,
        pqkem_available=True
    )
    
    if result:
        suite, level, warning = result
        print(f"\nFallback successful:")
        print(f"  Suite: {suite}")
        print(f"  Security: {level}")
        print(f"  Warning: {warning}")
    
    # 4. Configuration management
    print("\n### 4. Configuration Management ###\n")
    
    config_mgr = ConfigManager()
    
    print("Default configuration:")
    print(f"  Default suite: {config_mgr.get('default_ciphersuite')}")
    print(f"  Min entropy: {config_mgr.get('skg_params.min_entropy_bits')} bits")
    print(f"  Rekey interval: {config_mgr.get('kdf_params.rekeying_interval_sec')} sec")
    
    # Validate
    errors = config_mgr.validate()
    print(f"\nValidation: {'PASS' if not errors else 'FAIL'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    demo_config_and_negotiation()
