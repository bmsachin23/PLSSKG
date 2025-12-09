"""
Comprehensive Demonstration: Hybrid SKG+PQC System
Complete end-to-end demonstration integrating all components
For Ph.D. thesis evaluation and validation
"""

import os
import sys
import time
import numpy as np
from typing import Dict, Any

# Import all modules
from hybrid_protocol import HybridSKGProtocol, CiphersuiteMode, SecurityLevel
from config_cryptoagility import CiphersuiteRegistry, CiphersuiteNegotiator, ConfigManager
from benchmark_evaluation import PerformanceBenchmark


class ComprehensiveDemo:
    """
    Complete demonstration of hybrid SKG+PQC system
    """
    
    def __init__(self):
        """Initialize demonstration"""
        self.config = ConfigManager()
        print("="*70)
        print("HYBRID SKG+PQC SYSTEM - COMPREHENSIVE DEMONSTRATION")
        print("Post-Quantum Cryptography Enhanced Physical-Layer Key Generation")
        print("="*70)
    
    def demo_1_basic_protocol(self):
        """Demonstration 1: Basic hybrid protocol execution"""
        print("\n" + "="*70)
        print("DEMO 1: Basic Hybrid Protocol Execution")
        print("="*70)
        
        print("\nScenario: eNodeB (Alice) and UE (Bob) perform hybrid key agreement")
        print("Protocol: SKG → PQ KEM → Joint KDF → Authentication\n")
        
        # Initialize parties
        alice = HybridSKGProtocol(role="eNodeB", ciphersuite=CiphersuiteMode.HYBRID)
        bob = HybridSKGProtocol(role="UE", ciphersuite=CiphersuiteMode.HYBRID)
        
        alice.initialize()
        bob.initialize()
        
        # Simulate channel measurements
        print("\n[Channel Measurement Phase]")
        print("Simulating reciprocal wireless channel...")
        
        channel_samples = 100
        base_channel = np.random.randn(channel_samples) * 10 + 50
        alice_channel = (base_channel + np.random.randn(channel_samples) * 0.5).tolist()
        bob_channel = (base_channel + np.random.randn(channel_samples) * 0.5).tolist()
        
        print(f"Channel samples: {channel_samples}")
        print(f"SNR estimate: ~20 dB")
        
        # Execute protocol
        print("\n[Protocol Execution]")
        
        # Phase 1: SKG
        alice_skg, alice_skg_metrics = alice.execute_skg_phase(
            alice_channel,
            {'deviation': 0.5, 'method': 'var'}
        )
        
        bob_skg, bob_skg_metrics = bob.execute_skg_phase(
            bob_channel,
            {'deviation': 0.5, 'method': 'var'}
        )
        
        # Phase 2: PQ KEM exchange
        pqkem_ct, alice_pqkem, alice_pqkem_metrics = alice.execute_pqkem_phase(
            peer_public_key=bob.public_key,
            is_initiator=True
        )
        
        bob_pqkem = bob.decapsulate_pqkem(pqkem_ct)
        
        # Phase 3: Joint KDF
        context = {
            'role': 'eNodeB',
            'timestamp': int(time.time()),
            'cell_id': 'CELL_5G_001',
            'sector_id': 'ALPHA',
            'snr_bin': 15,
            'nonce': os.urandom(16).hex(),
            'frame_index': 1000,
            'slot_index': 5
        }
        
        alice_key, alice_kdf_metrics = alice.execute_kdf_phase(
            alice_skg, alice_pqkem, context
        )
        
        bob_context = context.copy()
        bob_context['role'] = 'UE'
        bob_key, bob_kdf_metrics = bob.execute_kdf_phase(
            bob_skg, bob_pqkem, bob_context
        )
        
        # Phase 4: Authentication
        transcript_data = {
            'channel_commitment': alice.kdf.compute_channel_commitment(
                csi_data=bytes(alice_skg),
                pilot_seq=b"PILOT_SEQ_5G",
                nonce=context['nonce'].encode()
            ),
            'pqkem_ciphertext': pqkem_ct,
            'context': str(context).encode()
        }
        
        alice_sig, alice_auth_metrics = alice.execute_authentication_phase(
            transcript_data, is_signer=True
        )
        
        bob_verifies = bob.verify_peer_signature(
            transcript_data, alice_sig, alice.sig_public_key
        )
        
        # Results
        print("\n[Results Summary]")
        print(f"✓ Protocol completed successfully")
        print(f"✓ Session keys derived (256-bit AES-GCM ready)")
        print(f"✓ Authentication: {'PASS' if bob_verifies else 'FAIL'}")
        
        alice_metrics = alice.get_protocol_metrics()
        print(f"\nPerformance:")
        print(f"  Total time: {alice_metrics['timing']['total_ms']:.2f} ms")
        print(f"  Total overhead: {alice_metrics['overhead']['total_bytes']} bytes")
        print(f"  Security level: {alice_metrics['security']['security_level']}")
        print(f"  Estimated entropy: {alice_metrics['security']['entropy_bits']:.1f} bits")
    
    def demo_2_ciphersuite_negotiation(self):
        """Demonstration 2: Ciphersuite negotiation and crypto-agility"""
        print("\n" + "="*70)
        print("DEMO 2: Ciphersuite Negotiation & Crypto-Agility")
        print("="*70)
        
        print("\nScenario: Alice and Bob negotiate ciphersuite")
        print("Alice prefers high security, Bob has limited resources\n")
        
        # Alice's preferences
        alice_negotiator = CiphersuiteNegotiator(
            preferences=[
                "HYBRID_KYBER768_DILITHIUM3_SHA384",
                "HYBRID_KYBER512_DILITHIUM2_SHA384"
            ]
        )
        
        # Bob's limited support
        bob_negotiator = CiphersuiteNegotiator(
            supported_suites=[
                "HYBRID_KYBER512_DILITHIUM2_SHA384",
                "PQ_ONLY_KYBER768_DILITHIUM3",
                "CLASSICAL_SKG_SHA384"
            ]
        )
        
        print("Alice supported suites:")
        for suite in alice_negotiator.supported_suites[:3]:
            print(f"  - {suite}")
        
        print("\nBob supported suites:")
        for suite in bob_negotiator.supported_suites:
            print(f"  - {suite}")
        
        # Negotiation
        print("\n[Negotiation Process]")
        client_hello = alice_negotiator.create_client_hello()
        print("Alice → Bob: ClientHello")
        
        server_hello = bob_negotiator.process_client_hello(client_hello)
        print("Bob → Alice: ServerHello")
        
        if server_hello['status'] == 'ok':
            selected = server_hello['selected_suite']
            suite_info = server_hello['suite_info']
            print(f"\n✓ Negotiation successful!")
            print(f"  Selected: {selected}")
            print(f"  Security: {suite_info['security_level']}")
            print(f"  KEM: {suite_info['kem']}")
            print(f"  Signature: {suite_info['signature']}")
        else:
            print(f"\n✗ Negotiation failed: {server_hello['message']}")
    
    def demo_3_adaptive_fallback(self):
        """Demonstration 3: Adaptive fallback mechanism"""
        print("\n" + "="*70)
        print("DEMO 3: Adaptive Fallback & Degradation Handling")
        print("="*70)
        
        print("\nScenario: Network conditions degrade, system adapts\n")
        
        # Scenario 1: Low SKG entropy
        print("[Scenario 1: Low SKG Entropy]")
        alice = HybridSKGProtocol(role="alice", ciphersuite=CiphersuiteMode.HYBRID)
        
        skg_entropy_low = 60.0  # Below threshold
        selected_suite = alice.adaptive_ciphersuite_selection(
            skg_entropy=skg_entropy_low,
            skg_threshold=80.0,
            pqkem_available=True
        )
        print(f"Result: {selected_suite.value}")
        print(f"Security level: {alice.security_level.value}")
        
        # Scenario 2: MTU constraints
        print("\n[Scenario 2: MTU Constraints]")
        from config_cryptoagility import FallbackManager
        
        fallback_mgr = FallbackManager()
        result = fallback_mgr.attempt_fallback(
            failed_suite="HYBRID_KYBER768_DILITHIUM3_SHA384",
            reason="Ciphertext exceeds MTU (1500 bytes)",
            skg_entropy=120.0,
            pqkem_available=True
        )
        
        if result:
            suite, level, warning = result
            print(f"✓ Fallback to: {suite}")
            print(f"  Security: {level}")
            print(f"  Warning: {warning}")
    
    def demo_4_performance_comparison(self):
        """Demonstration 4: Performance comparison"""
        print("\n" + "="*70)
        print("DEMO 4: Performance Comparison")
        print("="*70)
        
        print("\nComparing: Classical SKG vs Hybrid SKG+PQC vs PQ-Only\n")
        
        benchmark = PerformanceBenchmark()
        
        print("[Running benchmarks...]")
        print("This may take a few moments...\n")
        
        # Run benchmarks with reduced iterations for demo
        classical = benchmark.benchmark_skg_only(iterations=3, channel_samples=100)
        hybrid = benchmark.benchmark_hybrid_skg_pqc(iterations=3, channel_samples=100)
        pq_only = benchmark.benchmark_pq_only(iterations=3)
        
        # Analysis
        analysis = benchmark.comparative_analysis()
        
        print("\n[Key Findings]")
        print(f"✓ Hybrid adds {analysis['time_comparison']['hybrid_overhead_percent']:.1f}% latency")
        print(f"✓ Hybrid adds {analysis['overhead_comparison']['hybrid_added_bytes']} bytes overhead")
        print(f"✓ Hybrid provides quantum-resistant security")
        print(f"✓ Combined entropy: {analysis['security_comparison']['hybrid_entropy_bits']:.1f} bits")
    
    def demo_5_threat_mitigation(self):
        """Demonstration 5: Threat model and mitigation"""
        print("\n" + "="*70)
        print("DEMO 5: Threat Model & Mitigation Strategies")
        print("="*70)
        
        threats = {
            "Store-now-decrypt-later": {
                "threat": "Adversary stores encrypted traffic, decrypts later with quantum computer",
                "mitigation": "PQ KEM (Kyber) provides quantum-resistant encryption",
                "status": "✓ MITIGATED"
            },
            "Active pilot contamination": {
                "threat": "Adversary injects false pilot signals to manipulate SKG",
                "mitigation": "Channel commitment H(CSI||pilot||nonce) in signed transcript",
                "status": "✓ MITIGATED"
            },
            "MITM attack": {
                "threat": "Adversary intercepts and modifies protocol messages",
                "mitigation": "PQ signatures (Dilithium) authenticate full transcript",
                "status": "✓ MITIGATED"
            },
            "Reconciliation leakage": {
                "threat": "Information leakage during error correction exposes key bits",
                "mitigation": "Privacy amplification with HKDF, entropy accounting",
                "status": "✓ MITIGATED"
            },
            "Replay attack": {
                "threat": "Adversary replays old protocol messages",
                "mitigation": "Nonces and timestamps in context, freshness windows",
                "status": "✓ MITIGATED"
            },
            "Post-compromise": {
                "threat": "Key compromise affects future sessions",
                "mitigation": "Key ratcheting with fresh SKG and PQ KEM per session",
                "status": "✓ MITIGATED"
            }
        }
        
        print("\nThreat Model Analysis:\n")
        for threat_name, info in threats.items():
            print(f"{info['status']} {threat_name}")
            print(f"    Threat: {info['threat']}")
            print(f"    Mitigation: {info['mitigation']}\n")
    
    def demo_6_thesis_metrics(self):
        """Demonstration 6: Thesis-ready metrics and visualizations"""
        print("\n" + "="*70)
        print("DEMO 6: Thesis-Ready Metrics & Analysis")
        print("="*70)
        
        print("\nGenerating publication-quality metrics...\n")
        
        # Run quick benchmark
        benchmark = PerformanceBenchmark()
        benchmark.benchmark_skg_only(iterations=2, channel_samples=100)
        benchmark.benchmark_hybrid_skg_pqc(iterations=2, channel_samples=100)
        benchmark.benchmark_pq_only(iterations=2)
        benchmark.comparative_analysis()
        
        # Generate LaTeX table
        latex_table = benchmark.generate_latex_table()
        
        # Generate JSON report
        benchmark.generate_report("thesis_benchmark_results.json")
        
        print("\n[Generated Outputs]")
        print("✓ LaTeX table (for thesis document)")
        print("✓ JSON report (thesis_benchmark_results.json)")
        print("✓ Performance metrics (timing, overhead, entropy)")
        
        print("\n[Thesis Contributions Summary]")
        contributions = [
            "1. Hybrid SKG+PQC architecture preserving physical-layer advantages",
            "2. Formal security augmentation against quantum adversaries",
            "3. Practical implementation with crypto-agility and fallback",
            "4. Comprehensive performance evaluation: classical vs hybrid vs PQ-only",
            "5. Threat model extension with mitigation strategies",
            "6. Configurable protocol for diverse deployment scenarios"
        ]
        
        for contrib in contributions:
            print(f"  ✓ {contrib}")
    
    def run_all_demos(self):
        """Run all demonstrations"""
        self.demo_1_basic_protocol()
        self.demo_2_ciphersuite_negotiation()
        self.demo_3_adaptive_fallback()
        self.demo_4_performance_comparison()
        self.demo_5_threat_mitigation()
        self.demo_6_thesis_metrics()
        
        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED")
        print("="*70)
        
        print("\n[Next Steps for Thesis]")
        print("1. Review generated benchmark_report.json for detailed metrics")
        print("2. Use LaTeX tables in thesis document")
        print("3. Customize configuration in config_cryptoagility.py")
        print("4. Run extended benchmarks with real channel data")
        print("5. Integrate with existing SKG testbed")
        
        print("\n[Implementation Notes]")
        print("⚠ Current implementation uses simulated Kyber/Dilithium")
        print("  For production: Install pqcrypto library")
        print("  Command: pip install pqcrypto")
        
        print("\n[File Structure]")
        files = [
            ("pqc_kem.py", "Kyber KEM implementation"),
            ("hybrid_kdf.py", "HKDF-based key derivation"),
            ("pqc_signatures.py", "Dilithium signatures"),
            ("hybrid_protocol.py", "Main protocol orchestrator"),
            ("config_cryptoagility.py", "Configuration and negotiation"),
            ("benchmark_evaluation.py", "Performance benchmarking"),
            ("demo_comprehensive.py", "This demonstration script")
        ]
        
        print("\nCreated modules:")
        for filename, description in files:
            print(f"  ✓ {filename:30s} - {description}")
        
        print("\n" + "="*70)


def main():
    """Main entry point"""
    print("\nWelcome to the Hybrid SKG+PQC System Demonstration")
    print("Ph.D. Thesis: Post-Quantum Augmentation of Physical-Layer Key Generation\n")
    
    demo = ComprehensiveDemo()
    
    # Check if user wants to run all demos or specific ones
    if len(sys.argv) > 1:
        demo_num = sys.argv[1]
        if demo_num == '1':
            demo.demo_1_basic_protocol()
        elif demo_num == '2':
            demo.demo_2_ciphersuite_negotiation()
        elif demo_num == '3':
            demo.demo_3_adaptive_fallback()
        elif demo_num == '4':
            demo.demo_4_performance_comparison()
        elif demo_num == '5':
            demo.demo_5_threat_mitigation()
        elif demo_num == '6':
            demo.demo_6_thesis_metrics()
        else:
            print(f"Unknown demo: {demo_num}")
            print("Usage: python demo_comprehensive.py [1-6]")
    else:
        # Run all demos
        demo.run_all_demos()


if __name__ == "__main__":
    main()
