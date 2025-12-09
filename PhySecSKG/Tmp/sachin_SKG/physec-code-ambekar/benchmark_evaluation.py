"""
Performance Benchmarking and Evaluation Module
Comprehensive metrics collection for thesis evaluation
Compares SKG-only vs SKG+PQ hybrid vs PQ-only
"""

import time
import os
import statistics
import json
from typing import Dict, List, Any, Optional
import numpy as np


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking for hybrid SKG+PQC
    """
    
    def __init__(self):
        """Initialize benchmark suite"""
        self.results = {
            'classical_skg': [],
            'hybrid_skg_pqc': [],
            'pq_only': []
        }
        self.analysis_results = {}
    
    def benchmark_skg_only(self, 
                          iterations: int = 10,
                          channel_samples: int = 100) -> Dict[str, Any]:
        """
        Benchmark classical SKG-only approach
        
        Args:
            iterations: Number of iterations
            channel_samples: Number of channel measurements
            
        Returns:
            Benchmark results dictionary
        """
        print(f"\n=== Benchmarking SKG-Only (Classical) ===")
        print(f"Iterations: {iterations}, Samples: {channel_samples}")
        
        from quantizer import Quantizer
        from keyencryption import Keyencryption
        from bitdisagreement import Bitdisagreement
        
        results = []
        
        for i in range(iterations):
            # Simulate channel measurements
            base_channel = np.random.randn(channel_samples) * 10 + 50
            alice_measurements = (base_channel + np.random.randn(channel_samples) * 0.5).tolist()
            bob_measurements = (base_channel + np.random.randn(channel_samples) * 0.5).tolist()
            
            # Time SKG process
            start_time = time.time()
            
            # Alice quantizes
            alice_quantizer = Quantizer(alice_measurements, 0.5)
            alice_key = alice_quantizer.quantize_var()
            
            # Bob quantizes
            bob_quantizer = Quantizer(bob_measurements, 0.5)
            bob_key = bob_quantizer.quantize_var()
            
            # Privacy amplification
            encryptor = Keyencryption()
            alice_final = encryptor.encryptkey(alice_key)
            bob_final = encryptor.encryptkey(bob_key)
            
            elapsed_time = time.time() - start_time
            
            # Calculate BDR
            bdr_checker = Bitdisagreement()
            disagreed_bits, bdr_rate, positions = bdr_checker.bdr(alice_key, bob_key)
            
            results.append({
                'time_ms': elapsed_time * 1000,
                'key_bits': len(alice_key),
                'disagreed_bits': disagreed_bits,
                'bdr_percent': bdr_rate,
                'overhead_bytes': 0  # No additional overhead for classical
            })
        
        # Aggregate statistics
        avg_results = {
            'mode': 'classical_skg',
            'iterations': iterations,
            'avg_time_ms': statistics.mean([r['time_ms'] for r in results]),
            'std_time_ms': statistics.stdev([r['time_ms'] for r in results]) if iterations > 1 else 0,
            'avg_key_bits': statistics.mean([r['key_bits'] for r in results]),
            'avg_bdr_percent': statistics.mean([r['bdr_percent'] for r in results]),
            'avg_overhead_bytes': 0,
            'raw_results': results
        }
        
        self.results['classical_skg'].append(avg_results)
        
        print(f"Average time: {avg_results['avg_time_ms']:.3f} ms")
        print(f"Average BDR: {avg_results['avg_bdr_percent']:.2f}%")
        
        return avg_results
    
    def benchmark_hybrid_skg_pqc(self,
                                iterations: int = 10,
                                channel_samples: int = 100) -> Dict[str, Any]:
        """
        Benchmark hybrid SKG+PQC approach
        
        Args:
            iterations: Number of iterations
            channel_samples: Number of channel measurements
            
        Returns:
            Benchmark results dictionary
        """
        print(f"\n=== Benchmarking Hybrid SKG+PQC ===")
        print(f"Iterations: {iterations}, Samples: {channel_samples}")
        
        from hybrid_protocol import HybridSKGProtocol, CiphersuiteMode
        
        results = []
        
        for i in range(iterations):
            # Initialize protocol
            alice = HybridSKGProtocol(role="alice", ciphersuite=CiphersuiteMode.HYBRID)
            bob = HybridSKGProtocol(role="bob", ciphersuite=CiphersuiteMode.HYBRID)
            
            start_time = time.time()
            
            # Initialize
            alice.initialize()
            bob.initialize()
            
            # Simulate channel
            base_channel = np.random.randn(channel_samples) * 10 + 50
            alice_measurements = (base_channel + np.random.randn(channel_samples) * 0.5).tolist()
            bob_measurements = (base_channel + np.random.randn(channel_samples) * 0.5).tolist()
            
            # SKG phase
            alice_skg, _ = alice.execute_skg_phase(alice_measurements, {'deviation': 0.5, 'method': 'var'})
            bob_skg, _ = bob.execute_skg_phase(bob_measurements, {'deviation': 0.5, 'method': 'var'})
            
            # PQ KEM phase
            pqkem_ct, alice_pqkem, _ = alice.execute_pqkem_phase(bob.public_key, is_initiator=True)
            bob_pqkem = bob.decapsulate_pqkem(pqkem_ct)
            
            # KDF phase
            context = {'role': 'alice', 'timestamp': int(time.time())}
            alice_key, _ = alice.execute_kdf_phase(alice_skg, alice_pqkem, context)
            
            context['role'] = 'bob'
            bob_key, _ = bob.execute_kdf_phase(bob_skg, bob_pqkem, context)
            
            # Authentication phase
            transcript_data = {'context': str(context).encode()}
            alice_sig, _ = alice.execute_authentication_phase(transcript_data, is_signer=True)
            
            elapsed_time = time.time() - start_time
            
            # Get metrics
            alice_metrics = alice.get_protocol_metrics()
            
            results.append({
                'time_ms': elapsed_time * 1000,
                'skg_time_ms': alice_metrics['timing']['skg_ms'],
                'pqkem_time_ms': alice_metrics['timing']['pqkem_ms'],
                'kdf_time_ms': alice_metrics['timing']['kdf_ms'],
                'sig_time_ms': alice_metrics['timing']['signature_ms'],
                'overhead_bytes': alice_metrics['overhead']['total_bytes'],
                'entropy_bits': alice_metrics['security']['entropy_bits']
            })
        
        # Aggregate statistics
        avg_results = {
            'mode': 'hybrid_skg_pqc',
            'iterations': iterations,
            'avg_total_time_ms': statistics.mean([r['time_ms'] for r in results]),
            'std_time_ms': statistics.stdev([r['time_ms'] for r in results]) if iterations > 1 else 0,
            'avg_skg_time_ms': statistics.mean([r['skg_time_ms'] for r in results]),
            'avg_pqkem_time_ms': statistics.mean([r['pqkem_time_ms'] for r in results]),
            'avg_kdf_time_ms': statistics.mean([r['kdf_time_ms'] for r in results]),
            'avg_sig_time_ms': statistics.mean([r['sig_time_ms'] for r in results]),
            'avg_overhead_bytes': statistics.mean([r['overhead_bytes'] for r in results]),
            'avg_entropy_bits': statistics.mean([r['entropy_bits'] for r in results]),
            'raw_results': results
        }
        
        self.results['hybrid_skg_pqc'].append(avg_results)
        
        print(f"Average total time: {avg_results['avg_total_time_ms']:.3f} ms")
        print(f"Average overhead: {avg_results['avg_overhead_bytes']:.0f} bytes")
        print(f"Average entropy: {avg_results['avg_entropy_bits']:.1f} bits")
        
        return avg_results
    
    def benchmark_pq_only(self,
                         iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark PQ-only approach (no SKG)
        
        Args:
            iterations: Number of iterations
            
        Returns:
            Benchmark results dictionary
        """
        print(f"\n=== Benchmarking PQ-Only ===")
        print(f"Iterations: {iterations}")
        
        from pqc_kem import PQCKyberKEM
        from hybrid_kdf import HybridKDF
        from pqc_signatures import PQCDilithiumSignature
        
        results = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # KEM exchange
            alice_kem = PQCKyberKEM()
            bob_kem = PQCKyberKEM()
            
            alice_pk, alice_sk = alice_kem.generate_keypair()
            bob_pk, bob_sk = bob_kem.generate_keypair()
            
            ct, alice_ss = alice_kem.encapsulate(bob_pk)
            bob_ss = bob_kem.decapsulate(ct, bob_sk)
            
            # KDF
            kdf = HybridKDF()
            alice_key = kdf.derive_session_key(
                ss_skg=b"",
                ss_pqkem=alice_ss,
                context={'role': 'alice'},
                key_length=32
            )
            
            # Signature
            sig = PQCDilithiumSignature()
            sig_pk, sig_sk = sig.generate_keypair()
            signature = sig.sign_transcript(b"transcript_data", sig_sk)
            
            elapsed_time = time.time() - start_time
            
            overhead = len(alice_pk) + len(ct) + len(sig_pk) + len(signature)
            
            results.append({
                'time_ms': elapsed_time * 1000,
                'overhead_bytes': overhead,
                'entropy_bits': len(alice_ss) * 8
            })
        
        # Aggregate statistics
        avg_results = {
            'mode': 'pq_only',
            'iterations': iterations,
            'avg_time_ms': statistics.mean([r['time_ms'] for r in results]),
            'std_time_ms': statistics.stdev([r['time_ms'] for r in results]) if iterations > 1 else 0,
            'avg_overhead_bytes': statistics.mean([r['overhead_bytes'] for r in results]),
            'avg_entropy_bits': statistics.mean([r['entropy_bits'] for r in results]),
            'raw_results': results
        }
        
        self.results['pq_only'].append(avg_results)
        
        print(f"Average time: {avg_results['avg_time_ms']:.3f} ms")
        print(f"Average overhead: {avg_results['avg_overhead_bytes']:.0f} bytes")
        
        return avg_results
    
    def comparative_analysis(self) -> Dict[str, Any]:
        """
        Perform comparative analysis across all modes
        
        Returns:
            Comparative analysis results
        """
        print("\n" + "="*70)
        print("COMPARATIVE ANALYSIS")
        print("="*70)
        
        if not self.results['classical_skg'] or not self.results['hybrid_skg_pqc']:
            print("Error: Run benchmarks first")
            return {}
        
        classical = self.results['classical_skg'][-1]
        hybrid = self.results['hybrid_skg_pqc'][-1]
        pq_only = self.results['pq_only'][-1] if self.results['pq_only'] else None
        
        analysis = {
            'time_comparison': {
                'classical_ms': classical['avg_time_ms'],
                'hybrid_ms': hybrid['avg_total_time_ms'],
                'pq_only_ms': pq_only['avg_time_ms'] if pq_only else None,
                'hybrid_overhead_vs_classical': hybrid['avg_total_time_ms'] - classical['avg_time_ms'],
                'hybrid_overhead_percent': ((hybrid['avg_total_time_ms'] / classical['avg_time_ms']) - 1) * 100
            },
            'overhead_comparison': {
                'classical_bytes': classical['avg_overhead_bytes'],
                'hybrid_bytes': hybrid['avg_overhead_bytes'],
                'pq_only_bytes': pq_only['avg_overhead_bytes'] if pq_only else None,
                'hybrid_added_bytes': hybrid['avg_overhead_bytes'] - classical['avg_overhead_bytes']
            },
            'security_comparison': {
                'classical_quantum_resistant': False,
                'hybrid_quantum_resistant': True,
                'pq_only_quantum_resistant': True,
                'hybrid_entropy_bits': hybrid['avg_entropy_bits'],
                'classical_bdr_percent': classical.get('avg_bdr_percent', 'N/A')
            },
            'time_breakdown_hybrid': {
                'skg_percent': (hybrid['avg_skg_time_ms'] / hybrid['avg_total_time_ms']) * 100,
                'pqkem_percent': (hybrid['avg_pqkem_time_ms'] / hybrid['avg_total_time_ms']) * 100,
                'kdf_percent': (hybrid['avg_kdf_time_ms'] / hybrid['avg_total_time_ms']) * 100,
                'sig_percent': (hybrid['avg_sig_time_ms'] / hybrid['avg_total_time_ms']) * 100
            }
        }
        
        self.analysis_results = analysis
        
        # Print analysis
        print("\n### Time Comparison ###")
        print(f"Classical SKG: {analysis['time_comparison']['classical_ms']:.3f} ms")
        print(f"Hybrid SKG+PQC: {analysis['time_comparison']['hybrid_ms']:.3f} ms")
        if pq_only:
            print(f"PQ-Only: {analysis['time_comparison']['pq_only_ms']:.3f} ms")
        print(f"Hybrid overhead: +{analysis['time_comparison']['hybrid_overhead_percent']:.1f}%")
        
        print("\n### Overhead Comparison ###")
        print(f"Classical: {analysis['overhead_comparison']['classical_bytes']} bytes")
        print(f"Hybrid: {analysis['overhead_comparison']['hybrid_bytes']} bytes")
        if pq_only:
            print(f"PQ-Only: {analysis['overhead_comparison']['pq_only_bytes']} bytes")
        print(f"Hybrid added: +{analysis['overhead_comparison']['hybrid_added_bytes']} bytes")
        
        print("\n### Security Comparison ###")
        print(f"Classical quantum-resistant: {analysis['security_comparison']['classical_quantum_resistant']}")
        print(f"Hybrid quantum-resistant: {analysis['security_comparison']['hybrid_quantum_resistant']}")
        print(f"Hybrid entropy: {analysis['security_comparison']['hybrid_entropy_bits']:.1f} bits")
        
        print("\n### Hybrid Time Breakdown ###")
        for component, percent in analysis['time_breakdown_hybrid'].items():
            print(f"{component}: {percent:.1f}%")
        
        return analysis
    
    def generate_report(self, output_file: str = "benchmark_report.json"):
        """
        Generate comprehensive benchmark report
        
        Args:
            output_file: Output JSON file path
        """
        report = {
            'timestamp': time.time(),
            'results': self.results,
            'comparative_analysis': self.analysis_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n[Report] Saved to {output_file}")
    
    def generate_latex_table(self) -> str:
        """
        Generate LaTeX table for thesis
        
        Returns:
            LaTeX table code
        """
        if not self.analysis_results:
            return "% Run comparative_analysis() first"
        
        analysis = self.analysis_results
        
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Performance Comparison: Classical SKG vs Hybrid SKG+PQC}
\label{tab:skg_pqc_comparison}
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{Classical} & \textbf{Hybrid} & \textbf{Overhead} \\
\midrule
"""
        
        classical_time = analysis['time_comparison']['classical_ms']
        hybrid_time = analysis['time_comparison']['hybrid_ms']
        time_overhead = analysis['time_comparison']['hybrid_overhead_percent']
        
        classical_bytes = analysis['overhead_comparison']['classical_bytes']
        hybrid_bytes = analysis['overhead_comparison']['hybrid_bytes']
        bytes_added = analysis['overhead_comparison']['hybrid_added_bytes']
        
        hybrid_entropy = analysis['security_comparison']['hybrid_entropy_bits']
        
        latex += f"Latency (ms) & {classical_time:.2f} & {hybrid_time:.2f} & +{time_overhead:.1f}\\% \\\\\n"
        latex += f"Overhead (bytes) & {classical_bytes} & {hybrid_bytes} & +{bytes_added} \\\\\n"
        latex += f"Entropy (bits) & -- & {hybrid_entropy:.1f} & -- \\\\\n"
        latex += "Quantum-resistant & No & Yes & -- \\\\\n"
        
        latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        
        print("\n=== LaTeX Table ===")
        print(latex)
        
        return latex


def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite"""
    print("="*70)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("="*70)
    
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    benchmark.benchmark_skg_only(iterations=5, channel_samples=100)
    benchmark.benchmark_hybrid_skg_pqc(iterations=5, channel_samples=100)
    benchmark.benchmark_pq_only(iterations=5)
    
    # Comparative analysis
    benchmark.comparative_analysis()
    
    # Generate report
    benchmark.generate_report("benchmark_report.json")
    
    # Generate LaTeX table
    benchmark.generate_latex_table()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    run_comprehensive_benchmark()
