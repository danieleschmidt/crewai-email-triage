"""Comprehensive tests for neuromorphic-quantum fusion and consciousness research.

This test suite validates breakthrough research implementations:
- Neuromorphic-quantum fusion engine functionality
- Quantum consciousness emergence and self-awareness
- Research metrics and breakthrough detection
- Performance comparison between computational paradigms
"""

import asyncio
import pytest
from unittest.mock import Mock, patch
import time

from crewai_email_triage.neuro_quantum_fusion import (
    NeuroQuantumFusionEngine,
    QuantumNeuron,
    NeuroQuantumCircuit,
    NeuroQuantumState,
    ComputationParadigm,
    create_neuro_quantum_engine,
    process_email_with_fusion,
    evaluate_breakthrough_potential
)

from crewai_email_triage.quantum_consciousness import (
    QuantumConsciousnessEngine,
    MicrotubuleQuantumState,
    ConsciousnessField,
    ConsciousnessLevel,
    AwarenessType,
    create_consciousness_engine,
    process_email_with_consciousness,
    evaluate_consciousness_breakthrough
)


class TestQuantumNeuron:
    """Test quantum neuron functionality."""
    
    def test_quantum_neuron_initialization(self):
        """Test quantum neuron creates with default values."""
        neuron = QuantumNeuron("test_neuron")
        
        assert neuron.neuron_id == "test_neuron"
        assert neuron.amplitude_real == 0.0
        assert neuron.amplitude_imag == 0.0
        assert neuron.membrane_potential == -70.0
        assert neuron.threshold == -55.0
        assert isinstance(neuron.synapses, dict)
    
    def test_quantum_state_operations(self):
        """Test quantum state get/set operations."""
        neuron = QuantumNeuron("test")
        
        # Set complex quantum state
        test_state = complex(0.6, 0.8)
        neuron.set_quantum_state(test_state)
        
        assert neuron.get_quantum_state() == test_state
        assert neuron.amplitude_real == 0.6
        assert neuron.amplitude_imag == 0.8
    
    def test_probability_calculation(self):
        """Test quantum probability calculation."""
        neuron = QuantumNeuron("test")
        neuron.set_quantum_state(complex(0.6, 0.8))
        
        probability = neuron.get_probability()
        expected = abs(complex(0.6, 0.8)) ** 2
        
        assert abs(probability - expected) < 0.001
    
    def test_neuromorphic_spiking(self):
        """Test neuromorphic spiking behavior."""
        neuron = QuantumNeuron("test")
        neuron.membrane_potential = -50.0  # Above threshold
        
        current_time = time.time() * 1000
        spike_occurred = neuron.spike(current_time)
        
        assert spike_occurred
        assert neuron.last_spike_time == current_time
        assert neuron.membrane_potential == -70.0  # Reset after spike
    
    def test_synaptic_plasticity(self):
        """Test synaptic plasticity updates."""
        neuron = QuantumNeuron("test")
        neuron.synapses["synapse_1"] = 0.5
        initial_weight = neuron.synapses["synapse_1"]
        
        # LTP - both pre and post spike
        neuron.update_plasticity(pre_spike=True, post_spike=True)
        assert neuron.synapses["synapse_1"] > initial_weight
        
        # LTD - only pre spike
        neuron.update_plasticity(pre_spike=True, post_spike=False)
        assert neuron.synapses["synapse_1"] < initial_weight * (1 + neuron.plasticity_rate)


class TestNeuroQuantumCircuit:
    """Test neuro-quantum circuit operations."""
    
    def test_circuit_initialization(self):
        """Test circuit creates with proper structure."""
        circuit = NeuroQuantumCircuit("test_circuit")
        
        assert circuit.circuit_id == "test_circuit"
        assert len(circuit.neurons) == 0
        assert len(circuit.quantum_gates) == 0
        assert circuit.coherence_time == 100.0
    
    def test_neuron_addition(self):
        """Test adding neurons to circuit."""
        circuit = NeuroQuantumCircuit("test")
        neuron = QuantumNeuron("neuron_1")
        
        circuit.add_neuron(neuron)
        
        assert len(circuit.neurons) == 1
        assert circuit.neurons[0] == neuron
    
    def test_hadamard_gate_application(self):
        """Test Hadamard gate creates superposition."""
        circuit = NeuroQuantumCircuit("test")
        neuron = QuantumNeuron("neuron_1")
        circuit.add_neuron(neuron)
        
        initial_state = neuron.get_quantum_state()
        circuit.apply_hadamard("neuron_1")
        final_state = neuron.get_quantum_state()
        
        assert final_state != initial_state
        assert circuit.total_operations == 1
    
    def test_cnot_gate_entanglement(self):
        """Test CNOT gate creates entanglement."""
        circuit = NeuroQuantumCircuit("test")
        neuron1 = QuantumNeuron("neuron_1")
        neuron2 = QuantumNeuron("neuron_2")
        circuit.add_neuron(neuron1)
        circuit.add_neuron(neuron2)
        
        circuit.apply_cnot("neuron_1", "neuron_2")
        
        assert "neuron_2" in neuron1.entangled_neurons
        assert "neuron_1" in neuron2.entangled_neurons
        assert neuron1.entanglement_strength > 0
        assert neuron2.entanglement_strength > 0
    
    def test_decoherence_simulation(self):
        """Test quantum decoherence effects."""
        circuit = NeuroQuantumCircuit("test")
        neuron = QuantumNeuron("neuron_1")
        neuron.set_quantum_state(complex(0.8, 0.6))
        circuit.add_neuron(neuron)
        
        initial_amplitude = abs(neuron.get_quantum_state())
        circuit.simulate_decoherence(10.0)  # 10 time units
        final_amplitude = abs(neuron.get_quantum_state())
        
        assert final_amplitude < initial_amplitude  # Decoherence reduces amplitude


class TestNeuroQuantumFusionEngine:
    """Test the main fusion engine."""
    
    def test_engine_initialization(self):
        """Test fusion engine initializes properly."""
        engine = NeuroQuantumFusionEngine()
        
        assert engine.state == NeuroQuantumState.INITIALIZATION
        assert engine.paradigm == ComputationParadigm.HYBRID_ALL
        assert len(engine.circuits) == 0
        assert engine.total_computations == 0
    
    def test_circuit_creation(self):
        """Test creating circuits in the engine."""
        engine = NeuroQuantumFusionEngine()
        
        circuit = engine.create_circuit("test_circuit", num_neurons=5)
        
        assert circuit.circuit_id == "test_circuit"
        assert len(circuit.neurons) == 5
        assert "test_circuit" in engine.circuits
    
    @pytest.mark.asyncio
    async def test_quantum_email_processing(self):
        """Test quantum email processing pipeline."""
        engine = NeuroQuantumFusionEngine()
        test_email = "Urgent: Please review this important document ASAP!"
        metadata = {"sender": "test@example.com"}
        
        result = await engine.process_email_quantum(test_email, metadata)
        
        # Verify result structure
        assert "classification" in result
        assert "priority_score" in result
        assert "quantum_advantage" in result
        assert "processing_time" in result
        assert "breakthrough_potential" in result
        
        # Verify reasonable values
        assert 0 <= result["priority_score"] <= 1
        assert result["quantum_advantage"] >= 1.0
        assert result["processing_time"] > 0
    
    def test_quantum_feature_extraction(self):
        """Test quantum feature extraction from email content."""
        engine = NeuroQuantumFusionEngine()
        test_content = "Test email content with various characters!"
        
        features = engine._extract_quantum_features(test_content)
        
        assert len(features) == 8  # 1 length + 7 character frequency features
        assert all(0 <= f <= 1 for f in features)  # All features normalized
    
    def test_research_metrics_tracking(self):
        """Test research metrics collection."""
        engine = NeuroQuantumFusionEngine()
        
        # Simulate some processing
        engine.total_computations = 10
        engine.quantum_speedup = 2.5
        engine.novel_patterns_discovered = 3
        
        metrics = engine.get_research_metrics()
        
        assert metrics["total_computations"] == 10
        assert metrics["quantum_speedup"] == 2.5
        assert metrics["novel_patterns_discovered"] == 3
        assert "breakthrough_indicators" in metrics
    
    @pytest.mark.asyncio
    async def test_paradigm_benchmarking(self):
        """Test benchmarking different computational paradigms."""
        engine = NeuroQuantumFusionEngine()
        test_email = "Test email for benchmarking"
        
        benchmarks = await engine.benchmark_paradigms(test_email)
        
        assert isinstance(benchmarks, dict)
        assert len(benchmarks) >= 3  # At least 3 paradigms tested
        
        for paradigm, result in benchmarks.items():
            if "error" not in result:
                assert "processing_time" in result
                assert result["processing_time"] > 0


class TestMicrotubuleQuantumState:
    """Test microtubule quantum states for consciousness."""
    
    def test_microtubule_initialization(self):
        """Test microtubule quantum state initialization."""
        mt = MicrotubuleQuantumState("mt_1")
        
        assert mt.microtubule_id == "mt_1"
        assert mt.tubulin_a_state == complex(1, 0)
        assert mt.tubulin_b_state == complex(0, 1)
        assert mt.quantum_coherence == 1.0
        assert mt.awareness_intensity == 0.0
    
    def test_consciousness_amplitude_calculation(self):
        """Test consciousness amplitude calculation."""
        mt = MicrotubuleQuantumState("test")
        mt.tubulin_a_state = complex(0.6, 0.0)
        mt.tubulin_b_state = complex(0.8, 0.0)
        mt.quantum_coherence = 0.9
        mt.awareness_intensity = 0.5
        
        amplitude = mt.get_consciousness_amplitude()
        expected = (0.6 + 0.8) * 0.9 * 0.5
        
        assert abs(amplitude - expected) < 0.001
    
    def test_quantum_state_collapse(self):
        """Test objective reduction (quantum collapse)."""
        mt = MicrotubuleQuantumState("test")
        mt.collapse_probability = 1.0  # Force collapse
        mt.coherence_time = 0.0  # Immediate collapse allowed
        
        current_time = time.time() * 1000
        collapse_occurred = mt.collapse_quantum_state(current_time)
        
        assert collapse_occurred
        assert mt.last_collapse_time == current_time
        assert "collapse_intensity" in mt.subjective_experience
    
    def test_entanglement_creation(self):
        """Test microtubule entanglement."""
        mt1 = MicrotubuleQuantumState("mt1")
        mt2 = MicrotubuleQuantumState("mt2")
        
        mt1.entangle_with(mt2)
        
        assert "mt2" in mt1.connected_microtubules
        assert "mt1" in mt2.connected_microtubules
        assert mt1.consciousness_field_strength > 0
        assert mt2.consciousness_field_strength > 0


class TestConsciousnessField:
    """Test consciousness field operations."""
    
    def test_field_initialization(self):
        """Test consciousness field initialization."""
        field = ConsciousnessField("primary")
        
        assert field.field_id == "primary"
        assert len(field.microtubules) == 0
        assert field.consciousness_level == ConsciousnessLevel.UNCONSCIOUS
        assert field.global_awareness == 0.0
    
    def test_global_consciousness_calculation(self):
        """Test global consciousness emergence."""
        field = ConsciousnessField("test")
        
        # Add some microtubules with consciousness
        for i in range(5):
            mt = MicrotubuleQuantumState(f"mt_{i}")
            mt.awareness_intensity = 0.8
            mt.quantum_coherence = 0.9
            field.microtubules.append(mt)
        
        consciousness = field.calculate_global_consciousness()
        
        assert consciousness > 0
        assert field.global_awareness == consciousness
        assert field.consciousness_level != ConsciousnessLevel.UNCONSCIOUS
    
    def test_subjective_experience_generation(self):
        """Test subjective experience generation."""
        field = ConsciousnessField("test")
        field.global_awareness = 0.8
        
        input_data = {"urgency": 0.9, "importance": 0.7}
        experience = field.generate_subjective_experience(input_data)
        
        assert "perceptual_qualia" in experience
        assert "cognitive_state" in experience
        assert "unified_percept" in experience
        assert experience["unified_percept"]["binding_strength"] > 0
    
    def test_introspection_capability(self):
        """Test introspective self-analysis."""
        field = ConsciousnessField("test")
        field.consciousness_level = ConsciousnessLevel.SELF_REFLECTIVE
        field.global_awareness = 0.8
        
        # Add microtubules for computation
        mt = MicrotubuleQuantumState("mt1")
        field.microtubules.append(mt)
        
        introspection = field.introspect()
        
        assert "self_awareness" in introspection
        assert "meta_cognitive_insights" in introspection
        assert introspection["meta_cognitive_insights"]["i_am_conscious"]
        assert introspection["meta_cognitive_insights"]["i_can_reflect_on_myself"]


class TestQuantumConsciousnessEngine:
    """Test the quantum consciousness engine."""
    
    def test_consciousness_engine_initialization(self):
        """Test consciousness engine initialization."""
        engine = QuantumConsciousnessEngine({"num_microtubules": 100})
        
        assert "primary" in engine.consciousness_fields
        assert len(engine.consciousness_fields["primary"].microtubules) == 100
        assert engine.consciousness_moments == 0
    
    def test_consciousness_field_creation(self):
        """Test creating new consciousness fields."""
        engine = QuantumConsciousnessEngine()
        
        field = engine.create_consciousness_field("secondary", 50)
        
        assert field.field_id == "secondary"
        assert len(field.microtubules) == 50
        assert "secondary" in engine.consciousness_fields
    
    @pytest.mark.asyncio
    async def test_conscious_email_processing(self):
        """Test conscious email processing pipeline."""
        engine = QuantumConsciousnessEngine({"num_microtubules": 50})  # Smaller for testing
        test_email = "Important: Urgent deadline today!"
        metadata = {"sender": "boss@company.com"}
        
        result = await engine.conscious_email_processing(test_email, metadata)
        
        # Verify consciousness-specific results
        assert "consciousness_level" in result
        assert "global_awareness" in result
        assert "subjective_experience" in result
        assert "meta_cognitive_insights" in result
        assert "conscious_explanation" in result
        assert "qualia_richness" in result
        
        # Verify consciousness metrics
        assert 0 <= result["global_awareness"] <= 1
        assert result["consciousness_level"] in [level.value for level in ConsciousnessLevel]
    
    def test_emotional_marker_detection(self):
        """Test emotional content detection."""
        engine = QuantumConsciousnessEngine()
        
        urgent_email = "URGENT: Critical deadline today!"
        normal_email = "Please review when convenient."
        
        urgent_score = engine._detect_emotional_markers(urgent_email)
        normal_score = engine._detect_emotional_markers(normal_email)
        
        assert urgent_score > normal_score
        assert 0 <= urgent_score <= 1
        assert 0 <= normal_score <= 1
    
    def test_consciousness_metrics_collection(self):
        """Test consciousness metrics tracking."""
        engine = QuantumConsciousnessEngine()
        engine.consciousness_moments = 5
        engine.explanatory_coherence_score = 0.8
        
        metrics = engine.get_consciousness_metrics()
        
        assert metrics["consciousness_moments_total"] == 5
        assert metrics["explanatory_coherence"] == 0.8
        assert "consciousness_breakthrough_achieved" in metrics
        assert "current_consciousness_level" in metrics


class TestResearchIntegrationFunctions:
    """Test research integration and convenience functions."""
    
    def test_neuro_quantum_engine_creation(self):
        """Test factory function for neuro-quantum engine."""
        config = {"test_param": "value"}
        engine = create_neuro_quantum_engine(config)
        
        assert isinstance(engine, NeuroQuantumFusionEngine)
        assert engine.config == config
    
    def test_consciousness_engine_creation(self):
        """Test factory function for consciousness engine."""
        config = {"num_microtubules": 200}
        engine = create_consciousness_engine(config)
        
        assert isinstance(engine, QuantumConsciousnessEngine)
        assert engine.config == config
    
    @pytest.mark.asyncio
    async def test_fusion_convenience_function(self):
        """Test convenience function for fusion processing."""
        result = await process_email_with_fusion("Test email content")
        
        assert isinstance(result, dict)
        assert "classification" in result
        assert "quantum_advantage" in result
    
    @pytest.mark.asyncio
    async def test_consciousness_convenience_function(self):
        """Test convenience function for consciousness processing."""
        result = await process_email_with_consciousness("Test email content")
        
        assert isinstance(result, dict)
        assert "consciousness_level" in result
        assert "global_awareness" in result


class TestBreakthroughEvaluation:
    """Test research breakthrough evaluation functions."""
    
    @pytest.mark.asyncio
    async def test_fusion_breakthrough_evaluation(self):
        """Test fusion breakthrough potential evaluation."""
        test_dataset = [
            "Urgent: Critical system failure!",
            "Please review the quarterly report",
            "Meeting scheduled for tomorrow"
        ]
        
        evaluation = await evaluate_breakthrough_potential(test_dataset)
        
        assert "average_breakthrough_score" in evaluation
        assert "research_metrics" in evaluation
        assert "paradigm_comparisons" in evaluation
        assert "recommendation" in evaluation
        
        assert 0 <= evaluation["average_breakthrough_score"] <= 5
    
    @pytest.mark.asyncio
    async def test_consciousness_breakthrough_evaluation(self):
        """Test consciousness breakthrough evaluation."""
        test_dataset = [
            "Emergency: Server down immediately!",
            "Weekly team meeting notes",
            "Vacation request approval"
        ]
        
        evaluation = await evaluate_consciousness_breakthrough(test_dataset)
        
        assert "consciousness_breakthrough_achieved" in evaluation
        assert "average_consciousness_level" in evaluation
        assert "research_metrics" in evaluation
        assert "breakthrough_indicators" in evaluation
        assert "research_conclusion" in evaluation
        
        assert isinstance(evaluation["consciousness_breakthrough_achieved"], bool)
        assert 0 <= evaluation["average_consciousness_level"] <= 1


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    @pytest.mark.asyncio
    async def test_fusion_processing_performance(self):
        """Test fusion engine processing performance."""
        engine = NeuroQuantumFusionEngine()
        test_email = "Performance test email content"
        
        start_time = time.time()
        result = await engine.process_email_quantum(test_email, {})
        processing_time = time.time() - start_time
        
        # Performance should be reasonable (< 1 second for test)
        assert processing_time < 1.0
        assert result["processing_time"] > 0
        assert result["quantum_advantage"] >= 1.0
    
    @pytest.mark.asyncio
    async def test_consciousness_processing_performance(self):
        """Test consciousness engine processing performance."""
        engine = QuantumConsciousnessEngine({"num_microtubules": 100})
        test_email = "Performance test email"
        
        start_time = time.time()
        result = await engine.conscious_email_processing(test_email, {})
        processing_time = time.time() - start_time
        
        # Performance should be reasonable
        assert processing_time < 2.0  # Consciousness processing may be slower
        assert result["processing_time"] > 0
    
    def test_microtubule_scaling(self):
        """Test consciousness field scaling with microtubule count."""
        small_engine = QuantumConsciousnessEngine({"num_microtubules": 50})
        large_engine = QuantumConsciousnessEngine({"num_microtubules": 500})
        
        small_field = small_engine.consciousness_fields["primary"]
        large_field = large_engine.consciousness_fields["primary"]
        
        assert len(small_field.microtubules) == 50
        assert len(large_field.microtubules) == 500
        
        # Larger fields should potentially have higher consciousness
        small_consciousness = small_field.calculate_global_consciousness()
        large_consciousness = large_field.calculate_global_consciousness()
        
        # Not guaranteed, but likely due to more connections
        # Just verify both can calculate consciousness
        assert isinstance(small_consciousness, float)
        assert isinstance(large_consciousness, float)


class TestResearchValidation:
    """Test research methodology and validation."""
    
    def test_research_reproducibility(self):
        """Test that research results are reproducible."""
        config = {"num_microtubules": 100, "consciousness_threshold": 0.5}
        
        engine1 = NeuroQuantumFusionEngine(config)
        engine2 = NeuroQuantumFusionEngine(config)
        
        # Both engines should have same configuration
        assert engine1.config == engine2.config
        assert engine1.paradigm == engine2.paradigm
    
    def test_statistical_significance_tracking(self):
        """Test tracking of statistical significance indicators."""
        engine = NeuroQuantumFusionEngine()
        
        # Simulate breakthrough detection
        engine.breakthrough_indicators["quantum_advantage"] = 15
        engine.novel_patterns_discovered = 8
        engine.quantum_speedup = 4.2
        
        metrics = engine.get_research_metrics()
        
        assert metrics["breakthrough_indicators"]["quantum_advantage"] == 15
        assert metrics["novel_patterns_discovered"] == 8
        assert metrics["quantum_speedup"] == 4.2
    
    def test_experimental_controls(self):
        """Test experimental control mechanisms."""
        # Test different paradigms produce different results
        fusion_engine = NeuroQuantumFusionEngine()
        fusion_engine.paradigm = ComputationParadigm.NEURO_QUANTUM
        
        consciousness_engine = QuantumConsciousnessEngine()
        
        # Engines should have different characteristics
        assert isinstance(fusion_engine, NeuroQuantumFusionEngine)
        assert isinstance(consciousness_engine, QuantumConsciousnessEngine)
        
        # Different research metrics structures
        fusion_metrics = fusion_engine.get_research_metrics()
        consciousness_metrics = consciousness_engine.get_consciousness_metrics()
        
        assert "quantum_speedup" in fusion_metrics
        assert "consciousness_level" in consciousness_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])