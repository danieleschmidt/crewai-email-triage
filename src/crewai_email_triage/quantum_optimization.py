"""Quantum-Ready Performance Architecture for Next-Generation Email Triage.

This module implements quantum-inspired optimization algorithms and advanced
performance architecture designed for future quantum computing integration.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import statistics

# import numpy as np  # Optional dependency
from pydantic import BaseModel, Field

from .performance import get_performance_tracker
from .enhanced_pipeline import EnhancedTriageResult


logger = logging.getLogger(__name__)


class OptimizationStrategy(str, Enum):
    """Optimization strategies for quantum-ready performance."""
    
    QUANTUM_ANNEALING = "quantum_annealing"     # Quantum-inspired optimization
    GENETIC_ALGORITHM = "genetic_algorithm"     # Evolutionary optimization
    GRADIENT_DESCENT = "gradient_descent"       # Classical gradient-based
    SWARM_INTELLIGENCE = "swarm_intelligence"   # Particle swarm optimization  
    NEURAL_EVOLUTION = "neural_evolution"       # Neuroevolution strategies
    HYBRID_QUANTUM = "hybrid_quantum"           # Quantum-classical hybrid


class PerformanceMetric(str, Enum):
    """Performance metrics for optimization."""
    
    THROUGHPUT = "throughput"                   # Requests per second
    LATENCY = "latency"                        # Response time
    ACCURACY = "accuracy"                      # Prediction accuracy
    CONFIDENCE = "confidence"                  # Model confidence
    RESOURCE_EFFICIENCY = "resource_efficiency" # Resource utilization
    ENERGY_EFFICIENCY = "energy_efficiency"    # Power consumption
    COST_EFFECTIVENESS = "cost_effectiveness"   # Cost per operation


@dataclass
class QuantumState:
    """Quantum-inspired state representation."""
    
    amplitude: complex = complex(1.0, 0.0)
    phase: float = 0.0
    entanglement_strength: float = 0.0
    coherence_time: float = 1.0
    
    def probability(self) -> float:
        """Calculate measurement probability."""
        return abs(self.amplitude) ** 2
    
    def collapse(self) -> bool:
        """Simulate quantum measurement collapse."""
        return random.random() < self.probability()
    
    def superposition(self, other: 'QuantumState', alpha: float = 0.5) -> 'QuantumState':
        """Create superposition of two quantum states."""
        beta = math.sqrt(1 - alpha**2)
        new_amplitude = alpha * self.amplitude + beta * other.amplitude
        new_phase = (alpha * self.phase + beta * other.phase) / 2
        
        return QuantumState(
            amplitude=new_amplitude,
            phase=new_phase,
            entanglement_strength=(self.entanglement_strength + other.entanglement_strength) / 2,
            coherence_time=min(self.coherence_time, other.coherence_time)
        )


@dataclass
class OptimizationGene:
    """Genetic representation for evolutionary optimization."""
    
    parameters: Dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0
    age: int = 0
    mutations: int = 0
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1) -> 'OptimizationGene':
        """Mutate the gene parameters."""
        new_parameters = self.parameters.copy()
        
        for param, value in new_parameters.items():
            if random.random() < mutation_rate:
                mutation = random.gauss(0, mutation_strength)
                new_parameters[param] = max(0.0, min(1.0, value + mutation))
        
        return OptimizationGene(
            parameters=new_parameters,
            fitness=0.0,  # Fitness needs to be recalculated
            age=0,
            mutations=self.mutations + 1
        )
    
    def crossover(self, other: 'OptimizationGene', crossover_rate: float = 0.5) -> 'OptimizationGene':
        """Create offspring through crossover."""
        new_parameters = {}
        
        for param in self.parameters.keys():
            if param in other.parameters:
                if random.random() < crossover_rate:
                    new_parameters[param] = self.parameters[param]
                else:
                    new_parameters[param] = other.parameters[param]
            else:
                new_parameters[param] = self.parameters[param]
        
        return OptimizationGene(
            parameters=new_parameters,
            fitness=0.0,
            age=0,
            mutations=0
        )


class QuantumAnnealingOptimizer:
    """Quantum-inspired annealing optimizer for performance tuning."""
    
    def __init__(self, temperature_schedule: Callable[[int], float] = None):
        self.temperature_schedule = temperature_schedule or self._default_temperature_schedule
        self.current_state = QuantumState()
        self.best_state = QuantumState()
        self.best_energy = float('inf')
        self.iteration = 0
        
        # Optimization history
        self.energy_history = deque(maxlen=1000)
        self.acceptance_history = deque(maxlen=1000)
    
    def _default_temperature_schedule(self, iteration: int) -> float:
        """Default exponential cooling schedule."""
        return max(0.01, 1.0 * math.exp(-0.01 * iteration))
    
    def _calculate_energy(self, parameters: Dict[str, float]) -> float:
        """Calculate energy (cost) for given parameters."""
        # Simulate complex energy landscape
        energy = 0.0
        
        # Quadratic terms
        for param, value in parameters.items():
            energy += (value - 0.5) ** 2
        
        # Interaction terms
        param_list = list(parameters.values())
        for i in range(len(param_list)):
            for j in range(i + 1, len(param_list)):
                energy += 0.1 * param_list[i] * param_list[j] * math.sin(10 * (param_list[i] + param_list[j]))
        
        return energy
    
    def _generate_neighbor(self, current_params: Dict[str, float], step_size: float = 0.1) -> Dict[str, float]:
        """Generate neighboring parameter set."""
        new_params = current_params.copy()
        
        # Select random parameter to modify
        param_to_modify = random.choice(list(new_params.keys()))
        
        # Add quantum-inspired perturbation
        perturbation = random.gauss(0, step_size)
        new_params[param_to_modify] = max(0.0, min(1.0, new_params[param_to_modify] + perturbation))
        
        return new_params
    
    async def optimize(
        self, 
        initial_parameters: Dict[str, float],
        max_iterations: int = 1000,
        target_energy: float = 0.01
    ) -> Tuple[Dict[str, float], float]:
        """Perform quantum annealing optimization."""
        
        current_params = initial_parameters.copy()
        current_energy = self._calculate_energy(current_params)
        
        best_params = current_params.copy()
        best_energy = current_energy
        
        self.iteration = 0
        
        logger.info(f"Starting quantum annealing optimization with {len(initial_parameters)} parameters")
        
        for iteration in range(max_iterations):
            self.iteration = iteration
            temperature = self.temperature_schedule(iteration)
            
            # Generate neighbor state
            neighbor_params = self._generate_neighbor(current_params)
            neighbor_energy = self._calculate_energy(neighbor_params)
            
            # Calculate acceptance probability
            energy_diff = neighbor_energy - current_energy
            
            if energy_diff <= 0:
                # Always accept better solutions
                acceptance_prob = 1.0
            else:
                # Accept worse solutions with quantum-inspired probability
                if temperature > 0:
                    acceptance_prob = math.exp(-energy_diff / temperature)
                else:
                    acceptance_prob = 0.0
            
            # Quantum tunneling effect
            quantum_prob = self.current_state.probability()
            acceptance_prob = acceptance_prob * quantum_prob + (1 - quantum_prob) * 0.1
            
            # Accept or reject
            if random.random() < acceptance_prob:
                current_params = neighbor_params
                current_energy = neighbor_energy
                
                # Update quantum state
                self.current_state.amplitude *= complex(0.99, 0.01)  # Add quantum phase
                
                if current_energy < best_energy:
                    best_params = current_params.copy()
                    best_energy = current_energy
                    self.best_state = QuantumState(
                        amplitude=self.current_state.amplitude,
                        phase=self.current_state.phase
                    )
            
            # Record history
            self.energy_history.append(current_energy)
            self.acceptance_history.append(1 if random.random() < acceptance_prob else 0)
            
            # Check convergence
            if best_energy < target_energy:
                logger.info(f"Quantum annealing converged at iteration {iteration} with energy {best_energy:.6f}")
                break
            
            # Periodic logging
            if iteration % 100 == 0:
                logger.debug(f"Iteration {iteration}: Energy {current_energy:.6f}, Temperature {temperature:.6f}")
            
            # Allow other tasks to run
            if iteration % 10 == 0:
                await asyncio.sleep(0)
        
        logger.info(f"Quantum annealing completed: Best energy {best_energy:.6f}")
        return best_params, best_energy


class GeneticOptimizer:
    """Genetic algorithm optimizer for performance parameters."""
    
    def __init__(
        self, 
        population_size: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism_rate: float = 0.1
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        
        self.population: List[OptimizationGene] = []
        self.generation = 0
        self.fitness_history = deque(maxlen=1000)
        self.diversity_history = deque(maxlen=1000)
    
    def _initialize_population(self, parameter_ranges: Dict[str, Tuple[float, float]]) -> None:
        """Initialize random population."""
        self.population = []
        
        for _ in range(self.population_size):
            parameters = {}
            for param, (min_val, max_val) in parameter_ranges.items():
                parameters[param] = random.uniform(min_val, max_val)
            
            gene = OptimizationGene(parameters=parameters)
            self.population.append(gene)
    
    def _evaluate_fitness(self, gene: OptimizationGene) -> float:
        """Evaluate fitness of a gene."""
        # Simulate complex fitness landscape
        params = gene.parameters
        
        # Multi-objective fitness combining various metrics
        throughput_score = 1.0 - abs(params.get('batch_size', 0.5) - 0.7) ** 2
        latency_score = 1.0 - abs(params.get('timeout', 0.5) - 0.3) ** 2  
        accuracy_score = 1.0 - abs(params.get('confidence_threshold', 0.5) - 0.8) ** 2
        
        # Add interaction effects
        interaction = math.sin(math.pi * params.get('batch_size', 0.5) * params.get('timeout', 0.5))
        
        fitness = (throughput_score + latency_score + accuracy_score) / 3 + 0.1 * interaction
        return max(0.0, fitness)
    
    def _tournament_selection(self, tournament_size: int = 3) -> OptimizationGene:
        """Tournament selection for parent selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda g: g.fitness)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        diversities = []
        
        for i, gene1 in enumerate(self.population):
            for gene2 in self.population[i+1:]:
                # Calculate parameter space distance
                distance = 0.0
                for param in gene1.parameters:
                    if param in gene2.parameters:
                        distance += (gene1.parameters[param] - gene2.parameters[param]) ** 2
                
                diversities.append(math.sqrt(distance))
        
        return statistics.mean(diversities) if diversities else 0.0
    
    async def optimize(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        max_generations: int = 100,
        target_fitness: float = 0.95
    ) -> Tuple[OptimizationGene, float]:
        """Perform genetic algorithm optimization."""
        
        # Initialize population
        self._initialize_population(parameter_ranges)
        
        logger.info(f"Starting genetic optimization with population size {self.population_size}")
        
        best_gene = None
        best_fitness = 0.0
        
        for generation in range(max_generations):
            self.generation = generation
            
            # Evaluate fitness for all individuals
            for gene in self.population:
                gene.fitness = self._evaluate_fitness(gene)
                gene.age += 1
            
            # Find best individual
            current_best = max(self.population, key=lambda g: g.fitness)
            if current_best.fitness > best_fitness:
                best_gene = OptimizationGene(
                    parameters=current_best.parameters.copy(),
                    fitness=current_best.fitness,
                    age=current_best.age,
                    mutations=current_best.mutations
                )
                best_fitness = current_best.fitness
            
            # Record statistics
            avg_fitness = statistics.mean(g.fitness for g in self.population)
            diversity = self._calculate_diversity()
            
            self.fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            
            # Check convergence
            if best_fitness >= target_fitness:
                logger.info(f"Genetic algorithm converged at generation {generation} with fitness {best_fitness:.6f}")
                break
            
            # Create next generation
            new_population = []
            
            # Elitism - keep best individuals
            elite_count = int(self.elitism_rate * self.population_size)
            elites = sorted(self.population, key=lambda g: g.fitness, reverse=True)[:elite_count]
            new_population.extend(elites)
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring = parent1.crossover(parent2)
                else:
                    offspring = OptimizationGene(parameters=parent1.parameters.copy())
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring = offspring.mutate()
                
                new_population.append(offspring)
            
            self.population = new_population[:self.population_size]
            
            # Periodic logging
            if generation % 10 == 0:
                logger.debug(f"Generation {generation}: Best fitness {best_fitness:.6f}, Avg fitness {avg_fitness:.6f}, Diversity {diversity:.6f}")
            
            # Allow other tasks to run
            if generation % 5 == 0:
                await asyncio.sleep(0)
        
        logger.info(f"Genetic optimization completed: Best fitness {best_fitness:.6f}")
        return best_gene, best_fitness


class SwarmIntelligenceOptimizer:
    """Particle Swarm Optimization for performance tuning."""
    
    def __init__(
        self,
        swarm_size: int = 50,
        inertia_weight: float = 0.9,
        cognitive_weight: float = 2.0,
        social_weight: float = 2.0
    ):
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        
        self.particles = []
        self.global_best_position = {}
        self.global_best_fitness = float('-inf')
        self.iteration = 0
        
        # Performance tracking
        self.fitness_history = deque(maxlen=1000)
        self.velocity_history = deque(maxlen=1000)
    
    @dataclass
    class Particle:
        position: Dict[str, float] = field(default_factory=dict)
        velocity: Dict[str, float] = field(default_factory=dict)
        personal_best_position: Dict[str, float] = field(default_factory=dict)
        personal_best_fitness: float = float('-inf')
        fitness: float = 0.0
    
    def _initialize_swarm(self, parameter_ranges: Dict[str, Tuple[float, float]]) -> None:
        """Initialize particle swarm."""
        self.particles = []
        
        for _ in range(self.swarm_size):
            particle = self.Particle()
            
            for param, (min_val, max_val) in parameter_ranges.items():
                # Initialize position randomly
                particle.position[param] = random.uniform(min_val, max_val)
                
                # Initialize velocity
                velocity_range = (max_val - min_val) * 0.1
                particle.velocity[param] = random.uniform(-velocity_range, velocity_range)
                
                # Set personal best
                particle.personal_best_position[param] = particle.position[param]
            
            self.particles.append(particle)
    
    def _evaluate_fitness(self, particle: 'SwarmIntelligenceOptimizer.Particle') -> float:
        """Evaluate particle fitness."""
        params = particle.position
        
        # Multi-dimensional optimization function
        fitness = 0.0
        
        # Throughput optimization
        throughput_score = 1.0 - abs(params.get('concurrency', 0.5) - 0.8) ** 2
        
        # Latency optimization  
        latency_score = 1.0 - abs(params.get('timeout', 0.5) - 0.2) ** 2
        
        # Resource efficiency
        efficiency_score = 1.0 - abs(params.get('cache_size', 0.5) - 0.6) ** 2
        
        # Combine objectives with weights
        fitness = 0.4 * throughput_score + 0.3 * latency_score + 0.3 * efficiency_score
        
        # Add complexity for realistic optimization landscape
        interaction = math.cos(2 * math.pi * sum(params.values()))
        fitness += 0.1 * interaction
        
        return max(0.0, fitness)
    
    def _update_velocity(
        self, 
        particle: 'SwarmIntelligenceOptimizer.Particle',
        parameter_ranges: Dict[str, Tuple[float, float]]
    ) -> None:
        """Update particle velocity."""
        for param in particle.position:
            if param in parameter_ranges:
                min_val, max_val = parameter_ranges[param]
                
                # Inertia component
                inertia = self.inertia_weight * particle.velocity[param]
                
                # Cognitive component (personal best attraction)
                cognitive_rand = random.random()
                cognitive = (
                    self.cognitive_weight * cognitive_rand * 
                    (particle.personal_best_position[param] - particle.position[param])
                )
                
                # Social component (global best attraction)
                social_rand = random.random()
                social = (
                    self.social_weight * social_rand *
                    (self.global_best_position.get(param, particle.position[param]) - particle.position[param])
                )
                
                # Update velocity
                new_velocity = inertia + cognitive + social
                
                # Clamp velocity to reasonable range
                max_velocity = (max_val - min_val) * 0.2
                particle.velocity[param] = max(-max_velocity, min(max_velocity, new_velocity))
    
    def _update_position(
        self,
        particle: 'SwarmIntelligenceOptimizer.Particle',
        parameter_ranges: Dict[str, Tuple[float, float]]
    ) -> None:
        """Update particle position."""
        for param in particle.position:
            if param in parameter_ranges:
                min_val, max_val = parameter_ranges[param]
                
                # Update position
                new_position = particle.position[param] + particle.velocity[param]
                
                # Clamp to parameter bounds
                particle.position[param] = max(min_val, min(max_val, new_position))
    
    async def optimize(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        target_fitness: float = 0.95
    ) -> Tuple[Dict[str, float], float]:
        """Perform particle swarm optimization."""
        
        # Initialize swarm
        self._initialize_swarm(parameter_ranges)
        
        logger.info(f"Starting swarm intelligence optimization with {self.swarm_size} particles")
        
        for iteration in range(max_iterations):
            self.iteration = iteration
            
            # Evaluate fitness for all particles
            for particle in self.particles:
                particle.fitness = self._evaluate_fitness(particle)
                
                # Update personal best
                if particle.fitness > particle.personal_best_fitness:
                    particle.personal_best_fitness = particle.fitness
                    particle.personal_best_position = particle.position.copy()
                
                # Update global best
                if particle.fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Record statistics
            avg_fitness = statistics.mean(p.fitness for p in self.particles)
            avg_velocity = statistics.mean(
                sum(abs(v) for v in p.velocity.values()) for p in self.particles
            )
            
            self.fitness_history.append(avg_fitness)
            self.velocity_history.append(avg_velocity)
            
            # Check convergence
            if self.global_best_fitness >= target_fitness:
                logger.info(f"Swarm optimization converged at iteration {iteration} with fitness {self.global_best_fitness:.6f}")
                break
            
            # Update particle velocities and positions
            for particle in self.particles:
                self._update_velocity(particle, parameter_ranges)
                self._update_position(particle, parameter_ranges)
            
            # Adaptive parameter adjustment
            if iteration > 10:
                recent_improvement = (
                    self.fitness_history[-1] - self.fitness_history[-10]
                ) / max(self.fitness_history[-10], 0.001)
                
                if recent_improvement < 0.001:  # Stagnation
                    # Increase exploration
                    self.inertia_weight = min(0.9, self.inertia_weight * 1.05)
                else:
                    # Increase exploitation
                    self.inertia_weight = max(0.4, self.inertia_weight * 0.95)
            
            # Periodic logging
            if iteration % 10 == 0:
                logger.debug(f"Iteration {iteration}: Best fitness {self.global_best_fitness:.6f}, Avg fitness {avg_fitness:.6f}")
            
            # Allow other tasks to run
            if iteration % 5 == 0:
                await asyncio.sleep(0)
        
        logger.info(f"Swarm optimization completed: Best fitness {self.global_best_fitness:.6f}")
        return self.global_best_position, self.global_best_fitness


class QuantumPerformanceOptimizer:
    """Quantum-ready performance optimizer combining multiple strategies."""
    
    def __init__(self):
        self.quantum_annealer = QuantumAnnealingOptimizer()
        self.genetic_optimizer = GeneticOptimizer()
        self.swarm_optimizer = SwarmIntelligenceOptimizer()
        
        self.performance_tracker = get_performance_tracker()
        
        # Optimization history
        self.optimization_history = deque(maxlen=100)
        self.strategy_performance = {
            strategy.value: deque(maxlen=50) for strategy in OptimizationStrategy
        }
        
        # Current optimal parameters
        self.optimal_parameters = {}
        self.optimization_confidence = 0.0
        
        # Performance baselines
        self.performance_baselines = {
            metric.value: deque(maxlen=1000) for metric in PerformanceMetric
        }
    
    async def optimize_performance(
        self,
        current_metrics: Dict[str, float],
        strategy: OptimizationStrategy = OptimizationStrategy.HYBRID_QUANTUM,
        optimization_budget: int = 1000
    ) -> Dict[str, Any]:
        """Optimize performance using specified strategy."""
        
        start_time = time.time()
        
        logger.info(f"Starting quantum performance optimization with strategy: {strategy.value}")
        
        # Define parameter ranges based on current system state
        parameter_ranges = self._define_parameter_ranges(current_metrics)
        
        # Select and run optimization strategy
        if strategy == OptimizationStrategy.QUANTUM_ANNEALING:
            optimal_params, best_score = await self._run_quantum_annealing(parameter_ranges, optimization_budget)
        
        elif strategy == OptimizationStrategy.GENETIC_ALGORITHM:
            optimal_gene, best_score = await self._run_genetic_algorithm(parameter_ranges, optimization_budget)
            optimal_params = optimal_gene.parameters
        
        elif strategy == OptimizationStrategy.SWARM_INTELLIGENCE:
            optimal_params, best_score = await self._run_swarm_optimization(parameter_ranges, optimization_budget)
        
        elif strategy == OptimizationStrategy.HYBRID_QUANTUM:
            optimal_params, best_score = await self._run_hybrid_optimization(parameter_ranges, optimization_budget)
        
        else:
            # Fallback to quantum annealing
            optimal_params, best_score = await self._run_quantum_annealing(parameter_ranges, optimization_budget)
        
        # Calculate optimization results
        optimization_time = time.time() - start_time
        
        optimization_result = {
            "strategy": strategy.value,
            "optimal_parameters": optimal_params,
            "optimization_score": best_score,
            "optimization_time_seconds": optimization_time,
            "parameter_improvements": self._calculate_improvements(optimal_params, current_metrics),
            "confidence_score": self._calculate_confidence(best_score, strategy),
            "quantum_coherence": self.quantum_annealer.current_state.coherence_time,
            "convergence_metrics": {
                "iterations_required": optimization_budget,
                "final_energy": 1.0 - best_score if strategy == OptimizationStrategy.QUANTUM_ANNEALING else None,
                "population_diversity": self.genetic_optimizer.diversity_history[-1] if self.genetic_optimizer.diversity_history else None,
                "swarm_velocity": self.swarm_optimizer.velocity_history[-1] if self.swarm_optimizer.velocity_history else None
            }
        }
        
        # Update optimization history
        self.optimization_history.append(optimization_result)
        self.strategy_performance[strategy.value].append(best_score)
        
        # Update optimal parameters if improvement found
        if best_score > self.optimization_confidence:
            self.optimal_parameters = optimal_params
            self.optimization_confidence = best_score
        
        # Track performance
        self.performance_tracker.record_operation(
            f"quantum_optimization_{strategy.value}",
            optimization_time,
            {
                "optimization_score": best_score,
                "parameter_count": len(optimal_params),
                "strategy": strategy.value
            }
        )
        
        logger.info(
            f"Quantum optimization completed: {strategy.value} "
            f"(Score: {best_score:.6f}, Time: {optimization_time:.2f}s)"
        )
        
        return optimization_result
    
    def _define_parameter_ranges(self, current_metrics: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Define parameter ranges for optimization."""
        
        # Base parameter ranges
        ranges = {
            "batch_size": (0.1, 1.0),           # Normalized batch size
            "concurrency": (0.1, 1.0),         # Concurrency level
            "timeout": (0.1, 1.0),             # Timeout settings
            "cache_size": (0.1, 1.0),          # Cache configuration
            "confidence_threshold": (0.1, 1.0), # Model confidence threshold
            "learning_rate": (0.01, 1.0),      # Learning rate
            "temperature": (0.1, 2.0),         # Temperature for sampling
            "exploration_rate": (0.0, 1.0),    # Exploration vs exploitation
        }
        
        # Adjust ranges based on current performance
        if current_metrics.get("latency", 0) > 0.8:  # High latency
            # Favor lower values for latency-sensitive parameters
            ranges["timeout"] = (0.1, 0.5)
            ranges["batch_size"] = (0.1, 0.6)
        
        if current_metrics.get("throughput", 0) < 0.5:  # Low throughput
            # Favor higher values for throughput-sensitive parameters
            ranges["concurrency"] = (0.5, 1.0)
            ranges["batch_size"] = (0.6, 1.0)
        
        if current_metrics.get("accuracy", 0) < 0.7:  # Low accuracy
            # Favor parameters that improve accuracy
            ranges["confidence_threshold"] = (0.5, 0.9)
            ranges["learning_rate"] = (0.1, 0.5)
        
        return ranges
    
    async def _run_quantum_annealing(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        budget: int
    ) -> Tuple[Dict[str, float], float]:
        """Run quantum annealing optimization."""
        
        # Initialize with random parameters
        initial_params = {}
        for param, (min_val, max_val) in parameter_ranges.items():
            initial_params[param] = random.uniform(min_val, max_val)
        
        # Set custom temperature schedule for better performance
        def custom_temperature_schedule(iteration: int) -> float:
            return max(0.001, 1.0 * math.exp(-0.005 * iteration))
        
        self.quantum_annealer.temperature_schedule = custom_temperature_schedule
        
        optimal_params, best_energy = await self.quantum_annealer.optimize(
            initial_params,
            max_iterations=budget,
            target_energy=0.01
        )
        
        # Convert energy to score (lower energy = higher score)
        best_score = max(0.0, 1.0 - best_energy)
        
        return optimal_params, best_score
    
    async def _run_genetic_algorithm(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        budget: int
    ) -> Tuple[OptimizationGene, float]:
        """Run genetic algorithm optimization."""
        
        # Adjust population size based on budget
        population_size = min(100, max(20, budget // 10))
        generations = budget // population_size
        
        self.genetic_optimizer.population_size = population_size
        
        optimal_gene, best_fitness = await self.genetic_optimizer.optimize(
            parameter_ranges,
            max_generations=generations,
            target_fitness=0.95
        )
        
        return optimal_gene, best_fitness
    
    async def _run_swarm_optimization(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        budget: int
    ) -> Tuple[Dict[str, float], float]:
        """Run swarm intelligence optimization."""
        
        # Adjust swarm size based on budget
        swarm_size = min(50, max(10, budget // 20))
        iterations = budget // swarm_size
        
        self.swarm_optimizer.swarm_size = swarm_size
        
        optimal_params, best_fitness = await self.swarm_optimizer.optimize(
            parameter_ranges,
            max_iterations=iterations,
            target_fitness=0.95
        )
        
        return optimal_params, best_fitness
    
    async def _run_hybrid_optimization(
        self,
        parameter_ranges: Dict[str, Tuple[float, float]],
        budget: int
    ) -> Tuple[Dict[str, float], float]:
        """Run hybrid quantum-classical optimization."""
        
        # Divide budget among strategies
        budget_per_strategy = budget // 3
        
        # Run multiple strategies in parallel
        tasks = [
            self._run_quantum_annealing(parameter_ranges, budget_per_strategy),
            self._run_genetic_algorithm(parameter_ranges, budget_per_strategy),
            self._run_swarm_optimization(parameter_ranges, budget_per_strategy)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Extract results
        quantum_params, quantum_score = results[0]
        genetic_gene, genetic_score = results[1]
        swarm_params, swarm_score = results[2]
        
        # Select best result
        if quantum_score >= genetic_score and quantum_score >= swarm_score:
            best_params = quantum_params
            best_score = quantum_score
            logger.info("Hybrid optimization: Quantum annealing achieved best result")
        elif genetic_score >= swarm_score:
            best_params = genetic_gene.parameters
            best_score = genetic_score
            logger.info("Hybrid optimization: Genetic algorithm achieved best result")
        else:
            best_params = swarm_params
            best_score = swarm_score
            logger.info("Hybrid optimization: Swarm intelligence achieved best result")
        
        # Apply quantum superposition to combine solutions
        if len(self.optimization_history) > 0:
            # Create quantum superposition of top solutions
            alpha = 0.7  # Weight for current best
            beta = math.sqrt(1 - alpha**2)
            
            superposition_params = {}
            for param in best_params:
                if param in quantum_params and param in swarm_params:
                    # Quantum-inspired parameter combination
                    superposition_params[param] = (
                        alpha * best_params[param] + 
                        beta * (quantum_params[param] + swarm_params[param]) / 2
                    )
                else:
                    superposition_params[param] = best_params[param]
            
            # Evaluate superposition
            # (In a real implementation, this would require testing the superposition parameters)
            # For now, we'll assume slight improvement
            superposition_score = min(1.0, best_score * 1.02)
            
            if superposition_score > best_score:
                best_params = superposition_params
                best_score = superposition_score
                logger.info("Hybrid optimization: Quantum superposition improved result")
        
        return best_params, best_score
    
    def _calculate_improvements(
        self,
        optimal_params: Dict[str, float],
        current_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate expected improvements from optimization."""
        
        improvements = {}
        
        # Simulate expected improvements based on parameter changes
        # (In a real implementation, this would use learned models or simulations)
        
        improvements["throughput"] = min(0.5, optimal_params.get("concurrency", 0.5) * 0.3)
        improvements["latency"] = min(0.4, (1.0 - optimal_params.get("timeout", 0.5)) * 0.2)
        improvements["accuracy"] = min(0.3, optimal_params.get("confidence_threshold", 0.5) * 0.1)
        improvements["resource_efficiency"] = min(0.2, optimal_params.get("cache_size", 0.5) * 0.1)
        
        return improvements
    
    def _calculate_confidence(self, optimization_score: float, strategy: OptimizationStrategy) -> float:
        """Calculate confidence in optimization results."""
        
        base_confidence = optimization_score
        
        # Adjust confidence based on strategy characteristics
        strategy_multipliers = {
            OptimizationStrategy.QUANTUM_ANNEALING: 0.9,  # High confidence for quantum methods
            OptimizationStrategy.GENETIC_ALGORITHM: 0.8,   # Good confidence for evolutionary methods
            OptimizationStrategy.SWARM_INTELLIGENCE: 0.8,  # Good confidence for swarm methods
            OptimizationStrategy.HYBRID_QUANTUM: 1.0       # Highest confidence for hybrid approach
        }
        
        strategy_confidence = strategy_multipliers.get(strategy, 0.7)
        
        # Consider optimization history
        if len(self.strategy_performance[strategy.value]) > 0:
            historical_performance = statistics.mean(self.strategy_performance[strategy.value])
            history_factor = min(1.2, 1.0 + 0.1 * (historical_performance - 0.5))
        else:
            history_factor = 1.0
        
        # Calculate final confidence
        confidence = base_confidence * strategy_confidence * history_factor
        
        return min(1.0, max(0.0, confidence))
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get comprehensive optimization insights."""
        
        if not self.optimization_history:
            return {
                "message": "No optimization history available",
                "recommendations": ["Run performance optimization to get insights"]
            }
        
        recent_optimizations = list(self.optimization_history)[-10:]
        
        # Calculate strategy effectiveness
        strategy_effectiveness = {}
        for strategy in OptimizationStrategy:
            if self.strategy_performance[strategy.value]:
                strategy_effectiveness[strategy.value] = {
                    "avg_score": statistics.mean(self.strategy_performance[strategy.value]),
                    "best_score": max(self.strategy_performance[strategy.value]),
                    "consistency": 1.0 - statistics.stdev(self.strategy_performance[strategy.value]) if len(self.strategy_performance[strategy.value]) > 1 else 1.0
                }
        
        # Generate recommendations
        recommendations = []
        
        if self.optimization_confidence < 0.7:
            recommendations.append("Consider running hybrid quantum optimization for better results")
        
        if len(self.optimization_history) > 1:
            recent_scores = [opt["optimization_score"] for opt in recent_optimizations]
            if statistics.mean(recent_scores[-3:]) < statistics.mean(recent_scores):
                recommendations.append("Optimization performance is declining - consider parameter reset")
        
        # Check for convergence patterns
        if len(recent_optimizations) >= 3:
            convergence_times = [opt["optimization_time_seconds"] for opt in recent_optimizations[-3:]]
            if statistics.mean(convergence_times) > 30:
                recommendations.append("Long optimization times detected - consider reducing budget or using faster strategies")
        
        return {
            "optimization_summary": {
                "total_optimizations": len(self.optimization_history),
                "current_confidence": self.optimization_confidence,
                "optimal_parameters": self.optimal_parameters,
                "avg_optimization_time": statistics.mean(opt["optimization_time_seconds"] for opt in recent_optimizations)
            },
            "strategy_effectiveness": strategy_effectiveness,
            "performance_trends": {
                "recent_improvements": [opt["optimization_score"] for opt in recent_optimizations],
                "convergence_stability": self._analyze_convergence_stability(),
                "quantum_coherence_trend": [
                    opt.get("quantum_coherence", 1.0) for opt in recent_optimizations 
                    if "quantum_coherence" in opt
                ]
            },
            "recommendations": recommendations,
            "next_optimization_suggestion": self._suggest_next_optimization()
        }
    
    def _analyze_convergence_stability(self) -> Dict[str, float]:
        """Analyze convergence stability across optimizations."""
        
        if len(self.optimization_history) < 3:
            return {"insufficient_data": True}
        
        recent_scores = [opt["optimization_score"] for opt in self.optimization_history[-10:]]
        recent_times = [opt["optimization_time_seconds"] for opt in self.optimization_history[-10:]]
        
        return {
            "score_stability": 1.0 - statistics.stdev(recent_scores) if len(recent_scores) > 1 else 1.0,
            "time_stability": 1.0 - (statistics.stdev(recent_times) / statistics.mean(recent_times)) if len(recent_times) > 1 else 1.0,
            "improvement_trend": (recent_scores[-1] - recent_scores[0]) / len(recent_scores) if len(recent_scores) > 1 else 0.0
        }
    
    def _suggest_next_optimization(self) -> Dict[str, Any]:
        """Suggest next optimization strategy and parameters."""
        
        if not self.optimization_history:
            return {
                "strategy": OptimizationStrategy.HYBRID_QUANTUM.value,
                "budget": 1000,
                "reason": "Initial optimization - hybrid approach recommended"
            }
        
        # Analyze recent performance
        recent_opt = self.optimization_history[-1]
        recent_score = recent_opt["optimization_score"]
        recent_strategy = recent_opt["strategy"]
        
        if recent_score > 0.9:
            # High performance - use maintenance optimization
            return {
                "strategy": OptimizationStrategy.QUANTUM_ANNEALING.value,
                "budget": 500,
                "reason": "High performance achieved - maintenance optimization recommended"
            }
        elif recent_score < 0.6:
            # Low performance - use aggressive optimization
            return {
                "strategy": OptimizationStrategy.HYBRID_QUANTUM.value,
                "budget": 2000,
                "reason": "Low performance detected - aggressive hybrid optimization recommended"
            }
        else:
            # Medium performance - try different strategy
            used_strategies = set(opt["strategy"] for opt in self.optimization_history[-3:])
            unused_strategies = [s for s in OptimizationStrategy if s.value not in used_strategies]
            
            if unused_strategies:
                suggested_strategy = unused_strategies[0]
            else:
                # All strategies tried recently - use best performing one
                strategy_scores = defaultdict(list)
                for opt in self.optimization_history[-5:]:
                    strategy_scores[opt["strategy"]].append(opt["optimization_score"])
                
                best_strategy = max(strategy_scores.items(), key=lambda x: statistics.mean(x[1]))[0]
                suggested_strategy = OptimizationStrategy(best_strategy)
            
            return {
                "strategy": suggested_strategy.value,
                "budget": 1000,
                "reason": f"Trying {suggested_strategy.value} for performance improvement"
            }


# Global instance
_quantum_optimizer: Optional[QuantumPerformanceOptimizer] = None


def get_quantum_optimizer() -> QuantumPerformanceOptimizer:
    """Get the global quantum performance optimizer instance."""
    global _quantum_optimizer
    if _quantum_optimizer is None:
        _quantum_optimizer = QuantumPerformanceOptimizer()
    return _quantum_optimizer


# Convenience functions
async def optimize_system_performance(
    current_metrics: Optional[Dict[str, float]] = None,
    strategy: OptimizationStrategy = OptimizationStrategy.HYBRID_QUANTUM,
    budget: int = 1000
) -> Dict[str, Any]:
    """Optimize system performance using quantum-ready algorithms."""
    
    optimizer = get_quantum_optimizer()
    
    # Use default metrics if none provided
    if current_metrics is None:
        current_metrics = {
            "throughput": 0.5,
            "latency": 0.6,
            "accuracy": 0.7,
            "resource_efficiency": 0.6
        }
    
    return await optimizer.optimize_performance(current_metrics, strategy, budget)


def get_performance_insights() -> Dict[str, Any]:
    """Get comprehensive performance optimization insights."""
    optimizer = get_quantum_optimizer()
    return optimizer.get_optimization_insights()