"""
Python Language System to Improve Probabilistic Reasoning and Uncertainty Management for Artificial Intelligence GOOGLE GEMINI 2.0 FLASH API
================================================================

A comprehensive system for integrating uncertainty into Artificial Intelligence GOOGLE GEMINI 2.0 FLASH API reasoning processes,
including Bayesian inference, conditional probability management,
confidence calibration, and decision-making under uncertainty.


"""

import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp, gammaln
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UncertaintyQuantification:
    """Structure for quantifying uncertainty"""
    epistemic: float  # Uncertainty due to lack of knowledge
    aleatoric: float  # Uncertainty inherent to the system
    total: float      # Total uncertainty
    confidence: float # Confidence level

class BaseUncertaintyModel(ABC):
    """Base class for all uncertainty models"""
    
    @abstractmethod
    def compute_uncertainty(self, *args, **kwargs) -> UncertaintyQuantification:
        """Computes uncertainty for the provided data"""
        pass
    
    @abstractmethod
    def update_beliefs(self, evidence: Any) -> None:
        """Updates beliefs based on new evidence"""
        pass

class ProbabilisticDistribution:
    """Class for representing and manipulating probabilistic distributions"""
    
    def __init__(self, distribution_type: str, parameters: Dict[str, float]):
        self.distribution_type = distribution_type
        self.parameters = parameters
        self._distribution = self._create_distribution()
    
    def _create_distribution(self):
        """Creates the corresponding scipy distribution object"""
        if self.distribution_type == 'normal':
            return stats.norm(loc=self.parameters['mean'], 
                            scale=self.parameters['std'])
        elif self.distribution_type == 'beta':
            return stats.beta(a=self.parameters['alpha'], 
                            b=self.parameters['beta'])
        elif self.distribution_type == 'gamma':
            return stats.gamma(a=self.parameters['shape'], 
                             scale=self.parameters['scale'])
        elif self.distribution_type == 'dirichlet':
            return stats.dirichlet(alpha=self.parameters['alpha'])
        elif self.distribution_type == 'uniform':
            return stats.uniform(loc=self.parameters['low'], 
                               scale=self.parameters['high'] - self.parameters['low'])
        else:
            raise ValueError(f"Distribution type {self.distribution_type} not supported")
    
    def pdf(self, x):
        """Probability density function"""
        return self._distribution.pdf(x)
    
    def logpdf(self, x):
        """Log-probability density function"""
        return self._distribution.logpdf(x)
    
    def cdf(self, x):
        """Cumulative distribution function"""
        return self._distribution.cdf(x)
    
    def sample(self, size: int = 1):
        """Sampling from the distribution"""
        return self._distribution.rvs(size=size)
    
    def mean(self):
        """Mean of the distribution"""
        return self._distribution.mean()
    
    def var(self):
        """Variance of the distribution"""
        return self._distribution.var()
    
    def entropy(self):
        """Entropy of the distribution"""
        return self._distribution.entropy()

class BayesianNetwork:
    """Bayesian network for modeling probabilistic dependencies"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.conditional_distributions = {}
        self.evidence = {}
    
    def add_node(self, name: str, distribution: ProbabilisticDistribution):
        """Adds a node to the network"""
        self.nodes[name] = distribution
        self.edges[name] = []
    
    def add_edge(self, parent: str, child: str):
        """Adds a directed edge parent -> child"""
        if parent not in self.edges:
            self.edges[parent] = []
        self.edges[parent].append(child)
    
    def set_conditional_distribution(self, node: str, parents: List[str], 
                                   conditional_func: Callable):
        """Defines the conditional distribution P(node|parents)"""
        self.conditional_distributions[node] = {
            'parents': parents,
            'function': conditional_func
        }
    
    def set_evidence(self, node: str, value: Any):
        """Sets evidence for a node"""
        self.evidence[node] = value
    
    def forward_sampling(self, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Forward sampling to generate samples from the network"""
        samples = {node: [] for node in self.nodes}
        
        # Topological sort of nodes
        ordered_nodes = self._topological_sort()
        
        for _ in range(num_samples):
            sample = {}
            for node in ordered_nodes:
                if node in self.evidence:
                    sample[node] = self.evidence[node]
                elif node in self.conditional_distributions:
                    parents = self.conditional_distributions[node]['parents']
                    parent_values = [sample[p] for p in parents]
                    func = self.conditional_distributions[node]['function']
                    sample[node] = func(*parent_values)
                else:
                    sample[node] = self.nodes[node].sample()
                
                samples[node].append(sample[node])
        
        return {node: np.array(values) for node, values in samples.items()}
    
    def _topological_sort(self) -> List[str]:
        """Topological sort of nodes"""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node):
            if node in temp_visited:
                raise ValueError("Cycle detected in the Bayesian network")
            if node not in visited:
                temp_visited.add(node)
                for child in self.edges.get(node, []):
                    visit(child)
                temp_visited.remove(node)
                visited.add(node)
                result.insert(0, node)
        
        for node in self.nodes:
            if node not in visited:
                visit(node)
        
        return result

class BayesianInference:
    """Class for Bayesian inference and belief updating"""
    
    def __init__(self):
        self.priors = {}
        self.posteriors = {}
        self.evidence_history = []
        self.belief_history = []
    
    def set_prior(self, parameter: str, distribution: ProbabilisticDistribution):
        """Sets a prior distribution for a parameter"""
        self.priors[parameter] = distribution
        self.posteriors[parameter] = distribution
    
    def bayesian_update(self, parameter: str, likelihood_func: Callable, 
                       evidence: Any) -> ProbabilisticDistribution:
        """Updates Bayesian beliefs based on new evidence"""
        if parameter not in self.posteriors:
            raise ValueError(f"Parameter {parameter} not found")
        
        prior = self.posteriors[parameter]
        
        # Likelihood calculation
        def unnormalized_posterior(x):
            prior_prob = prior.pdf(x)
            likelihood = likelihood_func(x, evidence)
            return prior_prob * likelihood
        
        # Normalization by numerical integration
        from scipy.integrate import quad
        
        # For simplicity, assume an updated normal distribution
        # In a complete implementation, use MCMC or VI
        if prior.distribution_type == 'normal':
            # Conjugate update for normal-normal
            if hasattr(evidence, '__len__'):
                n = len(evidence)
                sample_mean = np.mean(evidence)
                sample_var = np.var(evidence)
                
                # Prior parameters
                prior_mean = prior.parameters['mean']
                prior_var = prior.parameters['std']**2
                
                # Bayesian update
                posterior_var = 1 / (1/prior_var + n/sample_var)
                posterior_mean = posterior_var * (prior_mean/prior_var + n*sample_mean/sample_var)
                
                posterior = ProbabilisticDistribution(
                    'normal', 
                    {'mean': posterior_mean, 'std': np.sqrt(posterior_var)}
                )
            else:
                # Single observation
                observation = evidence
                prior_mean = prior.parameters['mean']
                prior_var = prior.parameters['std']**2
                obs_var = 1.0  # Assumed
                
                posterior_var = 1 / (1/prior_var + 1/obs_var)
                posterior_mean = posterior_var * (prior_mean/prior_var + observation/obs_var)
                
                posterior = ProbabilisticDistribution(
                    'normal',
                    {'mean': posterior_mean, 'std': np.sqrt(posterior_var)}
                )
        else:
            # For other distributions, use MCMC or approximation
            posterior = self._approximate_posterior(prior, likelihood_func, evidence)
        
        self.posteriors[parameter] = posterior
        self.evidence_history.append(evidence)
        self.belief_history.append(posterior)
        
        return posterior
    
    def _approximate_posterior(self, prior: ProbabilisticDistribution, 
                             likelihood_func: Callable, evidence: Any) -> ProbabilisticDistribution:
        """Posterior distribution approximation"""
        # Simplified implementation - in practice, use MCMC/VI
        samples = prior.sample(10000)
        log_weights = []
        
        for sample in samples:
            log_prior = prior.logpdf(sample)
            log_likelihood = np.log(likelihood_func(sample, evidence))
            log_weights.append(log_prior + log_likelihood)
        
        log_weights = np.array(log_weights)
        weights = np.exp(log_weights - logsumexp(log_weights))
        
        # Approximation by a normal distribution
        weighted_mean = np.average(samples, weights=weights)
        weighted_var = np.average((samples - weighted_mean)**2, weights=weights)
        
        return ProbabilisticDistribution(
            'normal',
            {'mean': weighted_mean, 'std': np.sqrt(weighted_var)}
        )
    
    def compute_marginal_likelihood(self, parameter: str, likelihood_func: Callable, 
                                  evidence: Any) -> float:
        """Computes the marginal likelihood (evidence)"""
        prior = self.priors[parameter]
        
        def integrand(x):
            return prior.pdf(x) * likelihood_func(x, evidence)
        
        from scipy.integrate import quad
        result, _ = quad(integrand, -10, 10)  # Adjustable limits
        return result
    
    def compute_bayes_factor(self, param1: str, param2: str, likelihood_func: Callable, 
                           evidence: Any) -> float:
        """Computes the Bayes factor between two models"""
        ml1 = self.compute_marginal_likelihood(param1, likelihood_func, evidence)
        ml2 = self.compute_marginal_likelihood(param2, likelihood_func, evidence)
        return ml1 / ml2

class ConditionalProbabilityManager:
    """Manager for complex conditional probabilities"""
    
    def __init__(self):
        self.conditional_tables = {}
        self.independence_assumptions = {}
        self.causal_graph = {}
    
    def add_conditional_probability(self, event: str, given: List[str], 
                                  probability_table: Dict[Tuple, float]):
        """Adds a conditional probability table P(event|given)"""
        self.conditional_tables[event] = {
            'given': given,
            'table': probability_table
        }
    
    def compute_conditional_probability(self, event: str, given_values: Dict[str, Any]) -> float:
        """Computes P(event|given_values)"""
        if event not in self.conditional_tables:
            raise ValueError(f"Conditional probability for {event} not defined")
        
        table_info = self.conditional_tables[event]
        given_vars = table_info['given']
        table = table_info['table']
        
        # Build the key for the table
        key = tuple(given_values[var] for var in given_vars)
        
        if key in table:
            return table[key]
        else:
            raise ValueError(f"Combination of values {key} not found in the table")
    
    def compute_joint_probability(self, events: Dict[str, Any]) -> float:
        """Computes the joint probability using the chain rule"""
        # P(A,B,C) = P(A) * P(B|A) * P(C|A,B)
        probability = 1.0
        
        # Topological order of events
        ordered_events = list(events.keys())  # Simplification
        
        for i, event in enumerate(ordered_events):
            if i == 0:
                # Marginal probability of the first event
                prob = self._get_marginal_probability(event, events[event])
            else:
                # Conditional probability
                given_events = {e: events[e] for e in ordered_events[:i]}
                prob = self.compute_conditional_probability(event, given_events)
            
            probability *= prob
        
        return probability
    
    def _get_marginal_probability(self, event: str, value: Any) -> float:
        """Gets the marginal probability of an event"""
        # Simplified implementation - in practice, compute from tables
        return 0.5  # Placeholder
    
    def check_conditional_independence(self, event_a: str, event_b: str, 
                                     given: List[str]) -> bool:
        """Checks conditional independence P(A|C) = P(A|B,C)"""
        # Implementation based on causal graph structure
        return self._d_separation(event_a, event_b, given)
    
    def _d_separation(self, node_a: str, node_b: str, conditioning_set: List[str]) -> bool:
        """D-separation algorithm to check conditional independence"""
        # Simplified implementation of the d-separation test
        # In a complete implementation, implement the full algorithm
        return False  # Placeholder

class ConfidenceCalibrator:
    """Class for confidence calibration and epistemic uncertainty quantification"""
    
    def __init__(self):
        self.calibration_data = []
        self.calibration_function = None
        self.reliability_diagram_data = None
    
    def add_prediction(self, confidence: float, correct: bool):
        """Adds a prediction with its confidence level and truthfulness"""
        self.calibration_data.append((confidence, correct))
    
    def compute_calibration_error(self, num_bins: int = 10) -> Tuple[float, Dict]:
        """Computes the calibration error (ECE - Expected Calibration Error)"""
        if not self.calibration_data:
            return 0.0, {}
        
        confidences = np.array([c for c, _ in self.calibration_data])
        corrects = np.array([c for _, c in self.calibration_data])
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_data = {}
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = corrects[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_data[f'bin_{bin_lower:.1f}_{bin_upper:.1f}'] = {
                    'confidence': avg_confidence_in_bin,
                    'accuracy': accuracy_in_bin,
                    'proportion': prop_in_bin
                }
        
        self.reliability_diagram_data = bin_data
        return ece, bin_data
    
    def temperature_scaling(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Calibration by temperature scaling"""
        from scipy.optimize import minimize_scalar
        
        def loss(temperature):
            scaled_logits = logits / temperature
            # Log-likelihood calculation
            exp_logits = np.exp(scaled_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Cross-entropy loss
            return -np.mean(np.log(probs[np.arange(len(labels)), labels]))
        
        result = minimize_scalar(loss, bounds=(0.1, 10.0), method='bounded')
        optimal_temperature = result.x
        
        return optimal_temperature
    
    def platt_scaling(self, scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Calibration by Platt scaling (logistic regression)"""
        from sklearn.linear_model import LogisticRegression
        
        lr = LogisticRegression()
        lr.fit(scores.reshape(-1, 1), labels)
        
        # Parameters A and B for P(y=1|f) = 1/(1 + exp(A*f + B))
        A = lr.coef_[0][0]
        B = lr.intercept_[0]
        
        return A, B
    
    def compute_epistemic_uncertainty(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Computes epistemic uncertainty from multiple predictions"""
        # Variance of predictions = epistemic uncertainty
        predictions_array = np.array(predictions)
        epistemic_uncertainty = np.var(predictions_array, axis=0)
        return epistemic_uncertainty
    
    def compute_aleatoric_uncertainty(self, prediction_variances: List[np.ndarray]) -> np.ndarray:
        """Computes average aleatoric uncertainty"""
        # Mean of variances = aleatoric uncertainty
        variances_array = np.array(prediction_variances)
        aleatoric_uncertainty = np.mean(variances_array, axis=0)
        return aleatoric_uncertainty
    
    def decompose_uncertainty(self, predictions: List[np.ndarray], 
                            variances: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Decomposes total uncertainty into epistemic and aleatoric"""
        epistemic = self.compute_epistemic_uncertainty(predictions)
        aleatoric = self.compute_aleatoric_uncertainty(variances)
        total = epistemic + aleatoric
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total
        }

class GameTheoryDecisionMaker:
    """Decision-making under uncertainty with game theory"""
    
    def __init__(self):
        self.payoff_matrices = {}
        self.strategies = {}
        self.uncertainty_models = {}
    
    def add_game(self, game_name: str, players: List[str], 
                strategies: Dict[str, List[str]], 
                payoff_matrix: np.ndarray):
        """Adds a game to the database"""
        self.strategies[game_name] = strategies
        self.payoff_matrices[game_name] = payoff_matrix
    
    def nash_equilibrium(self, game_name: str) -> List[Tuple]:
        """Computes Nash equilibria"""
        # Simplified implementation for 2x2 games
        payoff_matrix = self.payoff_matrices[game_name]
        
        if payoff_matrix.shape == (2, 2, 2):  # 2x2 game with 2 players
            equilibria = []
            
            # Check for pure strategy equilibria
            for i in range(2):
                for j in range(2):
                    if self._is_nash_equilibrium(payoff_matrix, i, j):
                        equilibria.append((i, j))
            
            # Compute mixed strategy equilibrium if no pure equilibrium exists
            if not equilibria:
                mixed_eq = self._compute_mixed_nash(payoff_matrix)
                if mixed_eq:
                    equilibria.append(mixed_eq)
            
            return equilibria
        else:
            raise NotImplementedError("Only 2x2 games are supported for now")
    
    def _is_nash_equilibrium(self, payoff_matrix: np.ndarray, i: int, j: int) -> bool:
        """Checks if (i,j) is a Nash equilibrium"""
        # Player 1: check if i is the best response to j
        player1_payoffs = payoff_matrix[:, j, 0]
        best_response_1 = np.argmax(player1_payoffs)
        
        # Player 2: check if j is the best response to i
        player2_payoffs = payoff_matrix[i, :, 1]
        best_response_2 = np.argmax(player2_payoffs)
        
        return best_response_1 == i and best_response_2 == j
    
    def _compute_mixed_nash(self, payoff_matrix: np.ndarray) -> Optional[Tuple]:
        """Computes the mixed Nash equilibrium for a 2x2 game"""
        # For a 2x2 game, a mixed equilibrium exists if no pure equilibrium exists
        
        # Payoffs for each player
        A = payoff_matrix[:, :, 0]  # Player 1 payoffs
        B = payoff_matrix[:, :, 1]  # Player 2 payoffs
        
        # Probability that Player 1 plays strategy 0
        if A[1, 0] - A[1, 1] - A[0, 0] + A[0, 1] != 0:
            p = (B[1, 1] - B[0, 1]) / (B[1, 0] - B[1, 1] - B[0, 0] + B[0, 1])
        else:
            return None
        
        # Probability that Player 2 plays strategy 0
        if B[0, 1] - B[1, 1] - B[0, 0] + B[1, 0] != 0:
            q = (A[1, 1] - A[1, 0]) / (A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1])
        else:
            return None
        
        if 0 <= p <= 1 and 0 <= q <= 1:
            return ((p, 1-p), (q, 1-q))
        else:
            return None
    
    def maximin_strategy(self, game_name: str, player: int) -> Tuple[int, float]:
        """Maximin strategy (maximize minimum gain)"""
        payoff_matrix = self.payoff_matrices[game_name]
        
        if player == 0:
            # Player 1: maximize the minimum across columns
            min_payoffs = np.min(payoff_matrix[:, :, 0], axis=1)
            best_strategy = np.argmax(min_payoffs)
            best_payoff = min_payoffs[best_strategy]
        else:
            # Player 2: maximize the minimum across rows
            min_payoffs = np.min(payoff_matrix[:, :, 1], axis=0)
            best_strategy = np.argmax(min_payoffs)
            best_payoff = min_payoffs[best_strategy]
        
        return best_strategy, best_payoff
    
    def minimax_regret(self, game_name: str, player: int) -> Tuple[int, float]:
        """Minimax regret strategy"""
        payoff_matrix = self.payoff_matrices[game_name]
        
        if player == 0:
            payoffs = payoff_matrix[:, :, 0]
            # Regret matrix calculation
            max_per_column = np.max(payoffs, axis=0)
            regret_matrix = max_per_column - payoffs
            
            # Minimax regret
            max_regret_per_row = np.max(regret_matrix, axis=1)
            best_strategy = np.argmin(max_regret_per_row)
            min_regret = max_regret_per_row[best_strategy]
        else:
            payoffs = payoff_matrix[:, :, 1]
            max_per_row = np.max(payoffs, axis=1)
            regret_matrix = max_per_row.reshape(-1, 1) - payoffs
            
            max_regret_per_column = np.max(regret_matrix, axis=0)
            best_strategy = np.argmin(max_regret_per_column)
            min_regret = max_regret_per_column[best_strategy]
        
        return best_strategy, min_regret
    
    def expected_utility_maximization(self, game_name: str, player: int, 
                                    opponent_strategy_distribution: np.ndarray) -> Tuple[int, float]:
        """Expected utility maximization given a distribution over opponent strategies"""
        payoff_matrix = self.payoff_matrices[game_name]
        
        if player == 0:
            payoffs = payoff_matrix[:, :, 0]
            expected_payoffs = np.dot(payoffs, opponent_strategy_distribution)
        else:
            payoffs = payoff_matrix[:, :, 1]
            expected_payoffs = np.dot(payoffs.T, opponent_strategy_distribution)
        
        best_strategy = np.argmax(expected_payoffs)
        best_expected_payoff = expected_payoffs[best_strategy]
        
        return best_strategy, best_expected_payoff
    
    def robust_decision_making(self, game_name: str, player: int, 
                             uncertainty_set: List[np.ndarray]) -> Tuple[int, float]:
        """Robust decision-making under set uncertainty"""
        payoff_matrix = self.payoff_matrices[game_name]
        
        worst_case_payoffs = []
        
        for strategy_idx in range(payoff_matrix.shape[player]):
            worst_payoff = float('inf')
            
            for uncertain_payoff in uncertainty_set:
                if player == 0:
                    payoff = uncertain_payoff[strategy_idx, :, 0].min()
                else:
                    payoff = uncertain_payoff[:, strategy_idx, 1].min()
                
                worst_payoff = min(worst_payoff, payoff)
            
            worst_case_payoffs.append(worst_payoff)
        
        best_strategy = np.argmax(worst_case_payoffs)
        best_worst_case = worst_case_payoffs[best_strategy]
        
        return best_strategy, best_worst_case

class IntegratedUncertaintySystem:
    """Integrated system combining all uncertainty management components"""
    
    def __init__(self):
        self.bayesian_inference = BayesianInference()
        self.bayesian_network = BayesianNetwork()
        self.conditional_manager = ConditionalProbabilityManager()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.game_theory_dm = GameTheoryDecisionMaker()
        self.uncertainty_history = []
    
    def process_uncertain_reasoning(self, evidence: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Reasoning process integrating uncertainty"""
        results = {}
        
        # 1. Bayesian belief updating
        for parameter, value in evidence.items():
            if parameter in self.bayesian_inference.posteriors:
                likelihood_func = context.get(f'{parameter}_likelihood', 
                                            lambda x, e: stats.norm.pdf(e, loc=x, scale=1))
                posterior = self.bayesian_inference.bayesian_update(
                    parameter, likelihood_func, value
                )
                results[f'{parameter}_posterior'] = posterior
        
        # 2. Inference in the Bayesian network
        if hasattr(context, 'network_evidence'):
            for node, value in context['network_evidence'].items():
                self.bayesian_network.set_evidence(node, value)
            
            network_samples = self.bayesian_network.forward_sampling(1000)
            results['network_inference'] = network_samples
        
        # 3. Uncertainty quantification
        epistemic_uncertainty = self._compute_epistemic_uncertainty(evidence)
        aleatoric_uncertainty = self._compute_aleatoric_uncertainty(evidence)
        
        uncertainty_quantification = UncertaintyQuantification(
            epistemic=epistemic_uncertainty,
            aleatoric=aleatoric_uncertainty,
            total=epistemic_uncertainty + aleatoric_uncertainty,
            confidence=self._compute_confidence_level(epistemic_uncertainty, aleatoric_uncertainty)
        )
        
        results['uncertainty'] = uncertainty_quantification
        
        # 4. Decision-making under uncertainty
        if 'decision_context' in context:
            decision_context = context['decision_context']
            if 'game_name' in decision_context:
                game_name = decision_context['game_name']
                player = decision_context['player']
                
                # Different decision strategies
                nash_eq = self.game_theory_dm.nash_equilibrium(game_name)
                maximin = self.game_theory_dm.maximin_strategy(game_name, player)
                
                results['decision_analysis'] = {
                    'nash_equilibria': nash_eq,
                    'maximin_strategy': maximin,
                    'uncertainty_level': uncertainty_quantification.total
                }
        
        # Record history
        self.uncertainty_history.append({
            'timestamp': np.datetime64('now'),
            'evidence': evidence,
            'results': results
        })
        
        return results
    
    def _compute_epistemic_uncertainty(self, evidence: Dict[str, Any]) -> float:
        """Computes epistemic uncertainty based on evidence"""
        # Simplification: based on the variance of posteriors
        total_epistemic = 0.0
        count = 0
        
        for param, posterior in self.bayesian_inference.posteriors.items():
            if hasattr(posterior, 'var'):
                total_epistemic += posterior.var()
                count += 1
        
        return total_epistemic / max(count, 1)
    
    def _compute_aleatoric_uncertainty(self, evidence: Dict[str, Any]) -> float:
        """Computes intrinsic aleatoric uncertainty"""
        # Simplification: based on observation variability
        if isinstance(evidence, dict) and len(evidence) > 0:
            values = [v for v in evidence.values() if isinstance(v, (int, float))]
            if values:
                return np.var(values)
        return 0.1  # Default value
    
    def _compute_confidence_level(self, epistemic: float, aleatoric: float) -> float:
        """Computes confidence level based on uncertainties"""
        total_uncertainty = epistemic + aleatoric
        # Logistic transformation to get a value between 0 and 1
        confidence = 1 / (1 + np.exp(total_uncertainty))
        return confidence
    
    def visualize_uncertainty(self, parameter: str = None):
        """Visualizes uncertainty evolution"""
        if not self.uncertainty_history:
            print("No history data available")
            return
        
        timestamps = [entry['timestamp'] for entry in self.uncertainty_history]
        uncertainties = [entry['results']['uncertainty'].total for entry in self.uncertainty_history]
        confidences = [entry['results']['uncertainty'].confidence for entry in self.uncertainty_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Total uncertainty chart
        ax1.plot(timestamps, uncertainties, 'b-', label='Total uncertainty')
        ax1.set_ylabel('Uncertainty')
        ax1.set_title('Evolution of uncertainty over time')
        ax1.legend()
        ax1.grid(True)
        
        # Confidence chart
        ax2.plot(timestamps, confidences, 'r-', label='Confidence level')
        ax2.set_ylabel('Confidence')
        ax2.set_xlabel('Time')
        ax2.set_title('Evolution of confidence over time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_uncertainty_report(self) -> Dict[str, Any]:
        """Generates a detailed report on the state of uncertainty"""
        report = {
            'timestamp': np.datetime64('now'),
            'system_state': {
                'num_parameters_tracked': len(self.bayesian_inference.posteriors),
                'num_network_nodes': len(self.bayesian_network.nodes),
                'num_conditional_tables': len(self.conditional_manager.conditional_tables),
                'history_length': len(self.uncertainty_history)
            },
            'current_uncertainties': {},
            'calibration_metrics': {},
            'recommendations': []
        }
        
        # Current uncertainties per parameter
        for param, posterior in self.bayesian_inference.posteriors.items():
            report['current_uncertainties'][param] = {
                'mean': posterior.mean(),
                'variance': posterior.var(),
                'entropy': posterior.entropy()
            }
        
        # Calibration metrics
        if self.confidence_calibrator.calibration_data:
            ece, bin_data = self.confidence_calibrator.compute_calibration_error()
            report['calibration_metrics'] = {
                'expected_calibration_error': ece,
                'reliability_diagram_data': bin_data
            }
        
        # Recommendations based on system state
        if len(self.uncertainty_history) > 10:
            recent_uncertainties = [
                entry['results']['uncertainty'].total 
                for entry in self.uncertainty_history[-10:]
            ]
            if np.std(recent_uncertainties) > 0.1:
                report['recommendations'].append(
                    "Uncertainty shows high recent variability. "
                    "Consider acquiring additional data."
                )
        
        return report

# Utility functions for system usage

def create_simple_uncertainty_model(prior_params: Dict[str, Dict[str, float]]) -> IntegratedUncertaintySystem:
    """Creates a simple uncertainty model with specified priors"""
    system = IntegratedUncertaintySystem()
    
    for param_name, param_config in prior_params.items():
        dist_type = param_config.get('type', 'normal')
        dist_params = {k: v for k, v in param_config.items() if k != 'type'}
        
        prior = ProbabilisticDistribution(dist_type, dist_params)
        system.bayesian_inference.set_prior(param_name, prior)
    
    return system

def demo_uncertainty_system():
    """Demonstration of the uncertainty management system"""
    print("=== Probabilistic Reasoning System Demonstration ===\n")
    
    # Create a system with priors
    prior_params = {
        'temperature': {'type': 'normal', 'mean': 20.0, 'std': 5.0},
        'humidity': {'type': 'beta', 'alpha': 2.0, 'beta': 3.0},
        'pressure': {'type': 'gamma', 'shape': 2.0, 'scale': 1.0}
    }
    
    system = create_simple_uncertainty_model(prior_params)
    
    # Simulate observations
    observations = {
        'temperature': [22.1, 21.8, 23.2, 20.9, 22.5],
        'humidity': 0.65,
        'pressure': 2.3
    }
    
    print("1. Initial Priors:")
    for param, prior in system.bayesian_inference.priors.items():
        print(f"   {param}: mean={prior.mean():.2f}, variance={prior.var():.4f}")
    
    print("\n2. Bayesian update with observations:")
    
    # Simple likelihood function
    def simple_likelihood(param_value, observation):
        if isinstance(observation, list):
            return np.prod([stats.norm.pdf(obs, loc=param_value, scale=1.0) for obs in observation])
        else:
            return stats.norm.pdf(observation, loc=param_value, scale=1.0)
    
    # Processing observations
    context = {
        'temperature_likelihood': simple_likelihood,
        'humidity_likelihood': simple_likelihood,
        'pressure_likelihood': simple_likelihood
    }
    
    results = system.process_uncertain_reasoning(observations, context)
    
    print("\n3. Updated Posteriors:")
    for param, posterior in system.bayesian_inference.posteriors.items():
        print(f"   {param}: mean={posterior.mean():.2f}, variance={posterior.var():.4f}")
    
    print(f"\n4. Uncertainty Quantification:")
    uncertainty = results['uncertainty']
    print(f"   Epistemic uncertainty: {uncertainty.epistemic:.4f}")
    print(f"   Aleatoric uncertainty: {uncertainty.aleatoric:.4f}")
    print(f"   Total uncertainty: {uncertainty.total:.4f}")
    print(f"   Confidence level: {uncertainty.confidence:.4f}")
    
    # Game Theory Demonstration
    print("\n5. Decision-making example with game theory:")
    
    # Simple Game: Modified Prisoner's Dilemma
    payoff_matrix = np.array([
        [[3, 3], [0, 5]],   # Cooperate
        [[5, 0], [1, 1]]    # Defect
    ])
    
    system.game_theory_dm.add_game(
        'prisoner_dilemma',
        ['Player1', 'Player2'],
        {'Player1': ['Cooperate', 'Defect'], 'Player2': ['Cooperate', 'Defect']},
        payoff_matrix
    )
    
    nash_equilibria = system.game_theory_dm.nash_equilibrium('prisoner_dilemma')
    maximin_p1 = system.game_theory_dm.maximin_strategy('prisoner_dilemma', 0)
    
    print(f"   Nash Equilibria: {nash_equilibria}")
    print(f"   Maximin strategy player 1: strategy {maximin_p1[0]}, gain {maximin_p1[1]}")
    
    print("\n6. System Status Report:")
    report = system.generate_uncertainty_report()
    print(f"   Number of parameters tracked: {report['system_state']['num_parameters_tracked']}")
    print(f"   History length: {report['system_state']['history_length']}")
    
    if report['recommendations']:
        print("   Recommendations:")
        for rec in report['recommendations']:
            print(f"   - {rec}")

if __name__ == "__main__":
    # Run demonstration
    demo_uncertainty_system()
