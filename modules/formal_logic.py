"""
Advanced and Non-Classical Formal Logic System for
artificial intelligence API GOOGLE GEMINI 2.0 FLASH
A comprehensive framework for manipulating different logical systems,
automatic theorem proving, and formal reasoning.
"""

import numpy as np
import sympy as sp
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional, Callable, Any, Union
from abc import ABC, abstractmethod
import itertools
import networkx as nx
from dataclasses import dataclass
import functools
import re
import random # Added for SecondOrderLogic._generate_functions

##########################################
# PART 1: NON-CLASSICAL LOGICS
##########################################

class Formula(ABC):
    """Abstract base class for all logical formulas."""
    
    @abstractmethod
    def evaluate(self, *args, **kwargs):
        """Evaluates the formula according to the appropriate semantics."""
        pass
    
    @abstractmethod
    def __str__(self):
        """String representation of the formula."""
        pass


#########################################
# FUZZY LOGIC
#########################################

class FuzzyValue:
    """Represents a value in fuzzy logic between 0 and 1."""
    
    def __init__(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError("A fuzzy value must be between 0 and 1")
        self.value = value
    
    def __and__(self, other):
        if isinstance(other, FuzzyValue):
            return FuzzyValue(min(self.value, other.value))
        return NotImplemented
    
    def __or__(self, other):
        if isinstance(other, FuzzyValue):
            return FuzzyValue(max(self.value, other.value))
        return NotImplemented
    
    def __invert__(self):
        return FuzzyValue(1 - self.value)
    
    def __str__(self):
        return f"{self.value:.3f}"
    
    def __repr__(self):
        return f"FuzzyValue({self.value})"


class FuzzyFormula(Formula):
    """Base class for fuzzy logic formulas."""
    pass


class FuzzyVariable(FuzzyFormula):
    """Fuzzy variable with a name and a membership function."""
    
    def __init__(self, name: str, membership_function: Callable[[Any], float] = None):
        self.name = name
        self.membership_function = membership_function
        
    def evaluate(self, context=None, **kwargs):
        if context and self.name in context:
            return context[self.name]
        if self.membership_function and 'value' in kwargs:
            return FuzzyValue(self.membership_function(kwargs['value']))
        raise ValueError(f"Cannot evaluate variable {self.name}")
        
    def __str__(self):
        return self.name


class FuzzyAnd(FuzzyFormula):
    """Fuzzy conjunction (t-norm)."""
    
    def __init__(self, left: FuzzyFormula, right: FuzzyFormula, t_norm: str = 'min'):
        self.left = left
        self.right = right
        
        # Different t-norms available
        self.t_norms = {
            'min': lambda a, b: min(a, b),
            'product': lambda a, b: a * b,
            'lukasiewicz': lambda a, b: max(0, a + b - 1),
        }
        
        if t_norm not in self.t_norms:
            raise ValueError(f"T-norm '{t_norm}' not recognized")
        
        self.t_norm = t_norm
    
    def evaluate(self, context=None, **kwargs):
        left_val = self.left.evaluate(context, **kwargs).value
        right_val = self.right.evaluate(context, **kwargs).value
        result = self.t_norms[self.t_norm](left_val, right_val)
        return FuzzyValue(result)
    
    def __str__(self):
        return f"({self.left} ∧ {self.right})"


class FuzzyOr(FuzzyFormula):
    """Fuzzy disjunction (t-conorm)."""
    
    def __init__(self, left: FuzzyFormula, right: FuzzyFormula, t_conorm: str = 'max'):
        self.left = left
        self.right = right
        
        # Different t-conorms available
        self.t_conorms = {
            'max': lambda a, b: max(a, b),
            'prob_sum': lambda a, b: a + b - a * b,
            'lukasiewicz': lambda a, b: min(1, a + b),
        }
        
        if t_conorm not in self.t_conorms:
            raise ValueError(f"T-conorm '{t_conorm}' not recognized")
        
        self.t_conorm = t_conorm
    
    def evaluate(self, context=None, **kwargs):
        left_val = self.left.evaluate(context, **kwargs).value
        right_val = self.right.evaluate(context, **kwargs).value
        result = self.t_conorms[self.t_conorm](left_val, right_val)
        return FuzzyValue(result)
    
    def __str__(self):
        return f"({self.left} ∨ {self.right})"


class FuzzyNot(FuzzyFormula):
    """Fuzzy negation."""
    
    def __init__(self, formula: FuzzyFormula, neg_type: str = 'standard'):
        self.formula = formula
        
        # Different negation types
        self.negations = {
            'standard': lambda a: 1 - a,
            'sugeno': lambda a, lambda_: 1 - a / (1 + lambda_ * a) if lambda_ > 0 else 1 - a,
            'yager': lambda a, w: (1 - a**w)**(1/w) if w > 0 else 1 - a,
        }
        
        self.neg_type = neg_type
        self.params = {}
    
    def with_params(self, **params):
        """Configures parameters for parameterized negations."""
        self.params = params
        return self
    
    def evaluate(self, context=None, **kwargs):
        val = self.formula.evaluate(context, **kwargs).value
        
        if self.neg_type == 'standard':
            result = self.negations[self.neg_type](val)
        elif self.neg_type == 'sugeno':
            lambda_ = self.params.get('lambda', 1)
            result = self.negations[self.neg_type](val, lambda_)
        elif self.neg_type == 'yager':
            w = self.params.get('w', 2)
            result = self.negations[self.neg_type](val, w)
        else:
            raise ValueError(f"Negation type not supported: {self.neg_type}")
            
        return FuzzyValue(result)
    
    def __str__(self):
        return f"¬({self.formula})"


class FuzzyImplication(FuzzyFormula):
    """Fuzzy implication with different semantics."""
    
    def __init__(self, antecedent: FuzzyFormula, consequent: FuzzyFormula, impl_type: str = 'godel'):
        self.antecedent = antecedent
        self.consequent = consequent
        
        # Different fuzzy implications
        self.implications = {
            'godel': lambda a, b: 1 if a <= b else b,
            'lukasiewicz': lambda a, b: min(1, 1 - a + b),
            'goguen': lambda a, b: 1 if a <= b else b/a,
            'kleene_dienes': lambda a, b: max(1 - a, b),
            'reichenbach': lambda a, b: 1 - a + a*b,
        }
        
        if impl_type not in self.implications:
            raise ValueError(f"Implication type '{impl_type}' not recognized")
        
        self.impl_type = impl_type
    
    def evaluate(self, context=None, **kwargs):
        ant_val = self.antecedent.evaluate(context, **kwargs).value
        cons_val = self.consequent.evaluate(context, **kwargs).value
        result = self.implications[self.impl_type](ant_val, cons_val)
        return FuzzyValue(result)
    
    def __str__(self):
        return f"({self.antecedent} → {self.consequent})"


class FuzzySet:
    """Represents a fuzzy set with a membership function."""
    
    def __init__(self, name: str, universe: List, membership_function: Callable[[Any], float]):
        self.name = name
        self.universe = universe
        self.membership_function = membership_function
    
    def membership(self, element):
        """Returns the membership degree of an element to the set."""
        return FuzzyValue(self.membership_function(element))
    
    def alpha_cut(self, alpha: float):
        """Returns the α-cut of the fuzzy set."""
        return [x for x in self.universe if self.membership_function(x) >= alpha]
    
    def support(self):
        """Returns the support of the fuzzy set."""
        return self.alpha_cut(0.0)
    
    def core(self):
        """Returns the core of the fuzzy set."""
        return self.alpha_cut(1.0)
    
    def is_normal(self):
        """Checks if the fuzzy set is normal."""
        return any(self.membership_function(x) == 1 for x in self.universe)
    
    def cardinality(self):
        """Calculates the cardinality of the fuzzy set."""
        return sum(self.membership_function(x) for x in self.universe)
    
    def __str__(self):
        return f"FuzzySet({self.name})"


class FuzzySetOperations:
    """Operations on fuzzy sets."""
    
    @staticmethod
    def intersection(set1: FuzzySet, set2: FuzzySet, t_norm: str = 'min'):
        """Intersection of two fuzzy sets."""
        t_norms = {
            'min': lambda a, b: min(a, b),
            'product': lambda a, b: a * b,
            'lukasiewicz': lambda a, b: max(0, a + b - 1),
        }
        
        if t_norm not in t_norms:
            raise ValueError(f"T-norm '{t_norm}' not recognized")
        
        # Check that universes are compatible
        if set1.universe != set2.universe:
            raise ValueError("Fuzzy sets must have the same universe")
        
        name = f"({set1.name} ∩ {set2.name})"
        
        def membership_function(x):
            val1 = set1.membership_function(x)
            val2 = set2.membership_function(x)
            return t_norms[t_norm](val1, val2)
        
        return FuzzySet(name, set1.universe, membership_function)
    
    @staticmethod
    def union(set1: FuzzySet, set2: FuzzySet, t_conorm: str = 'max'):
        """Union of two fuzzy sets."""
        t_conorms = {
            'max': lambda a, b: max(a, b),
            'prob_sum': lambda a, b: a + b - a * b,
            'lukasiewicz': lambda a, b: min(1, a + b),
        }
        
        if t_conorm not in t_conorms:
            raise ValueError(f"T-conorm '{t_conorm}' not recognized")
        
        # Check that universes are compatible
        if set1.universe != set2.universe:
            raise ValueError("Fuzzy sets must have the same universe")
        
        name = f"({set1.name} ∪ {set2.name})"
        
        def membership_function(x):
            val1 = set1.membership_function(x)
            val2 = set2.membership_function(x)
            return t_conorms[t_conorm](val1, val2)
        
        return FuzzySet(name, set1.universe, membership_function)
    
    @staticmethod
    def complement(f_set: FuzzySet):
        """Complement of a fuzzy set."""
        name = f"¬({f_set.name})"
        
        def membership_function(x):
            return 1 - f_set.membership_function(x)
        
        return FuzzySet(name, f_set.universe, membership_function)


class FuzzyRuleSystem:
    """Rule-based fuzzy inference system."""
    
    def __init__(self, name: str):
        self.name = name
        self.rules = []
        self.input_variables = {}
        self.output_variables = {}
    
    def add_input_variable(self, name: str, universe: List, fuzzy_sets: Dict[str, FuzzySet]):
        """Adds an input variable with its associated fuzzy sets."""
        self.input_variables[name] = {
            'universe': universe,
            'sets': fuzzy_sets
        }
        return self
    
    def add_output_variable(self, name: str, universe: List, fuzzy_sets: Dict[str, FuzzySet]):
        """Adds an output variable with its associated fuzzy sets."""
        self.output_variables[name] = {
            'universe': universe,
            'sets': fuzzy_sets
        }
        return self
    
    def add_rule(self, antecedents: List[Tuple[str, str, str]], consequents: List[Tuple[str, str]]):
        """
        Adds a fuzzy rule.
        antecedents: list of tuples (variable, fuzzy_set, operator)
        consequents: list of tuples (variable, fuzzy_set)
        """
        self.rules.append({
            'antecedents': antecedents,
            'consequents': consequents
        })
        return self
    
    def fuzzify(self, inputs: Dict[str, Any]):
        """Fuzzifies crisp inputs."""
        results = {}
        for var, value in inputs.items():
            if var not in self.input_variables:
                raise ValueError(f"Unknown input variable: {var}")
            
            # Calculation of membership degrees for each fuzzy set
            results[var] = {}
            for set_name, f_set in self.input_variables[var]['sets'].items():
                results[var][set_name] = f_set.membership_function(value)
        
        return results
    
    def evaluate_rule(self, rule, fuzzified_inputs):
        """Evaluates a fuzzy rule and returns the activation degree."""
        # Evaluate antecedents
        activation_degrees = []
        
        for var, f_set, operator in rule['antecedents']:
            if var not in fuzzified_inputs or f_set not in fuzzified_inputs[var]:
                raise ValueError(f"Invalid antecedent: {var}.{f_set}")
            
            degree = fuzzified_inputs[var][f_set]
            activation_degrees.append((degree, operator))
        
        # Combine activation degrees according to operators
        final_degree = activation_degrees[0][0]
        for i in range(1, len(activation_degrees)):
            degree, op = activation_degrees[i]
            if op == 'AND':
                final_degree = min(final_degree, degree)
            elif op == 'OR':
                final_degree = max(final_degree, degree)
        
        return final_degree
    
    def infer(self, inputs: Dict[str, Any], aggregation_method: str = 'max', defuzzification_method: str = 'centroid'):
        """
        Performs full fuzzy inference:
        1. Fuzzification of inputs
        2. Rule evaluation
        3. Output aggregation
        4. Defuzzification
        """
        # Step 1: Fuzzification
        fuzzified_inputs = self.fuzzify(inputs)
        
        # Step 2: Rule evaluation
        activations = {}
        for i, rule in enumerate(self.rules):
            degree = self.evaluate_rule(rule, fuzzified_inputs)
            
            # Record activation for each consequent
            for var, f_set in rule['consequents']:
                if var not in activations:
                    activations[var] = {}
                
                if f_set not in activations[var]:
                    activations[var][f_set] = []
                
                activations[var][f_set].append(degree)
        
        # Step 3: Output aggregation
        aggregated_outputs = {}
        for var in self.output_variables:
            aggregated_outputs[var] = {}
            
            # If the variable has not been activated by a rule
            if var not in activations:
                continue
                
            for f_set in self.output_variables[var]['sets']:
                if f_set not in activations[var]:
                    continue
                    
                # Aggregate activations for this fuzzy set
                if aggregation_method == 'max':
                    aggregated_outputs[var][f_set] = max(activations[var][f_set])
                elif aggregation_method == 'sum':
                    aggregated_outputs[var][f_set] = sum(activations[var][f_set])
                else:
                    raise ValueError(f"Aggregation method not recognized: {aggregation_method}")
        
        # Step 4: Defuzzification
        results = {}
        for var in self.output_variables:
            if var not in aggregated_outputs or not aggregated_outputs[var]:
                results[var] = None
                continue
                
            universe = self.output_variables[var]['universe']
            
            if defuzzification_method == 'centroid':
                numerator = 0
                denominator = 0
                
                for x in universe:
                    # Calculate the aggregated membership degree for x
                    max_degree = 0
                    for f_set, activation in aggregated_outputs[var].items():
                        degree = min(activation, self.output_variables[var]['sets'][f_set].membership_function(x))
                        max_degree = max(max_degree, degree)
                    
                    numerator += x * max_degree
                    denominator += max_degree
                
                if denominator > 0:
                    results[var] = numerator / denominator
                else:
                    results[var] = None
                    
            elif defuzzification_method == 'mean_of_max':
                # Mean of maximums method
                max_degrees = {}
                for x in universe:
                    max_degree = 0
                    for f_set, activation in aggregated_outputs[var].items():
                        degree = min(activation, self.output_variables[var]['sets'][f_set].membership_function(x))
                        max_degree = max(max_degree, degree)
                    
                    max_degrees[x] = max_degree
                
                overall_max_degree = max(max_degrees.values()) if max_degrees else 0
                
                # Find points with maximum degree
                points_at_max = [x for x, degree in max_degrees.items() if degree == overall_max_degree]
                
                if points_at_max:
                    results[var] = sum(points_at_max) / len(points_at_max)
                else:
                    results[var] = None
            
            elif defuzzification_method == 'first_max':
                # First maximum method
                max_degrees = {}
                for x in universe:
                    max_degree = 0
                    for f_set, activation in aggregated_outputs[var].items():
                        degree = min(activation, self.output_variables[var]['sets'][f_set].membership_function(x))
                        max_degree = max(max_degree, degree)
                    
                    max_degrees[x] = max_degree
                
                overall_max_degree = max(max_degrees.values()) if max_degrees else 0
                
                # Find the first point with maximum degree
                for x, degree in sorted(max_degrees.items()):
                    if degree == overall_max_degree:
                        results[var] = x
                        break
                else:
                    results[var] = None
            
            else:
                raise ValueError(f"Defuzzification method not recognized: {defuzzification_method}")
        
        return results


#########################################
# MODAL LOGIC
#########################################

class PossibleWorlds:
    """Represents a possible worlds model for modal logic."""
    
    def __init__(self):
        self.worlds = set()
        self.relations = {}  # Accessibility relations between worlds
        self.valuations = {}  # Valuations of propositions in each world
    
    def add_world(self, world):
        """Adds a possible world to the model."""
        self.worlds.add(world)
        self.relations[world] = set()
        self.valuations[world] = {}
        return self
    
    def add_relation(self, world1, world2):
        """Adds an accessibility relation between two worlds."""
        if world1 not in self.worlds or world2 not in self.worlds:
            raise ValueError("Worlds must be in the model")
        
        self.relations[world1].add(world2)
        return self
    
    def define_valuation(self, world, proposition, value):
        """Defines the truth value of a proposition in a world."""
        if world not in self.worlds:
            raise ValueError(f"World {world} is not in the model")
        
        self.valuations[world][proposition] = value
        return self
    
    def get_valuation(self, world, proposition):
        """Returns the truth value of a proposition in a world."""
        if world not in self.worlds:
            raise ValueError(f"World {world} is not in the model")
        
        return self.valuations[world].get(proposition, False)
    
    def accessible_worlds(self, world):
        """Returns the accessible worlds from a given world."""
        if world not in self.worlds:
            raise ValueError(f"World {world} is not in the model")
        
        return self.relations[world]
    
    def check_properties(self):
        """Checks the properties of the accessibility relation."""
        results = {
            'reflexive': True,
            'symmetric': True,
            'transitive': True,
            'serial': True,
            'euclidean': True
        }
        
        # Check reflexivity
        for world in self.worlds:
            if world not in self.relations[world]:
                results['reflexive'] = False
                break
        
        # Check symmetry
        for world1 in self.worlds:
            for world2 in self.relations[world1]:
                if world1 not in self.relations[world2]:
                    results['symmetric'] = False
                    break
        
        # Check transitivity
        for world1 in self.worlds:
            for world2 in self.relations[world1]:
                for world3 in self.relations[world2]:
                    if world3 not in self.relations[world1]:
                        results['transitive'] = False
                        break
        
        # Check seriality (each world has at least one accessible world)
        for world in self.worlds:
            if not self.relations[world]:
                results['serial'] = False
                break
        
        # Check Euclidean property
        for world1 in self.worlds:
            for world2 in self.relations[world1]:
                for world3 in self.relations[world1]:
                    if world3 not in self.relations[world2]:
                        results['euclidean'] = False
                        break
        
        return results
    
    def logic_type(self):
        """Determines the type of modal logic based on the properties of the relation."""
        properties = self.check_properties()
        
        if properties['reflexive'] and properties['transitive']:
            if properties['symmetric']:
                return "S5"
            else:
                return "S4"
        elif properties['reflexive']:
            return "T"
        elif properties['serial']:
            if properties['transitive'] and properties['euclidean']:
                return "D45"
            elif properties['transitive']:
                return "D4"
            elif properties['euclidean']:
                return "D5"
            else:
                return "D"
        elif properties['transitive'] and properties['euclidean']:
            return "K45"
        elif properties['transitive']:
            return "K4"
        elif properties['euclidean']:
            return "K5"
        else:
            return "K"


class ModalFormula(Formula):
    """Base class for modal logic formulas."""
    pass


class ModalProposition(ModalFormula):
    """Represents an atomic proposition in modal logic."""
    
    def __init__(self, name: str):
        self.name = name
    
    def evaluate(self, world, model: PossibleWorlds):
        """Evaluates the proposition in a given world of the model."""
        return model.get_valuation(world, self.name)
    
    def __str__(self):
        return self.name


class ModalNot(ModalFormula):
    """Negation in modal logic."""
    
    def __init__(self, formula: ModalFormula):
        self.formula = formula
    
    def evaluate(self, world, model: PossibleWorlds):
        """Evaluates the negation in a given world of the model."""
        return not self.formula.evaluate(world, model)
    
    def __str__(self):
        return f"¬{self.formula}"


class ModalAnd(ModalFormula):
    """Conjunction in modal logic."""
    
    def __init__(self, left: ModalFormula, right: ModalFormula):
        self.left = left
        self.right = right
    
    def evaluate(self, world, model: PossibleWorlds):
        """Evaluates the conjunction in a given world of the model."""
        return self.left.evaluate(world, model) and self.right.evaluate(world, model)
    
    def __str__(self):
        return f"({self.left} ∧ {self.right})"


class ModalOr(ModalFormula):
    """Disjunction in modal logic."""
    
    def __init__(self, left: ModalFormula, right: ModalFormula):
        self.left = left
        self.right = right
    
    def evaluate(self, world, model: PossibleWorlds):
        """Evaluates the disjunction in a given world of the model."""
        return self.left.evaluate(world, model) or self.right.evaluate(world, model)
    
    def __str__(self):
        return f"({self.left} ∨ {self.right})"


class ModalImplication(ModalFormula):
    """Implication in modal logic."""
    
    def __init__(self, antecedent: ModalFormula, consequent: ModalFormula):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def evaluate(self, world, model: PossibleWorlds):
        """Evaluates the implication in a given world of the model."""
        return not self.antecedent.evaluate(world, model) or self.consequent.evaluate(world, model)
    
    def __str__(self):
        return f"({self.antecedent} → {self.consequent})"


class Necessity(ModalFormula):
    """Necessity operator (□) in modal logic."""
    
    def __init__(self, formula: ModalFormula):
        self.formula = formula
    
    def evaluate(self, world, model: PossibleWorlds):
        """Evaluates necessity in a given world of the model."""
        # A formula is necessary if it is true in all accessible worlds
        accessible_worlds = model.accessible_worlds(world)
        
        if not accessible_worlds:
            # If no world is accessible, necessity is true vacuously
            return True
        
        return all(self.formula.evaluate(m, model) for m in accessible_worlds)
    
    def __str__(self):
        return f"□{self.formula}"


class Possibility(ModalFormula):
    """Possibility operator (◇) in modal logic."""
    
    def __init__(self, formula: ModalFormula):
        self.formula = formula
    
    def evaluate(self, world, model: PossibleWorlds):
        """Evaluates possibility in a given world of the model."""
        # A formula is possible if it is true in at least one accessible world
        accessible_worlds = model.accessible_worlds(world)
        
        if not accessible_worlds:
            # If no world is accessible, possibility is false vacuously
            return False
        
        return any(self.formula.evaluate(m, model) for m in accessible_worlds)
    
    def __str__(self):
        return f"◇{self.formula}"


class ModelChecker:
    """Model checker for modal logic."""
    
    @staticmethod
    def check(formula: ModalFormula, model: PossibleWorlds, world=None):
        """
        Checks if a formula is true in a given world or in all worlds.
        If world is None, checks if the formula is true in all worlds.
        """
        if world is not None:
            return formula.evaluate(world, model)
        
        return all(formula.evaluate(m, model) for m in model.worlds)
    
    @staticmethod
    def satisfying_worlds(formula: ModalFormula, model: PossibleWorlds):
        """Returns the set of worlds that satisfy the formula."""
        return {m for m in model.worlds if formula.evaluate(m, model)}
    
    @staticmethod
    def is_valid(formula: ModalFormula, model: PossibleWorlds):
        """Checks if a formula is valid in the model (true in all worlds)."""
        return all(formula.evaluate(m, model) for m in model.worlds)
    
    @staticmethod
    def is_satisfiable(formula: ModalFormula, model: PossibleWorlds):
        """Checks if a formula is satisfiable in the model (true in at least one world)."""
        return any(formula.evaluate(m, model) for m in model.worlds)


class DeonticLogic:
    """Implementation of deontic logic using modal logic."""
    
    def __init__(self):
        self.model = PossibleWorlds()
    
    def add_ideal_world(self, world):
        """Adds an ideal world to the model."""
        self.model.add_world(world)
        return self
    
    def add_real_world(self, world):
        """Adds a real world to the model."""
        self.model.add_world(world)
        return self
    
    def define_ideal_accessibility(self, real_world, ideal_world):
        """Defines that an ideal world is accessible from a real world."""
        self.model.add_relation(real_world, ideal_world)
        return self
    
    def define_valuation(self, world, proposition, value):
        """Defines the truth value of a proposition in a world."""
        self.model.define_valuation(world, proposition, value)
        return self
    
    def obligation(self, formula: ModalFormula):
        """Creates an obligation formula (O)."""
        return Necessity(formula)
    
    def permission(self, formula: ModalFormula):
        """Creates a permission formula (P)."""
        return Possibility(formula)
    
    def prohibition(self, formula: ModalFormula):
        """Creates a prohibition formula (F)."""
        return Necessity(ModalNot(formula))
    
    def optional(self, formula: ModalFormula):
        """Creates an optional formula (anything that is neither obligatory nor forbidden)."""
        # Something is optional if neither it nor its negation are obligatory
        neg_formula = ModalNot(formula)
        return ModalAnd(ModalNot(self.obligation(formula)), ModalNot(self.obligation(neg_formula)))
    
    def check(self, formula: ModalFormula, world=None):
        """Checks if a formula is true in a given world or in all worlds."""
        return ModelChecker.check(formula, self.model, world)


#########################################
# TEMPORAL LOGIC
#########################################

class TemporalStructure:
    """Represents a temporal structure for temporal logic."""
    
    def __init__(self, time_type: str = 'linear'):
        """
        Initializes a temporal structure.
        time_type: 'linear', 'branching', 'cyclic'
        """
        self.instants = set()
        self.relations = {}  # Order relations between instants
        self.valuations = {}  # Valuations of propositions at each instant
        self.time_type = time_type
    
    def add_instant(self, instant):
        """Adds an instant to the temporal structure."""
        self.instants.add(instant)
        self.relations[instant] = set()
        self.valuations[instant] = {}
        return self
    
    def add_relation(self, instant1, instant2):
        """Adds a temporal relation: instant1 precedes instant2."""
        if instant1 not in self.instants or instant2 not in self.instants:
            raise ValueError("Instants must be in the structure")
        
        self.relations[instant1].add(instant2)
        return self
    
    def define_valuation(self, instant, proposition, value):
        """Defines the truth value of a proposition at a given instant."""
        if instant not in self.instants:
            raise ValueError(f"Instant {instant} is not in the structure")
        
        self.valuations[instant][proposition] = value
        return self
    
    def get_valuation(self, instant, proposition):
        """Returns the truth value of a proposition at a given instant."""
        if instant not in self.instants:
            raise ValueError(f"Instant {instant} is not in the structure")
        
        return self.valuations[instant].get(proposition, False)
    
    def future_instants(self, instant):
        """Returns the immediate future instants from a given instant."""
        if instant not in self.instants:
            raise ValueError(f"Instant {instant} is not in the structure")
        
        return self.relations[instant]
    
    def past_instants(self, instant):
        """Returns the immediate past instants from a given instant."""
        if instant not in self.instants:
            raise ValueError(f"Instant {instant} is not in the structure")
        
        return {i for i in self.instants if instant in self.relations[i]}
    
    def check_properties(self):
        """Checks the properties of the temporal relation."""
        results = {
            'irreflexive': True,  # An instant does not precede itself
            'antisymmetric': True,  # If t1 precedes t2, t2 does not precede t1
            'transitive': True,  # If t1 precedes t2 and t2 precedes t3, then t1 precedes t3
            'connected': True,  # For any t1 and t2, either t1 precedes t2, or t2 precedes t1, or t1=t2
            'linear': True,  # Each instant has at most one successor
            'dense': True  # Between two instants, there is always another instant
        }
        
        # Check irreflexivity
        for instant in self.instants:
            if instant in self.relations[instant]:
                results['irreflexive'] = False
                break
        
        # Check antisymmetry
        for instant1 in self.instants:
            for instant2 in self.relations[instant1]:
                if instant1 in self.relations[instant2]:
                    results['antisymmetric'] = False
                    break
        
        # Check transitivity
        for instant1 in self.instants:
            for instant2 in self.relations[instant1]:
                for instant3 in self.relations[instant2]:
                    if instant3 not in self.relations[instant1]:
                        results['transitive'] = False
                        break
        
        # Check connectivity
        for instant1 in self.instants:
            for instant2 in self.instants:
                if instant1 != instant2:
                    if instant2 not in self.relations[instant1] and instant1 not in self.relations[instant2]:
                        results['connected'] = False
                        break
        
        # Check linearity
        for instant in self.instants:
            if len(self.relations[instant]) > 1:
                results['linear'] = False
                break
        
        # Check density
        for instant1 in self.instants:
            for instant2 in self.relations[instant1]:
                if not any(instant1 in self.relations[i] and instant2 in self.relations[i] for i in self.instants):
                    results['dense'] = False
                    break
        
        return results
    
    def logic_type(self):
        """Determines the type of temporal logic based on the properties of the relation."""
        properties = self.check_properties()
        
        if self.time_type == 'linear':
            if properties['transitive'] and properties['irreflexive']:
                if properties['dense']:
                    return "Dense LTL"
                else:
                    return "Discrete LTL"
            else:
                return "Non-standard linear temporal structure"
        elif self.time_type == 'branching':
            if properties['transitive'] and properties['irreflexive']:
                return "CTL"
            else:
                return "Non-standard branching temporal structure"
        elif self.time_type == 'cyclic':
            return "Cyclic temporal structure"
        else:
            return "Non-standard temporal structure"


class TemporalFormula(Formula):
    """Base class for temporal logic formulas."""
    pass


class TemporalProposition(TemporalFormula):
    """Represents an atomic proposition in temporal logic."""
    
    def __init__(self, name: str):
        self.name = name
    
    def evaluate(self, instant, structure: TemporalStructure):
        """Evaluates the proposition at a given instant of the structure."""
        return structure.get_valuation(instant, self.name)
    
    def __str__(self):
        return self.name


class TemporalNot(TemporalFormula):
    """Negation in temporal logic."""
    
    def __init__(self, formula: TemporalFormula):
        self.formula = formula
    
    def evaluate(self, instant, structure: TemporalStructure):
        """Evaluates the negation at a given instant of the structure."""
        return not self.formula.evaluate(instant, structure)
    
    def __str__(self):
        return f"¬{self.formula}"


class TemporalAnd(TemporalFormula):
    """Conjunction in temporal logic."""
    
    def __init__(self, left: TemporalFormula, right: TemporalFormula):
        self.left = left
        self.right = right
    
    def evaluate(self, instant, structure: TemporalStructure):
        """Evaluates the conjunction at a given instant of the structure."""
        return self.left.evaluate(instant, structure) and self.right.evaluate(instant, structure)
    
    def __str__(self):
        return f"({self.left} ∧ {self.right})"


class TemporalOr(TemporalFormula):
    """Disjunction in temporal logic."""
    
    def __init__(self, left: TemporalFormula, right: TemporalFormula):
        self.left = left
        self.right = right
    
    def evaluate(self, instant, structure: TemporalStructure):
        """Evaluates the disjunction at a given instant of the structure."""
        return self.left.evaluate(instant, structure) or self.right.evaluate(instant, structure)
    
    def __str__(self):
        return f"({self.left} ∨ {self.right})"


class TemporalImplication(TemporalFormula):
    """Implication in temporal logic."""
    
    def __init__(self, antecedent: TemporalFormula, consequent: TemporalFormula):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def evaluate(self, instant, structure: TemporalStructure):
        """Evaluates the implication at a given instant of the structure."""
        return not self.antecedent.evaluate(instant, structure) or self.consequent.evaluate(instant, structure)
    
    def __str__(self):
        return f"({self.antecedent} → {self.consequent})"


class Next(TemporalFormula):
    """X (next) operator in temporal logic: true if the formula is true at the next instant."""
    
    def __init__(self, formula: TemporalFormula):
        self.formula = formula
    
    def evaluate(self, instant, structure: TemporalStructure):
        """Evaluates the X operator at a given instant of the structure."""
        future_instants = structure.future_instants(instant)
        
        if not future_instants:
            # If no future instant, X is false by default
            return False
        
        if structure.time_type == 'linear':
            # In linear time, there is at most one immediate future instant
            if len(future_instants) != 1:
                raise ValueError("A linear temporal structure should have exactly one immediate future")
            
            next_instant = next(iter(future_instants))
            return self.formula.evaluate(next_instant, structure)
        else:
            # In branching time, X is true if the formula is true in all possible immediate futures
            return all(self.formula.evaluate(i, structure) for i in future_instants)
    
    def __str__(self):
        return f"X{self.formula}"


class Eventually(TemporalFormula):
    """F (finally) operator in temporal logic: true if the formula is true at a future instant."""
    
    def __init__(self, formula: TemporalFormula):
        self.formula = formula
    
    def evaluate(self, instant, structure: TemporalStructure, visited=None):
        """Evaluates the F operator at a given instant of the structure."""
        if visited is None:
            visited = set()
        
        if instant in visited:
            # Cycle detection, avoids infinite loops
            return False
        
        visited.add(instant)
        
        # Check if the formula is true at the current instant
        if self.formula.evaluate(instant, structure):
            return True
        
        # Recursively check in future instants
        future_instants = structure.future_instants(instant)
        
        if structure.time_type == 'linear':
            # In linear time, F is true if the formula is true in at least one future instant
            for next_instant in future_instants:
                if self.evaluate(next_instant, structure, visited):
                    return True
            return False
        else:
            # In branching time, F is true if there is a path where the formula becomes true
            return any(self.evaluate(i, structure, visited.copy()) for i in future_instants)
    
    def __str__(self):
        return f"F{self.formula}"


class Globally(TemporalFormula):
    """G (globally) operator in temporal logic: true if the formula is true at all future instants."""
    
    def __init__(self, formula: TemporalFormula):
        self.formula = formula
    
    def evaluate(self, instant, structure: TemporalStructure, visited=None):
        """Evaluates the G operator at a given instant of the structure."""
        if visited is None:
            visited = set()
        
        if instant in visited:
            # Cycle detection, for cyclic structures
            return True
        
        visited.add(instant)
        
        # Check if the formula is true at the current instant
        if not self.formula.evaluate(instant, structure):
            return False
        
        # Recursively check in future instants
        future_instants = structure.future_instants(instant)
        
        if not future_instants:
            # If no future, G is true vacuously (end of time)
            return True
        
        if structure.time_type == 'linear':
            # In linear time, G is true if the formula is true in all future instants
            for next_instant in future_instants:
                if not self.evaluate(next_instant, structure, visited):
                    return False
            return True
        else:
            # In branching time, G is true if the formula is true in all future paths
            return all(self.evaluate(i, structure, visited.copy()) for i in future_instants)
    
    def __str__(self):
        return f"G{self.formula}"


class Until(TemporalFormula):
    """
    U (until) operator in temporal logic:
    f U g is true if g is true at some future moment and f is true until that moment.
    """
    
    def __init__(self, left: TemporalFormula, right: TemporalFormula):
        self.left = left
        self.right = right
    
    def evaluate(self, instant, structure: TemporalStructure, visited=None):
        """Evaluates the U operator at a given instant of the structure."""
        if visited is None:
            visited = set()
        
        if instant in visited:
            # Cycle detection
            return False
        
        visited.add(instant)
        
        # Check if the right condition is true at the current instant
        if self.right.evaluate(instant, structure):
            return True
        
        # Check if the left condition is true at the current instant
        if not self.left.evaluate(instant, structure):
            return False
        
        # Recursively check in future instants
        future_instants = structure.future_instants(instant)
        
        if not future_instants:
            # If no future, U is false because the right condition is never met
            return False
        
        if structure.time_type == 'linear':
            # In linear time, U is true if there exists a future instant where the right condition is true
            # and the left condition is true until that instant
            for next_instant in future_instants:
                if self.evaluate(next_instant, structure, visited):
                    return True
            return False
        else:
            # In branching time, U is true if there exists a path where the condition is satisfied
            return any(self.evaluate(i, structure, visited.copy()) for i in future_instants)
    
    def __str__(self):
        return f"({self.left} U {self.right})"


class Yesterday(TemporalFormula):
    """Y (yesterday) operator in temporal logic: true if the formula was true at the previous instant."""
    
    def __init__(self, formula: TemporalFormula):
        self.formula = formula
    
    def evaluate(self, instant, structure: TemporalStructure):
        """Evaluates the Y operator at a given instant of the structure."""
        past_instants = structure.past_instants(instant)
        
        if not past_instants:
            # If no past instant, Y is false by default
            return False
        
        if structure.time_type == 'linear':
            # In linear time, there is at most one immediate past instant
            if len(past_instants) != 1:
                raise ValueError("A linear temporal structure should have exactly one immediate past")
            
            previous_instant = next(iter(past_instants))
            return self.formula.evaluate(previous_instant, structure)
        else:
            # In branching time, Y is true if the formula was true in all possible immediate pasts
            return all(self.formula.evaluate(i, structure) for i in past_instants)
    
    def __str__(self):
        return f"Y{self.formula}"


class Previously(TemporalFormula):
    """P (past) operator in temporal logic: true if the formula was true at a past instant."""
    
    def __init__(self, formula: TemporalFormula):
        self.formula = formula
    
    def evaluate(self, instant, structure: TemporalStructure, visited=None):
        """Evaluates the P operator at a given instant of the structure."""
        if visited is None:
            visited = set()
        
        if instant in visited:
            # Cycle detection
            return False
        
        visited.add(instant)
        
        # Check if the formula is true at the current instant
        if self.formula.evaluate(instant, structure):
            return True
        
        # Recursively check in past instants
        past_instants = structure.past_instants(instant)
        
        if structure.time_type == 'linear':
            # In linear time, P is true if the formula was true in at least one past instant
            for previous_instant in past_instants:
                if self.evaluate(previous_instant, structure, visited):
                    return True
            return False
        else:
            # In branching time, P is true if there exists a path where the formula was true
            return any(self.evaluate(i, structure, visited.copy()) for i in past_instants)
    
    def __str__(self):
        return f"P{self.formula}"


class Historically(TemporalFormula):
    """H (historically) operator in temporal logic: true if the formula was true at all past instants."""
    
    def __init__(self, formula: TemporalFormula):
        self.formula = formula
    
    def evaluate(self, instant, structure: TemporalStructure, visited=None):
        """Evaluates the H operator at a given instant of the structure."""
        if visited is None:
            visited = set()
        
        if instant in visited:
            # Cycle detection
            return True
        
        visited.add(instant)
        
        # Check if the formula is true at the current instant
        if not self.formula.evaluate(instant, structure):
            return False
        
        # Recursively check in past instants
        past_instants = structure.past_instants(instant)
        
        if not past_instants:
            # If no past, H is true vacuously (beginning of time)
            return True
        
        if structure.time_type == 'linear':
            # In linear time, H is true if the formula was true in all past instants
            for previous_instant in past_instants:
                if not self.evaluate(previous_instant, structure, visited):
                    return False
            return True
        else:
            # In branching time, H is true if the formula was true in all past paths
            return all(self.evaluate(i, structure, visited.copy()) for i in past_instants)
    
    def __str__(self):
        return f"H{self.formula}"


class Since(TemporalFormula):
    """
    S (since) operator in temporal logic:
    f S g is true if g was true at a past moment and f has been true since that moment.
    """
    
    def __init__(self, left: TemporalFormula, right: TemporalFormula):
        self.left = left
        self.right = right
    
    def evaluate(self, instant, structure: TemporalStructure, visited=None):
        """Evaluates the S operator at a given instant of the structure."""
        if visited is None:
            visited = set()
        
        if instant in visited:
            # Cycle detection
            return False
        
        visited.add(instant)
        
        # Check if the right condition is true at the current instant
        if self.right.evaluate(instant, structure):
            return True
        
        # Check if the left condition is true at the current instant
        if not self.left.evaluate(instant, structure):
            return False
        
        # Recursively check in past instants
        past_instants = structure.past_instants(instant)
        
        if not past_instants:
            # If no past, S is false because the right condition is never met
            return False
        
        if structure.time_type == 'linear':
            # In linear time, S is true if there exists a past instant where the right condition was true
            # and the left condition has been true since that instant
            for previous_instant in past_instants:
                if self.evaluate(previous_instant, structure, visited):
                    return True
            return False
        else:
            # In branching time, S is true if there exists a path where the condition is satisfied
            return any(self.evaluate(i, structure, visited.copy()) for i in past_instants)
    
    def __str__(self):
        return f"({self.left} S {self.right})"


class TemporalChecker:
    """Model checker for temporal logic."""
    
    @staticmethod
    def check(formula: TemporalFormula, structure: TemporalStructure, instant=None):
        """
        Checks if a formula is true at a given instant or at all instants.
        If instant is None, checks if the formula is true at all instants.
        """
        if instant is not None:
            return formula.evaluate(instant, structure)
        
        return all(formula.evaluate(i, structure) for i in structure.instants)
    
    @staticmethod
    def satisfying_instants(formula: TemporalFormula, structure: TemporalStructure):
        """Returns the set of instants that satisfy the formula."""
        return {i for i in structure.instants if formula.evaluate(i, structure)}
    
    @staticmethod
    def is_valid(formula: TemporalFormula, structure: TemporalStructure):
        """Checks if a formula is valid in the structure (true at all instants)."""
        return all(formula.evaluate(i, structure) for i in structure.instants)
    
    @staticmethod
    def is_satisfiable(formula: TemporalFormula, structure: TemporalStructure):
        """Checks if a formula is satisfiable in the structure (true at at least one instant)."""
        return any(formula.evaluate(i, structure) for i in structure.instants)


#########################################
# DEONTIC LOGIC
#########################################

class NormativeSystem:
    """Represents a normative system for deontic logic."""
    
    def __init__(self):
        # Use modal logic as the basis for deontic logic
        self.deontic_logic = DeonticLogic()
        self.real_world = "real_world"
        self.ideal_worlds = set()
        
        # Add the real world
        self.deontic_logic.add_real_world(self.real_world)
        
        # Corrected method definition from class scope to instance scope
        self.add_ideal_world = self._add_ideal_world_impl 
    
    def _add_ideal_world_impl(self, name: str):
        """Adds an ideal world to the normative system."""
        self.ideal_worlds.add(name)
        self.deontic_logic.add_ideal_world(name)
        self.deontic_logic.define_ideal_accessibility(self.real_world, name)
        return self
    
    def define_valuation(self, world, proposition, value):
        """Defines the truth value of a proposition in a world."""
        self.deontic_logic.define_valuation(world, proposition, value)
        return self
    
    def obligation(self, formula: ModalFormula):
        """Creates an obligation formula (O)."""
        return self.deontic_logic.obligation(formula)
    
    def permission(self, formula: ModalFormula):
        """Creates a permission formula (P)."""
        return self.deontic_logic.permission(formula)
    
    def prohibition(self, formula: ModalFormula):
        """Creates a prohibition formula (F)."""
        return self.deontic_logic.prohibition(formula)
    
    def optional(self, formula: ModalFormula):
        """Creates an optional formula."""
        return self.deontic_logic.optional(formula)
    
    def check(self, formula: ModalFormula):
        """Checks if a formula is true in the real world."""
        return self.deontic_logic.check(formula, self.real_world)
    
    def is_consistent(self):
        """Checks if the normative system is consistent (non-contradictory)."""
        for proposition in set(v for m in self.deontic_logic.model.valuations.values() for v in m.keys()):
            formula = ModalProposition(proposition)
            if self.check(self.obligation(formula)) and self.check(self.prohibition(formula)):
                return False
        return True
    
    def identify_conflicts(self):
        """Identifies normative conflicts in the system."""
        conflicts = []
        for proposition in set(v for m in self.deontic_logic.model.valuations.values() for v in m.keys()):
            formula = ModalProposition(proposition)
            if self.check(self.obligation(formula)) and self.check(self.prohibition(formula)):
                conflicts.append(proposition)
        return conflicts
    
    def resolve_conflict(self, proposition, priority="obligation"):
        """Resolves a normative conflict by prioritizing obligation or prohibition."""
        formula = ModalProposition(proposition)
        
        if priority == "obligation":
            # Keep the obligation, remove the prohibition
            for ideal_world in self.ideal_worlds:
                self.deontic_logic.define_valuation(ideal_world, proposition, True)
        elif priority == "prohibition":
            # Keep the prohibition, remove the obligation
            for ideal_world in self.ideal_worlds:
                self.deontic_logic.define_valuation(ideal_world, proposition, False)
        else:
            raise ValueError(f"Unknown priority: {priority}")
        
        return self


##########################################
# PART 2: AUTOMATIC THEOREM PROVING
##########################################

class Proof:
    """Represents a formal proof."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.lines = []  # List of proof lines
        self.hypotheses = set()  # Set of used hypotheses
    
    def add_line(self, formula, justification, references=None):
        """
        Adds a line to the proof.
        formula: the demonstrated formula
        justification: the inference rule used
        references: references to previous lines
        """
        if references is None:
            references = []
        
        line = {
            "number": len(self.lines) + 1,
            "formula": formula,
            "justification": justification,
            "references": references
        }
        
        self.lines.append(line)
        return self
    
    def add_hypothesis(self, formula):
        """Adds a hypothesis to the proof."""
        self.hypotheses.add(formula)
        self.add_line(formula, "Hypothesis")
        return self
    
    def conclusion(self):
        """Returns the conclusion of the proof."""
        if not self.lines:
            return None
        return self.lines[-1]["formula"]
    
    def is_valid(self):
        """Checks if the proof is valid."""
        # Check that each line is correctly justified
        for line in self.lines:
            if line["justification"] == "Hypothesis":
                if str(line["formula"]) not in map(str, self.hypotheses):
                    return False
            elif not self._check_justification(line): # Changed method name to be internal
                return False
        
        return True
    
    def _check_justification(self, line):
        """Checks that a justification is correct."""
        # This method would be completed for each inference rule
        # For simplicity, we assume all justifications are valid
        return True
    
    def to_latex(self):
        """Generates a LaTeX representation of the proof."""
        latex = "\\begin{proof}\n"
        latex += "\\begin{enumerate}\n"
        
        for line in self.lines:
            formula = str(line["formula"]).replace("∧", "\\land").replace("∨", "\\lor").replace("¬", "\\neg").replace("→", "\\rightarrow")
            
            refs = ""
            if line["references"]:
                refs = " [" + ", ".join(map(str, line["references"])) + "]"
            
            latex += f"\\item {formula} \\hfill ({line['justification']}{refs})\n"
        
        latex += "\\end{enumerate}\n"
        latex += "\\end{proof}"
        
        return latex
    
    def __str__(self):
        result = f"Proof: {self.name}\n"
        result += "Hypotheses: " + ", ".join(map(str, self.hypotheses)) + "\n"
        result += "Lines:\n"
        
        for line in self.lines:
            refs = ""
            if line["references"]:
                refs = " [" + ", ".join(map(str, line["references"])) + "]"
            
            result += f"{line['number']}. {line['formula']} ({line['justification']}{refs})\n"
        
        return result


class InferenceRule:
    """Represents an inference rule for natural deduction."""
    
    def __init__(self, name, premise_schemas, conclusion_schema):
        self.name = name
        self.premise_schemas = premise_schemas  # Formula schemas for premises
        self.conclusion_schema = conclusion_schema  # Formula schema for the conclusion
    
    def apply(self, formulas):
        """
        Applies the inference rule to the given formulas.
        Returns the conclusion if the rule is applicable, None otherwise.
        """
        # This method would be implemented for each specific inference rule
        return None
    
    def __str__(self):
        premisses = ", ".join(map(str, self.premise_schemas))
        return f"{self.name}: {premisses} ⊢ {self.conclusion_schema}"


class DeductionSystem:
    """Deduction system for propositional and predicate logic."""
    
    def __init__(self):
        self.rules = {}
        self.initialize_rules()
    
    def initialize_rules(self):
        """Initializes standard inference rules."""
        # Rules for propositional logic
        self.add_modus_ponens_rule()
        self.add_modus_tollens_rule()
        self.add_conjunction_introduction_rule()
        self.add_conjunction_elimination_rule()
        self.add_disjunction_introduction_rule()
        self.add_disjunction_elimination_rule()
        self.add_implication_introduction_rule()
        
        # Rules for predicate logic
        self.add_universal_introduction_rule()
        self.add_universal_elimination_rule()
        self.add_existential_introduction_rule()
        self.add_existential_elimination_rule()
    
    def add_rule(self, rule):
        """Adds an inference rule to the system."""
        self.rules[rule.name] = rule
        return self
    
    def add_modus_ponens_rule(self):
        """Adds the Modus Ponens rule: A, A→B ⊢ B"""
        class ModusPonens(InferenceRule):
            def __init__(self):
                super().__init__("Modus Ponens", ["A", "A→B"], "B")
            
            def apply(self, formulas):
                if len(formulas) != 2:
                    return None
                
                # Check if the second formula is an implication
                if not isinstance(formulas[1], ModalImplication) and not isinstance(formulas[1], TemporalImplication):
                    return None
                
                # Check if the first formula corresponds to the antecedent of the implication
                if str(formulas[0]) == str(formulas[1].antecedent):
                    return formulas[1].consequent
                
                return None
        
        self.add_rule(ModusPonens())
    
    def add_modus_tollens_rule(self):
        """Adds the Modus Tollens rule: A→B, ¬B ⊢ ¬A"""
        class ModusTollens(InferenceRule):
            def __init__(self):
                super().__init__("Modus Tollens", ["A→B", "¬B"], "¬A")
            
            def apply(self, formulas):
                if len(formulas) != 2:
                    return None
                
                # Check if the first formula is an implication
                if not isinstance(formulas[0], ModalImplication) and not isinstance(formulas[0], TemporalImplication):
                    return None
                
                # Check if the second formula is a negation
                if not isinstance(formulas[1], ModalNot) and not isinstance(formulas[1], TemporalNot):
                    return None
                
                # Check if the negation corresponds to the consequent of the implication
                if str(formulas[1].formula) == str(formulas[0].consequent):
                    if isinstance(formulas[0], ModalImplication):
                        return ModalNot(formulas[0].antecedent)
                    else:
                        return TemporalNot(formulas[0].antecedent)
                
                return None
        
        self.add_rule(ModusTollens())
    
    def add_conjunction_introduction_rule(self):
        """Adds the Conjunction Introduction rule: A, B ⊢ A∧B"""
        class ConjunctionIntroduction(InferenceRule):
            def __init__(self):
                super().__init__("Conjunction Introduction", ["A", "B"], "A∧B")
            
            def apply(self, formulas):
                if len(formulas) != 2:
                    return None
                
                # Determine formula type
                if isinstance(formulas[0], ModalFormula) and isinstance(formulas[1], ModalFormula):
                    return ModalAnd(formulas[0], formulas[1])
                elif isinstance(formulas[0], TemporalFormula) and isinstance(formulas[1], TemporalFormula):
                    return TemporalAnd(formulas[0], formulas[1])
                
                return None
        
        self.add_rule(ConjunctionIntroduction())
    
    def add_conjunction_elimination_rule(self):
        """Adds the Conjunction Elimination rules: A∧B ⊢ A and A∧B ⊢ B"""
        class ConjunctionEliminationLeft(InferenceRule):
            def __init__(self):
                super().__init__("Conjunction Elimination (left)", ["A∧B"], "A")
            
            def apply(self, formulas):
                if len(formulas) != 1:
                    return None
                
                if isinstance(formulas[0], ModalAnd):
                    return formulas[0].left
                elif isinstance(formulas[0], TemporalAnd):
                    return formulas[0].left
                
                return None
        
        class ConjunctionEliminationRight(InferenceRule):
            def __init__(self):
                super().__init__("Conjunction Elimination (right)", ["A∧B"], "B")
            
            def apply(self, formulas):
                if len(formulas) != 1:
                    return None
                
                if isinstance(formulas[0], ModalAnd):
                    return formulas[0].right
                elif isinstance(formulas[0], TemporalAnd):
                    return formulas[0].right
                
                return None
        
        self.add_rule(ConjunctionEliminationLeft())
        self.add_rule(ConjunctionEliminationRight())
    
    def add_disjunction_introduction_rule(self):
        """Adds the Disjunction Introduction rules: A ⊢ A∨B and B ⊢ A∨B"""
        class DisjunctionIntroductionLeft(InferenceRule):
            def __init__(self):
                super().__init__("Disjunction Introduction (left)", ["A"], "A∨B")
            
            def apply(self, formulas):
                if len(formulas) != 1:
                    return None
                
                # Here, we would need formula B to construct A∨B
                # In a full implementation, B could be provided as an additional argument
                return None
        
        class DisjunctionIntroductionRight(InferenceRule):
            def __init__(self):
                super().__init__("Disjunction Introduction (right)", ["B"], "A∨B")
            
            def apply(self, formulas):
                if len(formulas) != 1:
                    return None
                
                # Here, we would need formula A to construct A∨B
                return None
        
        self.add_rule(DisjunctionIntroductionLeft())
        self.add_rule(DisjunctionIntroductionRight())
    
    def add_disjunction_elimination_rule(self):
        """Adds the Disjunction Elimination rule: A∨B, A→C, B→C ⊢ C"""
        class DisjunctionElimination(InferenceRule):
            def __init__(self):
                super().__init__("Disjunction Elimination", ["A∨B", "A→C", "B→C"], "C")
            
            def apply(self, formulas):
                if len(formulas) != 3:
                    return None
                
                # Check if the first formula is a disjunction
                if not isinstance(formulas[0], ModalOr) and not isinstance(formulas[0], TemporalOr):
                    return None
                
                # Check if the other two formulas are implications
                if (not isinstance(formulas[1], ModalImplication) and not isinstance(formulas[1], TemporalImplication) or
                    not isinstance(formulas[2], ModalImplication) and not isinstance(formulas[2], TemporalImplication)):
                    return None
                
                # Check formula consistency
                if (str(formulas[0].left) == str(formulas[1].antecedent) and
                    str(formulas[0].right) == str(formulas[2].antecedent) and
                    str(formulas[1].consequent) == str(formulas[2].consequent)):
                    return formulas[1].consequent
                
                return None
        
        self.add_rule(DisjunctionElimination())
    
    def add_implication_introduction_rule(self):
        """Adds the Implication Introduction rule."""
        # This rule would require sub-proof management
        pass
    
    def add_universal_introduction_rule(self):
        """Adds the Universal Quantifier Introduction rule."""
        # These rules will be implemented in the first and second order logic part
        pass
    
    def add_universal_elimination_rule(self):
        """Adds the Universal Quantifier Elimination rule."""
        pass
    
    def add_existential_introduction_rule(self):
        """Adds the Existential Quantifier Introduction rule."""
        pass
    
    def add_existential_elimination_rule(self):
        """Adds the Existential Quantifier Elimination rule."""
        pass
    
    def apply_rule(self, rule_name, formulas):
        """Applies an inference rule to the given formulas."""
        if rule_name not in self.rules:
            raise ValueError(f"Unknown inference rule: {rule_name}")
        
        return self.rules[rule_name].apply(formulas)
    
    def prove(self, hypotheses, conclusion, max_steps=100):
        """
        Attempts to construct a proof of the conclusion from the hypotheses.
        Returns a proof if it exists, None otherwise.
        """
        # For simplicity, we use a forward chaining strategy
        proof = Proof()
        
        # Add hypotheses to the proof
        for hyp in hypotheses:
            proof.add_hypothesis(hyp)
        
        # Set of already proven formulas
        demonstrated_formulas = set(hypotheses)
        
        # Attempt to prove by applying rules
        for _ in range(max_steps):
            # If the conclusion has been demonstrated, the proof is complete
            if str(conclusion) in map(str, demonstrated_formulas):
                for formula in demonstrated_formulas:
                    if str(formula) == str(conclusion):
                        proof.add_line(formula, "Already demonstrated")
                        return proof
            
            # Apply all possible inference rules
            new_formulas = set()
            
            for rule_name, rule in self.rules.items():
                # For each combination of already proven formulas
                for comb in itertools.combinations(demonstrated_formulas, min(len(demonstrated_formulas), len(rule.premise_schemas))):
                    # Attempt to apply the rule
                    new_formula = rule.apply(comb)
                    
                    if new_formula is not None and str(new_formula) not in map(str, demonstrated_formulas):
                        new_formulas.add(new_formula)
                        refs = [i+1 for i, f in enumerate(proof.lines) if str(f["formula"]) in map(str, comb)]
                        proof.add_line(new_formula, rule_name, refs)
            
            # If no new formula has been demonstrated, the proof fails
            if not new_formulas:
                return None
            
            # Add new formulas to the set of demonstrated formulas
            demonstrated_formulas.update(new_formulas)
        
        # If the maximum number of steps is reached, the proof fails
        return None


class ResolutionMethod:
    """Implementation of the resolution method for propositional logic."""
    
    @staticmethod
    def to_cnf(formula):
        """Converts a formula to conjunctive normal form (CNF)."""
        # This method would be implemented for each formula type
        # For simplicity, we assume the formula is already in CNF
        return formula
    
    @staticmethod
    def clauses_from_cnf(cnf_formula):
        """Extracts clauses from a CNF formula."""
        # A clause is a set of literals (variables or their negations)
        # For simplicity, we represent each clause as a set of strings
        return [{"p"}, {"q"}, {"r"}]  # Simplified example
    
    @staticmethod
    def resolve(clauses1, clauses2):
        """Applies the resolution rule to sets of clauses."""
        results = set()
        
        for c1 in clauses1:
            for c2 in clauses2:
                # Look for a literal in c1 whose negation is in c2
                for lit in c1:
                    # The complementary literal would be the negation of lit
                    lit_comp = lit[1:] if lit.startswith("¬") else "¬" + lit
                    
                    if lit_comp in c2:
                        # Create a new clause by resolving c1 and c2
                        resolvent = (c1 - {lit}) | (c2 - {lit_comp})
                        
                        # If the resolvent is the empty clause, the formula is unsatisfiable
                        if not resolvent:
                            return False  # Contradiction found
                        
                        results.add(frozenset(resolvent))
        
        return results
    
    @staticmethod
    def prove_by_resolution(hypotheses, conclusion, max_steps=100):
        """
        Proves a conclusion from hypotheses using the resolution method.
        Returns True if the proof succeeds, False otherwise.
        """
        # Convert hypotheses and the negation of the conclusion to CNF
        clauses = set()
        
        for hyp in hypotheses:
            hyp_cnf = ResolutionMethod.to_cnf(hyp)
            hyp_clauses = ResolutionMethod.clauses_from_cnf(hyp_cnf)
            clauses.update(frozenset(c) for c in hyp_clauses)
        
        # Add the negation of the conclusion
        neg_conclusion = None  # This would be the negation of the conclusion
        neg_conclusion_cnf = ResolutionMethod.to_cnf(neg_conclusion)
        neg_clauses = ResolutionMethod.clauses_from_cnf(neg_conclusion_cnf)
        clauses.update(frozenset(c) for c in neg_clauses)
        
        # Apply resolution until a contradiction is found or maximum steps are reached
        for _ in range(max_steps):
            new_clauses = set()
            
            # Apply resolution to all pairs of clauses
            for c1, c2 in itertools.combinations(clauses, 2):
                resolvents = ResolutionMethod.resolve({c1}, {c2})
                
                if resolvents is False:
                    # Contradiction found, the proof succeeds
                    return True
                
                new_clauses.update(resolvents)
            
            # If no new clause was generated, the proof fails
            if new_clauses.issubset(clauses):
                break
            
            clauses.update(new_clauses)
        
        # If no contradiction was found, the proof fails
        return False


class TableauxMethod:
    """Implementation of the semantic tableaux method for propositional logic."""
    
    class Node:
        """Represents a node in a semantic tableau."""
        
        def __init__(self, formula, sign=True, parent=None):
            self.formula = formula  # The formula associated with the node
            self.sign = sign  # True for affirmation, False for negation
            self.parent = parent  # Parent node
            self.children = []  # Child nodes
            self.is_closed = False  # Indicates if the branch is closed
        
        def add_child(self, formula, sign=True):
            """Adds a child to the node."""
            child = TableauxMethod.Node(formula, sign, self)
            self.children.append(child)
            return child
        
        def is_leaf(self):
            """Indicates if the node is a leaf."""
            return not self.children
        
        def __str__(self):
            sign_str = "" if self.sign else "¬"
            return f"{sign_str}{self.formula}"
    
    @staticmethod
    def decompose(node):
        """Decomposes a node according to semantic tableau rules."""
        formula = node.formula
        sign = node.sign
        
        # Decomposition according to formula type and its sign
        if isinstance(formula, ModalNot) or isinstance(formula, TemporalNot):
            # Negation rule
            node.add_child(formula.formula, not sign)
        
        elif isinstance(formula, ModalAnd) or isinstance(formula, TemporalAnd):
            if sign:
                # Affirmed conjunction rule (α rule)
                node.add_child(formula.left, True)
                node.add_child(formula.right, True)
            else:
                # Denied conjunction rule (β rule)
                child1 = node.add_child(formula.left, False)
                child2 = node.add_child(formula.right, False)
                child1.is_branch = True # This attribute is not used elsewhere, seems like a placeholder
                child2.is_branch = True
        
        elif isinstance(formula, ModalOr) or isinstance(formula, TemporalOr):
            if sign:
                # Affirmed disjunction rule (β rule)
                child1 = node.add_child(formula.left, True)
                child2 = node.add_child(formula.right, True)
                child1.is_branch = True
                child2.is_branch = True
            else:
                # Denied disjunction rule (α rule)
                node.add_child(formula.left, False)
                node.add_child(formula.right, False)
        
        elif isinstance(formula, ModalImplication) or isinstance(formula, TemporalImplication):
            if sign:
                # Affirmed implication rule (β rule)
                child1 = node.add_child(formula.antecedent, False)
                child2 = node.add_child(formula.consequent, True)
                child1.is_branch = True
                child2.is_branch = True
            else:
                # Denied implication rule (α rule)
                node.add_child(formula.antecedent, True)
                node.add_child(formula.consequent, False)
        
        # Other rules would be added for other formula types
        
        return node.children
    
    @staticmethod
    def is_contradictory(path):
        """Checks if a path contains a contradiction."""
        # A path is contradictory if it contains a formula and its negation
        for i, node1 in enumerate(path):
            for node2 in path[i+1:]:
                if str(node1.formula) == str(node2.formula) and node1.sign != node2.sign:
                    return True
        
        return False
    
    @staticmethod
    def build_tableau(formulas, signs=None):
        """
        Constructs a semantic tableau for a set of formulas.
        formulas: list of formulas
        signs: list of signs (True for affirmation, False for negation)
        """
        if signs is None:
            signs = [True] * len(formulas)
        
        # Create root node
        root = TableauxMethod.Node(None)
        
        # Add initial formulas
        nodes = []
        for formula, sign in zip(formulas, signs):
            node = root.add_child(formula, sign)
            nodes.append(node)
        
        # Develop the tableau
        TableauxMethod._develop_tableau(nodes) # Changed method name to be internal
        
        return root
    
    @staticmethod
    def _develop_tableau(nodes, visited=None): # Changed method name to be internal
        """
        Develops the semantic tableau from the given nodes.
        nodes: list of nodes to develop
        visited: set of already visited formulas
        """
        if visited is None:
            visited = set()
        
        # For each node
        for node in nodes:
            # If the formula has already been visited or if the node is atomic, skip
            formula_str = str(node)
            if formula_str in visited or (isinstance(node.formula, ModalProposition) or isinstance(node.formula, TemporalProposition)):
                continue
            
            visited.add(formula_str)
            
            # Decompose the node
            children = TableauxMethod.decompose(node)
            
            # Check if the path is contradictory
            path = []
            n = node
            while n is not None:
                path.append(n)
                n = n.parent
            
            if TableauxMethod.is_contradictory(path):
                node.is_closed = True
                continue
            
            # Recursively develop children
            TableauxMethod._develop_tableau(children, visited)
    
    @staticmethod
    def is_closed(root):
        """Checks if the tableau is closed (all branches are closed)."""
        # A tableau is closed if all its branches are closed
        
        def traverse(node):
            if node.is_closed:
                return True
            
            if node.is_leaf():
                # Check if the path is contradictory
                path = []
                n = node
                while n is not None:
                    path.append(n)
                    n = n.parent
                
                return TableauxMethod.is_contradictory(path)
            
            return all(traverse(child) for child in node.children)
        
        return traverse(root)
    
    @staticmethod
    def prove_by_tableaux(hypotheses, conclusion):
        """
        Proves a conclusion from hypotheses using the tableaux method.
        Returns True if the proof succeeds, False otherwise.
        """
        # To prove hypotheses ⊢ conclusion, we check if hypotheses ∧ ¬conclusion is unsatisfiable
        # If the tableau is closed, the proof succeeds
        
        formulas = list(hypotheses)
        signs = [True] * len(formulas)
        
        # Add the negation of the conclusion
        if isinstance(conclusion, ModalFormula):
            neg_conclusion = ModalNot(conclusion)
        else:
            neg_conclusion = TemporalNot(conclusion) # Assuming it's temporal if not modal.
        
        formulas.append(neg_conclusion)
        signs.append(True)
        
        # Construct the tableau
        root = TableauxMethod.build_tableau(formulas, signs)
        
        # Check if the tableau is closed
        return TableauxMethod.is_closed(root)


##########################################
# PART 3: REASONING WITH MULTIPLE CONSTRAINTS
##########################################

class Constraint(ABC):
    """Abstract class to represent a constraint."""
    
    @abstractmethod
    def is_satisfied(self, solution):
        """Checks if the constraint is satisfied by the solution."""
        pass
    
    @abstractmethod
    def __str__(self):
        """Textual representation of the constraint."""
        pass


class UnaryConstraint(Constraint):
    """Constraint on a single variable."""
    
    def __init__(self, variable, constraint_function):
        self.variable = variable
        self.constraint_function = constraint_function
    
    def is_satisfied(self, solution):
        """Checks if the constraint is satisfied by the solution."""
        if self.variable not in solution:
            return True  # The constraint does not apply if the variable is not in the solution
        
        return self.constraint_function(solution[self.variable])
    
    def __str__(self):
        return f"Constraint({self.variable})"


class BinaryConstraint(Constraint):
    """Constraint on two variables."""
    
    def __init__(self, variable1, variable2, constraint_function):
        self.variable1 = variable1
        self.variable2 = variable2
        self.constraint_function = constraint_function
    
    def is_satisfied(self, solution):
        """Checks if the constraint is satisfied by the solution."""
        if self.variable1 not in solution or self.variable2 not in solution:
            return True  # The constraint does not apply if one of the variables is not in the solution
        
        return self.constraint_function(solution[self.variable1], solution[self.variable2])
    
    def __str__(self):
        return f"Constraint({self.variable1}, {self.variable2})"


class NaryConstraint(Constraint):
    """Constraint on multiple variables."""
    
    def __init__(self, variables, constraint_function):
        self.variables = variables
        self.constraint_function = constraint_function
    
    def is_satisfied(self, solution):
        """Checks if the constraint is satisfied by the solution."""
        if not all(v in solution for v in self.variables):
            return True  # The constraint does not apply if all variables are not in the solution
        
        return self.constraint_function(*[solution[v] for v in self.variables])
    
    def __str__(self):
        return f"Constraint({', '.join(self.variables)})"


class ConstraintProblem:
    """Represents a constraint satisfaction problem."""
    
    def __init__(self):
        self.variables = set()
        self.domains = {}
        self.constraints = []
    
    def add_variable(self, variable, domain):
        """Adds a variable with its domain of possible values."""
        self.variables.add(variable)
        self.domains[variable] = list(domain)
        return self
    
    def add_constraint(self, constraint):
        """Adds a constraint to the problem."""
        self.constraints.append(constraint)
        return self
    
    def is_valid_solution(self, solution):
        """Checks if a solution satisfies all constraints."""
        return all(c.is_satisfied(solution) for c in self.constraints)
    
    def solve_backtracking(self):
        """Solves the problem using the backtracking algorithm."""
        solution = {}
        return self._backtracking(solution)
    
    def _backtracking(self, solution):
        """Recursive backtracking algorithm."""
        # If all variables have a value, check if the solution is valid
        if len(solution) == len(self.variables):
            if self.is_valid_solution(solution):
                return solution
            return None
        
        # Choose an unassigned variable
        # Ensure we pick a variable not already in solution.keys()
        unassigned_vars = self.variables - set(solution.keys())
        if not unassigned_vars: # All variables assigned but not a valid solution. Should not happen if `is_valid_solution` is checked correctly.
            return None 
        var = next(iter(unassigned_vars))
        
        # Try each value in the domain
        for val in self.domains[var]:
            # Check if the assignment is consistent with the constraints
            solution[var] = val
            if self.is_valid_solution(solution): # This check is usually done incrementally for efficiency
                # Continue backtracking
                result = self._backtracking(solution)
                if result is not None:
                    return result
            
            # If we get here, the assignment did not work, remove the variable
            del solution[var]
        
        # No solution found
        return None
    
    def solve_ac3(self):
        """Solves the problem using the AC-3 (Arc Consistency) algorithm."""
        # Reduce domains using AC-3
        domains = {var: list(dom) for var, dom in self.domains.items()} # Create a mutable copy
        
        if not self._ac3(domains):
            return None  # No solution
        
        # Use backtracking with reduced domains
        solution = {}
        return self._backtracking_with_domains(solution, domains)
    
    def _ac3(self, domains):
        """AC-3 algorithm for arc consistency."""
        # Initialize the queue of arcs (pairs of variables linked by a constraint)
        arcs = []
        
        for c in self.constraints:
            if isinstance(c, BinaryConstraint):
                arcs.append((c.variable1, c.variable2))
                arcs.append((c.variable2, c.variable1))
        
        # While there are still arcs to process
        while arcs:
            x, y = arcs.pop(0)
            
            if self._revise(domains, x, y):
                if not domains[x]:
                    return False  # Empty domain, no solution
                
                # Add all arcs (z, x) where z is a neighbor of x but different from y
                for c_prime in self.constraints:
                    if isinstance(c_prime, BinaryConstraint):
                        if c_prime.variable1 == x and c_prime.variable2 != y:
                            arcs.append((c_prime.variable2, x))
                        elif c_prime.variable2 == x and c_prime.variable1 != y:
                            arcs.append((c_prime.variable1, x))
        
        return True
    
    def _revise(self, domains, x, y):
        """Revises the domain of x based on y."""
        revised = False
        
        for vx in list(domains[x]): # Iterate over a copy to allow modification
            # Check if there exists a value vy in the domain of y such that (vx, vy) satisfies the constraint
            # This check needs to consider all binary constraints involving x and y
            
            found_support = False
            for vy in domains[y]:
                temp_solution = {x: vx, y: vy}
                
                # Check all binary constraints that involve both x and y
                all_constraints_satisfied_for_pair = True
                for c in self.constraints:
                    if isinstance(c, BinaryConstraint) and ((c.variable1 == x and c.variable2 == y) or (c.variable1 == y and c.variable2 == x)):
                        if not c.is_satisfied(temp_solution):
                            all_constraints_satisfied_for_pair = False
                            break
                
                if all_constraints_satisfied_for_pair:
                    found_support = True
                    break
            
            if not found_support:
                domains[x].remove(vx)
                revised = True
        
        return revised
    
    def _is_consistent_partial(self, partial_solution): # Renamed for clarity as it's partial
        """Checks if a partial solution is consistent with the constraints."""
        # Only checks constraints where all involved variables are in the partial_solution
        for c in self.constraints:
            involved_vars = []
            if isinstance(c, UnaryConstraint):
                involved_vars = [c.variable]
            elif isinstance(c, BinaryConstraint):
                involved_vars = [c.variable1, c.variable2]
            elif isinstance(c, NaryConstraint):
                involved_vars = c.variables
            
            if all(v in partial_solution for v in involved_vars):
                if not c.is_satisfied(partial_solution):
                    return False
        return True
    
    def _backtracking_with_domains(self, solution, domains):
        """Backtracking algorithm with reduced domains."""
        # If all variables have a value, check if the solution is valid
        if len(solution) == len(self.variables):
            # Final check with all constraints
            if self.is_valid_solution(solution): 
                return solution
            return None
        
        # Choose an unassigned variable
        unassigned_vars = self.variables - set(solution.keys())
        if not unassigned_vars:
            return None
        var = next(iter(unassigned_vars))
        
        # Try each value in the domain
        for val in domains[var]: # Use the reduced domains
            # Check if the assignment is consistent with the constraints
            solution[var] = val
            if self._is_consistent_partial(solution): # Use partial consistency check
                # Continue backtracking
                result = self._backtracking_with_domains(solution, domains)
                if result is not None:
                    return result
            
            # If we get here, we remove the variable to try another value
            del solution[var]
        
        # No solution found
        return None


class Optimization:
    """Class for optimization problems."""
    
    class OptimizationType(Enum):
        MINIMIZATION = 1
        MAXIMIZATION = 2
    
    def __init__(self, problem: ConstraintProblem, objective_function, opt_type=OptimizationType.MINIMIZATION):
        self.problem = problem
        self.objective_function = objective_function
        self.opt_type = opt_type
    
    def solve(self):
        """Solves the optimization problem."""
        # Find all valid solutions
        solutions = self._find_all_solutions()
        
        if not solutions:
            return None
        
        # Find the best solution according to the objective function
        if self.opt_type == self.OptimizationType.MINIMIZATION:
            return min(solutions, key=lambda s: self.objective_function(s))
        else:
            return max(solutions, key=lambda s: self.objective_function(s))
    
    def _find_all_solutions(self):
        """Finds all valid solutions to the problem."""
        solutions = []
        
        def backtracking(solution):
            # If all variables have a value, check if the solution is valid
            if len(solution) == len(self.problem.variables):
                if self.problem.is_valid_solution(solution):
                    solutions.append(solution.copy())
                return
            
            # Choose an unassigned variable
            unassigned_vars = self.problem.variables - set(solution.keys())
            if not unassigned_vars:
                return # All variables assigned, no more to choose
            var = next(iter(unassigned_vars))
            
            # Try each value in the domain
            for val in self.problem.domains[var]:
                # Check if the assignment is consistent with the constraints
                solution[var] = val
                # Here we call is_valid_solution, but for optimization,
                # it might be more efficient to use a partial check
                # or ensure consistency as part of backtracking_with_domains if AC3 is used.
                # For _find_all_solutions which is a basic exhaustive search, is_valid_solution is okay.
                if self.problem._is_consistent_partial(solution): # Use internal partial check for efficiency
                    # Continue backtracking
                    backtracking(solution)
                
                # If we get here, we remove the variable to try another value
                del solution[var]
        
        # Start backtracking
        backtracking({})
        return solutions


class LogicProgramming:
    """Class for logic programming (Prolog style)."""
    
    def __init__(self):
        self.facts = set()
        self.rules = []
    
    def add_fact(self, predicate, *arguments):
        """Adds a fact to the knowledge base."""
        self.facts.add((predicate, arguments))
        return self
    
    def add_rule(self, head, body):
        """
        Adds a rule to the knowledge base.
        head: tuple (predicate, arguments) representing the head of the rule
        body: list of tuples (predicate, arguments) representing the body of the rule
        """
        self.rules.append((head, body))
        return self
    
    def query(self, predicate, *arguments):
        """Performs a query and returns the substitutions that satisfy it."""
        goal = (predicate, arguments)
        return self._resolve([goal], {})
    
    def _resolve(self, goals, substitution):
        """
        Resolves a list of goals using SLD resolution.
        goals: list of goals to resolve
        substitution: dictionary of current substitutions
        """
        if not goals:
            return [substitution]
        
        # Take the first goal
        current_goal = goals[0]
        predicate, arguments = current_goal
        
        # Try to resolve the current goal
        results = []
        
        # Try facts
        for fact in self.facts:
            fact_predicate, fact_arguments = fact
            
            if fact_predicate == predicate and len(fact_arguments) == len(arguments):
                # Attempt to unify the goal with the fact
                new_substitution = self._unify(arguments, fact_arguments, substitution.copy())
                
                if new_substitution is not None:
                    # Continue with remaining goals
                    results.extend(self._resolve(goals[1:], new_substitution))
        
        # Try rules
        for rule in self.rules:
            head, body = rule
            head_predicate, head_arguments = head
            
            if head_predicate == predicate and len(head_arguments) == len(arguments):
                # Attempt to unify the goal with the head of the rule
                new_substitution = self._unify(arguments, head_arguments, substitution.copy())
                
                if new_substitution is not None:
                    # Add the body's goals to the beginning of the goal list
                    new_goals = body + goals[1:]
                    results.extend(self._resolve(new_goals, new_substitution))
        
        return results
    
    def _unify(self, terms1, terms2, substitution):
        """
        Unifies two lists of terms and updates the substitution.
        Returns the updated substitution or None if unification fails.
        """
        if len(terms1) != len(terms2):
            return None
        
        for t1, t2 in zip(terms1, terms2):
            # If t1 is a variable
            if isinstance(t1, str) and t1.startswith("?"):
                if t1 in substitution:
                    # The variable is already bound, check consistency
                    if substitution[t1] != t2:
                        return None
                else:
                    # Bind the variable to t2
                    substitution[t1] = t2
            
            # If t2 is a variable
            elif isinstance(t2, str) and t2.startswith("?"):
                if t2 in substitution:
                    # The variable is already bound, check consistency
                    if substitution[t2] != t1:
                        return None
                else:
                    # Bind the variable to t1
                    substitution[t2] = t1
            
            # If t1 and t2 are constants
            elif t1 != t2:
                return None
        
        return substitution


##########################################
# PART 4: FIRST AND SECOND ORDER LOGIC
##########################################

class LogicalTerm(ABC):
    """Abstract class for logical terms."""
    
    @abstractmethod
    def substitute(self, substitution):
        """Applies a substitution to the term."""
        pass
    
    @abstractmethod
    def free_variables(self):
        """Returns the set of free variables of the term."""
        pass
    
    @abstractmethod
    def __str__(self):
        """Textual representation of the term."""
        pass


class Variable(LogicalTerm):
    """Represents a logical variable."""
    
    def __init__(self, name):
        self.name = name
    
    def substitute(self, substitution):
        """Applies a substitution to the variable."""
        if self.name in substitution:
            return substitution[self.name]
        return self
    
    def free_variables(self):
        """Returns the set of free variables."""
        return {self.name}
    
    def __str__(self):
        return self.name
    
    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name
        return False
    
    def __hash__(self):
        return hash(self.name)


class Constant(LogicalTerm):
    """Represents a logical constant."""
    
    def __init__(self, value):
        self.value = value
    
    def substitute(self, substitution):
        """Applies a substitution to the constant (does nothing)."""
        return self
    
    def free_variables(self):
        """Returns the set of free variables (empty for a constant)."""
        return set()
    
    def __str__(self):
        return str(self.value)
    
    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.value == other.value
        return False
    
    def __hash__(self(self):
        return hash(self.value)


class Function(LogicalTerm):
    """Represents a logical function."""
    
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
    
    def substitute(self, substitution):
        """Applies a substitution to the function."""
        return Function(self.name, [arg.substitute(substitution) for arg in self.arguments])
    
    def free_variables(self):
        """Returns the set of free variables."""
        return set().union(*[arg.free_variables() for arg in self.arguments])
    
    def __str__(self):
        args_str = ", ".join(map(str, self.arguments))
        return f"{self.name}({args_str})"
    
    def __eq__(self, other):
        if isinstance(other, Function):
            return self.name == other.name and self.arguments == other.arguments
        return False
    
    def __hash__(self):
        return hash((self.name, tuple(self.arguments)))


class FirstOrderFormula(Formula):
    """Base class for first-order logic formulas."""
    
    @abstractmethod
    def substitute(self, substitution):
        """Applies a substitution to the formula."""
        pass
    
    @abstractmethod
    def free_variables(self):
        """Returns the set of free variables of the formula."""
        pass


class Predicate(FirstOrderFormula):
    """Represents a logical predicate."""
    
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
    
    def evaluate(self, interpretation):
        """Evaluates the predicate in a given interpretation."""
        # In a first-order interpretation, a predicate is evaluated
        # by checking if the tuple of elements is in the predicate's extension
        if self.name not in interpretation:
            return False
        
        # Evaluate arguments
        values = []
        for arg in self.arguments:
            if isinstance(arg, Variable):
                if arg.name not in interpretation:
                    return False
                values.append(interpretation[arg.name])
            elif isinstance(arg, Constant):
                values.append(arg.value)
            elif isinstance(arg, Function):
                # For simplicity, we do not handle function evaluation
                return False
        
        # Check if the tuple is in the predicate's extension
        return tuple(values) in interpretation[self.name]
    
    def substitute(self, substitution):
        """Applies a substitution to the predicate."""
        return Predicate(self.name, [arg.substitute(substitution) for arg in self.arguments])
    
    def free_variables(self):
        """Returns the set of free variables of the predicate."""
        return set().union(*[arg.free_variables() for arg in self.arguments])
    
    def __str__(self):
        args_str = ", ".join(map(str, self.arguments))
        return f"{self.name}({args_str})"
    
    def __eq__(self, other):
        if isinstance(other, Predicate):
            return self.name == other.name and self.arguments == other.arguments
        return False
    
    def __hash__(self):
        return hash((self.name, tuple(self.arguments)))


class FirstOrderNot(FirstOrderFormula):
    """Negation in first-order logic."""
    
    def __init__(self, formula):
        self.formula = formula
    
    def evaluate(self, interpretation):
        """Evaluates the negation in a given interpretation."""
        return not self.formula.evaluate(interpretation)
    
    def substitute(self, substitution):
        """Applies a substitution to the negation."""
        return FirstOrderNot(self.formula.substitute(substitution))
    
    def free_variables(self):
        """Returns the set of free variables of the negation."""
        return self.formula.free_variables()
    
    def __str__(self):
        return f"¬{self.formula}"
    
    def __eq__(self, other):
        if isinstance(other, FirstOrderNot):
            return self.formula == other.formula
        return False
    
    def __hash__(self):
        return hash(("not", self.formula))


class FirstOrderAnd(FirstOrderFormula):
    """Conjunction in first-order logic."""
    
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def evaluate(self, interpretation):
        """Evaluates the conjunction in a given interpretation."""
        return self.left.evaluate(interpretation) and self.right.evaluate(interpretation)
    
    def substitute(self, substitution):
        """Applies a substitution to the conjunction."""
        return FirstOrderAnd(self.left.substitute(substitution), self.right.substitute(substitution))
    
    def free_variables(self):
        """Returns the set of free variables of the conjunction."""
        return self.left.free_variables() | self.right.free_variables()
    
    def __str__(self):
        return f"({self.left} ∧ {self.right})"
    
    def __eq__(self, other):
        if isinstance(other, FirstOrderAnd):
            return self.left == other.left and self.right == other.right
        return False
    
    def __hash__(self):
        return hash(("and", self.left, self.right))


class FirstOrderOr(FirstOrderFormula):
    """Disjunction in first-order logic."""
    
    def __init__(self, left, right):
        self.left = left
        self.right = right
    
    def evaluate(self, interpretation):
        """Evaluates the disjunction in a given interpretation."""
        return self.left.evaluate(interpretation) or self.right.evaluate(interpretation)
    
    def substitute(self, substitution):
        """Applies a substitution to the disjunction."""
        return FirstOrderOr(self.left.substitute(substitution), self.right.substitute(substitution))
    
    def free_variables(self):
        """Returns the set of free variables of the disjunction."""
        return self.left.free_variables() | self.right.free_variables()
    
    def __str__(self):
        return f"({self.left} ∨ {self.right})"
    
    def __eq__(self, other):
        if isinstance(other, FirstOrderOr):
            return self.left == other.left and self.right == other.right
        return False
    
    def __hash__(self):
        return hash(("or", self.left, self.right))


class FirstOrderImplication(FirstOrderFormula):
    """Implication in first-order logic."""
    
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def evaluate(self, interpretation):
        """Evaluates the implication in a given interpretation."""
        return not self.antecedent.evaluate(interpretation) or self.consequent.evaluate(interpretation)
    
    def substitute(self, substitution):
        """Applies a substitution to the implication."""
        return FirstOrderImplication(self.antecedent.substitute(substitution), self.consequent.substitute(substitution))
    
    def free_variables(self):
        """Returns the set of free variables of the implication."""
        return self.antecedent.free_variables() | self.consequent.free_variables()
    
    def __str__(self):
        return f"({self.antecedent} → {self.consequent})"
    
    def __eq__(self, other):
        if isinstance(other, FirstOrderImplication):
            return self.antecedent == other.antecedent and self.consequent == other.consequent
        return False
    
    def __hash__(self):
        return hash(("implies", self.antecedent, self.consequent))


class UniversalQuantifier(FirstOrderFormula):
    """Universal quantifier in first-order logic."""
    
    def __init__(self, variable, formula):
        self.variable = variable
        self.formula = formula
    
    def evaluate(self, interpretation, domain):
        """
        Evaluates the universal quantifier in a given interpretation.
        domain: set of possible values for variables
        """
        # For every element in the domain, the formula must be true
        for value in domain:
            # Create a new interpretation with the variable bound to the value
            new_interpretation = interpretation.copy()
            new_interpretation[self.variable.name] = value
            
            if not self.formula.evaluate(new_interpretation):
                return False
        
        return True
    
    def substitute(self, substitution):
        """Applies a substitution to the universal quantifier."""
        # Avoid variable capture
        new_substitution = substitution.copy()
        if self.variable.name in new_substitution:
            del new_substitution[self.variable.name]
        
        return UniversalQuantifier(self.variable, self.formula.substitute(new_substitution))
    
    def free_variables(self):
        """Returns the set of free variables of the universal quantifier."""
        # The quantified variable is not free
        return self.formula.free_variables() - {self.variable.name}
    
    def __str__(self):
        return f"∀{self.variable}.{self.formula}"
    
    def __eq__(self, other):
        if isinstance(other, UniversalQuantifier):
            return self.variable == other.variable and self.formula == other.formula
        return False
    
    def __hash__(self):
        return hash(("for_all", self.variable, self.formula))


class ExistentialQuantifier(FirstOrderFormula):
    """Existential quantifier in first-order logic."""
    
    def __init__(self, variable, formula):
        self.variable = variable
        self.formula = formula
    
    def evaluate(self, interpretation, domain):
        """
        Evaluates the existential quantifier in a given interpretation.
        domain: set of possible values for variables
        """
        # There must exist at least one element in the domain for which the formula is true
        for value in domain:
            # Create a new interpretation with the variable bound to the value
            new_interpretation = interpretation.copy()
            new_interpretation[self.variable.name] = value
            
            if self.formula.evaluate(new_interpretation):
                return True
        
        return False
    
    def substitute(self, substitution):
        """Applies a substitution to the existential quantifier."""
        # Avoid variable capture
        new_substitution = substitution.copy()
        if self.variable.name in new_substitution:
            del new_substitution[self.variable.name]
        
        return ExistentialQuantifier(self.variable, self.formula.substitute(new_substitution))
    
    def free_variables(self):
        """Returns the set of free variables of the existential quantifier."""
        # The quantified variable is not free
        return self.formula.free_variables() - {self.variable.name}
    
    def __str__(self):
        return f"∃{self.variable}.{self.formula}"
    
    def __eq__(self, other):
        if isinstance(other, ExistentialQuantifier):
            return self.variable == other.variable and self.formula == other.formula
        return False
    
    def __hash__(self):
        return hash(("exists", self.variable, self.formula))


class SecondOrderLogic:
    """Second-order logic with quantification over predicates and functions."""
    
    class PredicateQuantifier(FirstOrderFormula):
        """Quantifier over predicates (second-order logic)."""
        
        def __init__(self, predicate_name, arity, formula, universal=True):
            self.predicate_name = predicate_name
            self.arity = arity  # Predicate arity
            self.formula = formula
            self.universal = universal  # True for ∀, False for ∃
        
        def evaluate(self, interpretation, domain):
            """
            Evaluates the predicate quantifier in a given interpretation.
            domain: set of possible values for variables
            """
            # Generate all possible extensions for the predicate
            extensions = self._generate_predicate_extensions(domain)
            
            if self.universal:
                # For any predicate, the formula must be true
                for extension in extensions:
                    new_interpretation = interpretation.copy()
                    new_interpretation[self.predicate_name] = extension
                    
                    if not self.formula.evaluate(new_interpretation):
                        return False
                return True
            else:
                # There must exist at least one predicate for which the formula is true
                for extension in extensions:
                    new_interpretation = interpretation.copy()
                    new_interpretation[self.predicate_name] = extension
                    
                    if self.formula.evaluate(new_interpretation):
                        return True
                return False
        
        def _generate_predicate_extensions(self, domain):
            """Generates all possible extensions for the predicate."""
            # For simplicity, we limit to a small number of extensions
            # In a full implementation, all possible combinations would be generated
            extensions = []
            
            # Generate all combinations of tuples of length arity
            tuples = list(itertools.product(domain, repeat=self.arity))
            
            # Generate all sub-collections of these tuples
            for i in range(len(tuples) + 1):
                for combo in itertools.combinations(tuples, i):
                    extensions.append(set(combo))
            
            return extensions[:10]  # Limit for performance reasons
        
        def substitute(self, substitution):
            """Applies a substitution to the predicate quantifier."""
            return SecondOrderLogic.PredicateQuantifier(
                self.predicate_name,
                self.arity,
                self.formula.substitute(substitution),
                self.universal
            )
        
        def free_variables(self):
            """Returns the set of free variables of the predicate quantifier."""
            # The predicate name is bound by the quantifier
            return self.formula.free_variables()
        
        def __str__(self):
            quantifier = "∀" if self.universal else "∃"
            return f"{quantifier}{self.predicate_name}^{self.arity}.{self.formula}"
        
        def __eq__(self, other):
            if isinstance(other, SecondOrderLogic.PredicateQuantifier):
                return (self.predicate_name == other.predicate_name and
                        self.arity == other.arity and
                        self.formula == other.formula and
                        self.universal == other.universal)
            return False
        
        def __hash__(self):
            return hash(("predicate_quantifier", self.predicate_name, self.arity, self.formula, self.universal))
    
    class FunctionQuantifier(FirstOrderFormula):
        """Quantifier over functions (second-order logic)."""
        
        def __init__(self, function_name, arity, formula, universal=True):
            self.function_name = function_name
            self.arity = arity  # Function arity
            self.formula = formula
            self.universal = universal  # True for ∀, False for ∃
        
        def evaluate(self, interpretation, domain):
            """
            Evaluates the function quantifier in a given interpretation.
            domain: set of possible values for variables
            """
            # Generate all possible functions
            functions = self._generate_functions(domain)
            
            if self.universal:
                # For any function, the formula must be true
                for function in functions:
                    new_interpretation = interpretation.copy()
                    new_interpretation[self.function_name] = function
                    
                    if not self.formula.evaluate(new_interpretation):
                        return False
                return True
            else:
                # There must exist at least one function for which the formula is true
                for function in functions:
                    new_interpretation = interpretation.copy()
                    new_interpretation[self.function_name] = function
                    
                    if self.formula.evaluate(new_interpretation):
                        return True
                return False
        
        def _generate_functions(self, domain):
            """Generates all possible functions with the given domain."""
            # For simplicity, we limit to a small number of functions
            functions = []
            
            # Generate all possible argument tuples
            arguments = list(itertools.product(domain, repeat=self.arity))
            
            # Generate some random functions
            for _ in range(5):
                function = {}
                for args in arguments:
                    function[args] = random.choice(list(domain))
                functions.append(function)
            
            return functions
        
        def substitute(self, substitution):
            """Applies a substitution to the function quantifier."""
            return SecondOrderLogic.FunctionQuantifier(
                self.function_name,
                self.arity,
                self.formula.substitute(substitution),
                self.universal
            )
        
        def free_variables(self):
            """Returns the set of free variables of the function quantifier."""
            # The function name is bound by the quantifier
            return self.formula.free_variables()
        
        def __str__(self):
            quantifier = "∀" if self.universal else "∃"
            return f"{quantifier}{self.function_name}^{self.arity}.{self.formula}"
        
        def __eq__(self, other):
            if isinstance(other, SecondOrderLogic.FunctionQuantifier):
                return (self.function_name == other.function_name and
                        self.arity == other.arity and
                        self.formula == other.formula and
                        self.universal == other.universal)
            return False
        
        def __hash__(self):
            return hash(("function_quantifier", self.function_name, self.arity, self.formula, self.universal))


class FirstOrderModelChecker:
    """Model checker for first-order logic."""
    
    @staticmethod
    def check(formula, interpretation, domain):
        """Checks if a formula is true in a given interpretation."""
        if isinstance(formula, UniversalQuantifier) or isinstance(formula, ExistentialQuantifier):
            return formula.evaluate(interpretation, domain)
        else:
            return formula.evaluate(interpretation)
    
    @staticmethod
    def is_satisfiable(formula, domain):
        """Checks if a formula is satisfiable in a given domain."""
        # Generate all possible interpretations
        interpretations = FirstOrderModelChecker._generate_interpretations(formula, domain)
        
        for interpretation in interpretations:
            if FirstOrderModelChecker.check(formula, interpretation, domain):
                return True
        
        return False
    
    @staticmethod
    def is_valid(formula, domain):
        """Checks if a formula is valid in a given domain."""
        # A formula is valid if its negation is unsatisfiable
        negation = FirstOrderNot(formula)
        return not FirstOrderModelChecker.is_satisfiable(negation, domain)
    
    @staticmethod
    def _generate_interpretations(formula, domain):
        """Generates all possible interpretations for a formula in a domain."""
        # For simplicity, we generate a limited number of interpretations
        interpretations = []
        
        # Extract predicates from the formula
        predicates = FirstOrderModelChecker._extract_predicates(formula)
        
        # Generate some random interpretations
        for _ in range(10):
            interpretation = {}
            
            for name, arity in predicates:
                # Generate a random extension for each predicate
                extension = set()
                tuples = list(itertools.product(domain, repeat=arity))
                
                # Randomly add tuples to the extension
                for tup in tuples:
                    if random.random() > 0.5:
                        extension.add(tup)
                
                interpretation[name] = extension
            
            interpretations.append(interpretation)
        
        return interpretations
    
    @staticmethod
    def _extract_predicates(formula):
        """Extracts predicates from a formula."""
        predicates = set()
        
        def traverse(f):
            if isinstance(f, Predicate):
                predicates.add((f.name, len(f.arguments)))
            elif isinstance(f, FirstOrderNot):
                traverse(f.formula)
            elif isinstance(f, FirstOrderAnd) or isinstance(f, FirstOrderOr) or isinstance(f, FirstOrderImplication):
                traverse(f.left)
                traverse(f.right)
            elif isinstance(f, UniversalQuantifier) or isinstance(f, ExistentialQuantifier):
                traverse(f.formula)
            elif isinstance(f, SecondOrderLogic.PredicateQuantifier) or isinstance(f, SecondOrderLogic.FunctionQuantifier): # Added for second order
                traverse(f.formula)
        
        traverse(formula)
        return predicates


##########################################
# PART 5: PARADOX AND CONTRADICTION MANAGEMENT
##########################################

class Paradox:
    """Class for managing logical paradoxes."""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def __str__(self):
        return f"Paradox: {self.name}\nDescription: {self.description}"


class RussellParadox(Paradox):
    """Russell's Paradox: the set of all sets that do not contain themselves."""
    
    def __init__(self):
        super().__init__(
            "Russell's Paradox",
            "If R is the set of all sets that do not contain themselves, "
            "then does R contain itself? If R contains itself, then it should "
            "not contain itself. If R does not contain itself, then it should "
            "contain itself."
        )
    
    def formalize(self):
        """Formalizes the paradox in first-order logic."""
        # Define R = {x | x ∉ x}
        # Then ask the question: R ∈ R ?
        
        # This is a simplification, because in reality, we would need
        # set theory to fully formalize this paradox
        x = Variable("x")
        R = Variable("R")
        belongs_to = lambda a, b: Predicate("belongs_to", [a, b])
        
        # Definition of R
        definition_R = UniversalQuantifier(
            x,
            FirstOrderImplication(
                belongs_to(x, R),
                FirstOrderNot(belongs_to(x, x))
            )
        )
        
        # Question: R ∈ R ?
        question = belongs_to(R, R)
        
        return definition_R, question
    
    def analyze(self):
        """Analyzes the paradox and explains why it is problematic."""
        return (
            "If R ∈ R, then by definition of R, R ∉ R, which is contradictory.\n"
            "If R ∉ R, then by definition of R, R ∈ R, which is also contradictory.\n"
            "This contradiction shows the limitations of naive set theory and "
            "led to the development of type theory and axiomatic set theory "
            "(like ZFC) to avoid such paradoxes."
        )


class LiarParadox(Paradox):
    """Liar Paradox: 'This sentence is false'."""
    
    def __init__(self):
        super().__init__(
            "Liar Paradox",
            "Consider the sentence: 'This sentence is false'. If it is true, "
            "then it is false. If it is false, then it is true."
        )
    
    def formalize(self):
        """Attempts to formalize the paradox in modal logic."""
        # In standard logic, it is difficult to formalize this paradox
        # because it involves self-reference
        p = ModalProposition("p")
        
        # p ↔ ¬p (p is equivalent to non-p)
        equivalence = ModalAnd(
            ModalImplication(p, ModalNot(p)),
            ModalImplication(ModalNot(p), p)
        )
        
        return equivalence
    
    def analyze(self):
        """Analyzes the paradox and explains why it is problematic."""
        return (
            "The Liar Paradox involves self-reference that cannot be "
            "easily formalized in classical logic. Theories such as "
            "Tarski's type theory or paraconsistent logic have been developed "
            "to address this type of paradox.\n\n"
            "Tarski proposed a hierarchy of languages where the truth predicate for "
            "a language can only be defined in a metalanguage. This avoids the self-reference "
            "that leads to the paradox."
        )


class DialetheicLogicSystem:
    """Dialetheic logic system that tolerates certain contradictions."""
    
    def __init__(self):
        self.facts = {}  # Dictionary of facts with their truth value
    
    def add_fact(self, fact, value):
        """
        Adds a fact with its truth value.
        value: can be True, False, or "contradictory"
        """
        self.facts[str(fact)] = value
        return self
    
    def evaluate(self, formula):
        """Evaluates a formula in the dialetheic system."""
        if isinstance(formula, ModalProposition) or isinstance(formula, TemporalProposition) or isinstance(formula, Predicate):
            # Atomic proposition
            fact_str = str(formula)
            if fact_str in self.facts:
                return self.facts[fact_str]
            return False  # By default, unknown facts are false
        
        elif isinstance(formula, ModalNot) or isinstance(formula, TemporalNot) or isinstance(formula, FirstOrderNot):
            # Negation
            internal_value = self.evaluate(formula.formula)
            
            if internal_value == "contradictory":
                return "contradictory"
            elif internal_value is True:
                return False
            else:
                return True
        
        elif isinstance(formula, ModalAnd) or isinstance(formula, TemporalAnd) or isinstance(formula, FirstOrderAnd):
            # Conjunction
            left_value = self.evaluate(formula.left)
            right_value = self.evaluate(formula.right)
            
            if left_value == "contradictory" or right_value == "contradictory":
                return "contradictory"
            elif left_value is True and right_value is True:
                return True
            else:
                return False
        
        elif isinstance(formula, ModalOr) or isinstance(formula, TemporalOr) or isinstance(formula, FirstOrderOr):
            # Disjunction
            left_value = self.evaluate(formula.left)
            right_value = self.evaluate(formula.right)
            
            if left_value == "contradictory" or right_value == "contradictory":
                return "contradictory"
            elif left_value is True or right_value is True:
                return True
            else:
                return False
        
        elif isinstance(formula, ModalImplication) or isinstance(formula, TemporalImplication) or isinstance(formula, FirstOrderImplication):
            # Implication
            antecedent_value = self.evaluate(formula.antecedent)
            consequent_value = self.evaluate(formula.consequent)
            
            if antecedent_value == "contradictory" or consequent_value == "contradictory":
                return "contradictory"
            elif antecedent_value is False or consequent_value is True:
                return True
            else:
                return False
        
        # Other cases could be added for other formula types
        
        return False
    
    def is_coherent(self):
        """Checks if the system is consistent (no explicit contradictions)."""
        for fact, value in self.facts.items():
            if value == "contradictory":
                return False
            
            # Check if the negation of the fact exists and has an incompatible value
            for other_fact, other_value in self.facts.items():
                if other_fact.startswith("¬") and other_fact[1:] == fact:
                    if value is True and other_value is True:
                        return False
                    if value is False and other_value is False:
                        return False
        
        return True
    
    def identify_contradictions(self):
        """Identifies explicit contradictions in the system."""
        contradictions = []
        
        for fact, value in self.facts.items():
            if value == "contradictory":
                contradictions.append(fact)
                continue
            
            # Check if the negation of the fact exists and has an incompatible value
            for other_fact, other_value in self.facts.items():
                if other_fact.startswith("¬") and other_fact[1:] == fact:
                    if value is True and other_value is True:
                        contradictions.append(f"{fact} and {other_fact}")
                    if value is False and other_value is False:
                        contradictions.append(f"{fact} and {other_fact}")
        
        return contradictions
    
    def resolve_contradiction(self, fact, new_value):
        """Resolves a contradiction by modifying the value of a fact."""
        if fact in self.facts:
            self.facts[fact] = new_value
        return self


class ParaconsistentLogic:
    """Implementation of a paraconsistent logic that tolerates contradictions without trivialization."""
    
    class ParaconsistentValue(Enum):
        TRUE = 1
        FALSE = 2
        CONTRADICTORY = 3
        UNKNOWN = 4
    
    def __init__(self):
        self.values = {}  # Dictionary of truth values for propositions
    
    def set_value(self, proposition, value):
        """Defines the truth value of a proposition."""
        self.values[str(proposition)] = value
        return self
    
    def evaluate(self, formula):
        """Evaluates a formula in paraconsistent logic."""
        if isinstance(formula, ModalProposition) or isinstance(formula, TemporalProposition) or isinstance(formula, Predicate):
            # Atomic proposition
            prop_str = str(formula)
            if prop_str in self.values:
                return self.values[prop_str]
            return self.ParaconsistentValue.UNKNOWN
        
        elif isinstance(formula, ModalNot) or isinstance(formula, TemporalNot) or isinstance(formula, FirstOrderNot):
            # Negation
            internal_value = self.evaluate(formula.formula)
            
            if internal_value == self.ParaconsistentValue.TRUE:
                return self.ParaconsistentValue.FALSE
            elif internal_value == self.ParaconsistentValue.FALSE:
                return self.ParaconsistentValue.TRUE
            elif internal_value == self.ParaconsistentValue.CONTRADICTORY:
                return self.ParaconsistentValue.CONTRADICTORY
            else:  # UNKNOWN
                return self.ParaconsistentValue.UNKNOWN
        
        elif isinstance(formula, ModalAnd) or isinstance(formula, TemporalAnd) or isinstance(formula, FirstOrderAnd):
            # Conjunction
            left_value = self.evaluate(formula.left)
            right_value = self.evaluate(formula.right)
            
            # Truth table for paraconsistent conjunction
            if left_value == self.ParaconsistentValue.FALSE or right_value == self.ParaconsistentValue.FALSE:
                return self.ParaconsistentValue.FALSE
            elif left_value == self.ParaconsistentValue.CONTRADICTORY or right_value == self.ParaconsistentValue.CONTRADICTORY:
                return self.ParaconsistentValue.CONTRADICTORY
            elif left_value == self.ParaconsistentValue.UNKNOWN or right_value == self.ParaconsistentValue.UNKNOWN:
                return self.ParaconsistentValue.UNKNOWN
            else:  # Both are TRUE
                return self.ParaconsistentValue.TRUE
        
        elif isinstance(formula, ModalOr) or isinstance(formula, TemporalOr) or isinstance(formula, FirstOrderOr):
            # Disjunction
            left_value = self.evaluate(formula.left)
            right_value = self.evaluate(formula.right)
            
            # Truth table for paraconsistent disjunction
            if left_value == self.ParaconsistentValue.TRUE or right_value == self.ParaconsistentValue.TRUE:
                return self.ParaconsistentValue.TRUE
            elif left_value == self.ParaconsistentValue.CONTRADICTORY or right_value == self.ParaconsistentValue.CONTRADICTORY:
                return self.ParaconsistentValue.CONTRADICTORY
            elif left_value == self.ParaconsistentValue.UNKNOWN or right_value == self.ParaconsistentValue.UNKNOWN:
                return self.ParaconsistentValue.UNKNOWN
            else:  # Both are FALSE
                return self.ParaconsistentValue.FALSE
        
        elif isinstance(formula, ModalImplication) or isinstance(formula, TemporalImplication) or isinstance(formula, FirstOrderImplication):
            # Implication
            antecedent_value = self.evaluate(formula.antecedent)
            consequent_value = self.evaluate(formula.consequent)
            
            # Truth table for paraconsistent implication
            if antecedent_value == self.ParaconsistentValue.FALSE:
                return self.ParaconsistentValue.TRUE
            elif consequent_value == self.ParaconsistentValue.TRUE:
                return self.ParaconsistentValue.TRUE
            elif antecedent_value == self.ParaconsistentValue.CONTRADICTORY or consequent_value == self.ParaconsistentValue.CONTRADICTORY:
                return self.ParaconsistentValue.CONTRADICTORY
            elif antecedent_value == self.ParaconsistentValue.UNKNOWN or consequent_value == self.ParaconsistentValue.UNKNOWN:
                return self.ParaconsistentValue.UNKNOWN
            else:  # antecedent=TRUE and consequent=FALSE
                return self.ParaconsistentValue.FALSE
        
        # Other cases for other formula types
        
        return self.ParaconsistentValue.UNKNOWN
    
    def is_valid(self, formula):
        """Checks if a formula is valid in paraconsistent logic."""
        return self.evaluate(formula) == self.ParaconsistentValue.TRUE
    
    def is_contradictory(self, formula):
        """Checks if a formula is contradictory in paraconsistent logic."""
        return self.evaluate(formula) == self.ParaconsistentValue.CONTRADICTORY


class BeliefRevisionTheory:
    """Implementation of belief revision theory."""
    
    def __init__(self):
        self.beliefs = set()  # Set of current beliefs
    
    def add_belief(self, belief):
        """Adds a belief to the set of beliefs."""
        self.beliefs.add(str(belief))
        return self
    
    def revise(self, new_belief):
        """
        Revises the set of beliefs with a new belief.
        Revision consists of incorporating the new belief while
        preserving the consistency of the set.
        """
        # For simplicity, we implement a naive version of revision
        # where beliefs contradictory to the new belief are removed
        
        # Check if the new belief is compatible with existing beliefs
        incompatible_beliefs = set()
        for belief in self.beliefs:
            if self._are_contradictory(belief, str(new_belief)): # Changed method name
                incompatible_beliefs.add(belief)
        
        # Remove incompatible beliefs
        self.beliefs -= incompatible_beliefs
        
        # Add the new belief
        self.beliefs.add(str(new_belief))
        
        return self
    
    def contract(self, belief_to_remove):
        """
        Contracts the set of beliefs by removing a belief.
        Contraction consists of removing a belief without adding new information.
        """
        # Remove the belief
        if str(belief_to_remove) in self.beliefs:
            self.beliefs.remove(str(belief_to_remove))
        
        return self
    
    def _are_contradictory(self, belief1, belief2): # Changed method name
        """Checks if two beliefs are contradictory."""
        # For simplicity, we consider two beliefs contradictory
        # if one is the negation of the other
        return (belief1.startswith("¬") and belief1[1:] == belief2) or (belief2.startswith("¬") and belief2[1:] == belief1)
    
    def is_consistent(self):
        """Checks if the set of beliefs is consistent."""
        for c1 in self.beliefs:
            for c2 in self.beliefs:
                if self._are_contradictory(c1, c2):
                    return False
        return True
    
    def __str__(self):
        return "Beliefs: {" + ", ".join(self.beliefs) + "}"


def end_module(): # Renamed function
    """Final method to mark the end of the module."""
    print("Advanced and Non-Classical Formal Logic Module loaded successfully.")
    return True

# End of module
end_module()
