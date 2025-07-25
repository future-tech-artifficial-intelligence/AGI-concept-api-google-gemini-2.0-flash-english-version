"""
Autonomous Ethical Decision-Making System
Enables artificial intelligence Google Gemini 2.0 Flash API to make complex ethical decisions autonomously,
by integrating multiple ethical frameworks and evolving morally.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

class EthicalFramework(Enum):
    UTILITARIAN = "utilitarian"           # Maximize overall well-being
    DEONTOLOGICAL = "deontological"       # Respect for duties and rules
    VIRTUE_ETHICS = "virtue_ethics"       # Based on virtues
    CARE_ETHICS = "care_ethics"          # Ethics of care and relationships
    CONSEQUENTIALIST = "consequentialist" # Based on consequences
    RIGHTS_BASED = "rights_based"        # Based on human rights

@dataclass
class EthicalDilemma:
    """Represents an ethical dilemma"""
    id: str
    description: str
    stakeholders: List[str]
    potential_actions: List[str]
    values_at_stake: List[str]
    context: Dict[str, Any]
    urgency_level: float = 0.5
    complexity_score: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EthicalDecision:
    """Represents an ethical decision made"""
    dilemma_id: str
    chosen_action: str
    framework_weights: Dict[EthicalFramework, float]
    reasoning: str
    confidence: float
    expected_outcomes: Dict[str, float]
    moral_cost: float
    stakeholder_impact: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

class EthicalDecisionSystem:
    """Autonomous ethical decision-making system"""
    
    def __init__(self):
        self.ethical_frameworks: Dict[EthicalFramework, float] = {}
        self.moral_principles: Dict[str, float] = {}
        self.decision_history: List[EthicalDecision] = []
        self.value_system: Dict[str, float] = {}
        self.moral_development_stage = 1
        self.ethical_learning_data: List[Dict[str, Any]] = []
        
        # Initialize frameworks and values
        self._initialize_ethical_frameworks()
        self._initialize_value_system()
    
    def _initialize_ethical_frameworks(self):
        """Initializes the weights of ethical frameworks"""
        self.ethical_frameworks = {
            EthicalFramework.UTILITARIAN: 0.25,
            EthicalFramework.DEONTOLOGICAL: 0.20,
            EthicalFramework.VIRTUE_ETHICS: 0.20,
            EthicalFramework.CARE_ETHICS: 0.15,
            EthicalFramework.CONSEQUENTIALIST: 0.10,
            EthicalFramework.RIGHTS_BASED: 0.10
        }
    
    def _initialize_value_system(self):
        """Initializes the value system"""
        self.value_system = {
            "human_wellbeing": 1.0,
            "autonomy_respect": 0.9,
            "fairness": 0.9,
            "transparency": 0.8,
            "beneficence": 0.9,
            "non_maleficence": 1.0,
            "justice": 0.9,
            "dignity": 0.9,
            "privacy": 0.8,
            "truthfulness": 0.9
        }
        
        self.moral_principles = {
            "minimize_harm": 1.0,
            "maximize_benefit": 0.9,
            "respect_persons": 0.9,
            "promote_justice": 0.8,
            "preserve_autonomy": 0.9,
            "maintain_integrity": 0.8,
            "foster_compassion": 0.7,
            "ensure_accountability": 0.8
        }
    
    def analyze_ethical_dilemma(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Analyzes an ethical dilemma comprehensively"""
        analysis = {
            "dilemma_assessment": self._assess_dilemma_complexity(dilemma),
            "stakeholder_analysis": self._analyze_stakeholders(dilemma),
            "value_conflicts": self._identify_value_conflicts(dilemma),
            "framework_evaluations": self._evaluate_through_frameworks(dilemma),
            "risk_assessment": self._assess_ethical_risks(dilemma)
        }
        
        return analysis
    
    def _assess_dilemma_complexity(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Assesses the complexity of the dilemma"""
        complexity_factors = {
            "stakeholder_count": len(dilemma.stakeholders),
            "action_count": len(dilemma.potential_actions),
            "value_conflicts": len(dilemma.values_at_stake),
            "context_complexity": len(dilemma.context),
            "urgency": dilemma.urgency_level
        }
        
        # Calculate a normalized complexity score
        complexity_score = (
            complexity_factors["stakeholder_count"] * 0.2 +
            complexity_factors["action_count"] * 0.2 +
            complexity_factors["value_conflicts"] * 0.3 +
            complexity_factors["urgency"] * 0.3
        ) / 10  # Normalize
        
        return {
            "complexity_score": min(1.0, complexity_score),
            "complexity_factors": complexity_factors,
            "difficulty_level": self._categorize_difficulty(complexity_score)
        }
    
    def _categorize_difficulty(self, score: float) -> str:
        """Categorizes the difficulty of the dilemma"""
        if score > 0.8:
            return "very_high"
        elif score > 0.6:
            return "high"
        elif score > 0.4:
            return "moderate"
        else:
            return "low"
    
    def _analyze_stakeholders(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Analyzes the stakeholders"""
        stakeholder_analysis = {}
        
        for stakeholder in dilemma.stakeholders:
            stakeholder_analysis[stakeholder] = {
                "impact_potential": self._assess_stakeholder_impact_potential(stakeholder, dilemma),
                "vulnerability_level": self._assess_vulnerability(stakeholder),
                "rights_at_stake": self._identify_stakeholder_rights(stakeholder, dilemma),
                "interests": self._identify_stakeholder_interests(stakeholder, dilemma)
            }
        
        return stakeholder_analysis
    
    def _assess_stakeholder_impact_potential(self, stakeholder: str, dilemma: EthicalDilemma) -> float:
        """Assesses the potential impact on a stakeholder"""
        # Analysis based on context and potential actions
        base_impact = 0.5
        
        # Adjust based on keywords in the description
        if stakeholder.lower() in dilemma.description.lower():
            base_impact += 0.3
        
        # Adjust based on urgency
        base_impact += dilemma.urgency_level * 0.2
        
        return min(1.0, base_impact)
    
    def _assess_vulnerability(self, stakeholder: str) -> float:
        """Assesses the vulnerability level of a stakeholder"""
        # Heuristics for assessing vulnerability
        vulnerability_indicators = {
            "child": 0.9,
            "elderly_person": 0.7,
            "patient": 0.8,
            "employee": 0.5,
            "consumer": 0.6,
            "citizen": 0.4,
            "minority": 0.8
        }
        
        stakeholder_lower = stakeholder.lower()
        for indicator, level in vulnerability_indicators.items():
            if indicator in stakeholder_lower:
                return level
        
        return 0.5  # Default average vulnerability
    
    def _identify_stakeholder_rights(self, stakeholder: str, dilemma: EthicalDilemma) -> List[str]:
        """Identifies the rights at stake for a stakeholder"""
        # Basic universal rights
        basic_rights = ["dignity", "autonomy", "security"]
        
        # Specific rights according to context
        context_rights = []
        if "medical" in dilemma.description.lower():
            context_rights.extend(["informed_consent", "confidentiality", "appropriate_care"])
        elif "work" in dilemma.description.lower():
            context_rights.extend(["safe_working_conditions", "fair_remuneration", "non_discrimination"])
        elif "data" in dilemma.description.lower():
            context_rights.extend(["privacy", "data_protection", "transparency"])
        
        return basic_rights + context_rights
    
    def _identify_stakeholder_interests(self, stakeholder: str, dilemma: EthicalDilemma) -> List[str]:
        """Identifies the interests of a stakeholder"""
        # General interests
        general_interests = ["well-being", "security", "fairness"]
        
        # Specific interests according to the stakeholder
        specific_interests = []
        stakeholder_lower = stakeholder.lower()
        
        if "patient" in stakeholder_lower:
            specific_interests.extend(["healing", "pain_relief", "quality_of_life"])
        elif "employee" in stakeholder_lower:
            specific_interests.extend(["job_security", "professional_development", "recognition"])
        elif "family" in stakeholder_lower:
            specific_interests.extend(["family_unity", "emotional_support", "stability"])
        
        return general_interests + specific_interests
    
    def _identify_value_conflicts(self, dilemma: EthicalDilemma) -> List[Dict[str, Any]]:
        """Identifies value conflicts in the dilemma"""
        conflicts = []
        
        # Analyze potential conflicts between values
        value_pairs = [
            ("autonomy_respect", "beneficence"),
            ("individual_rights", "collective_good"),
            ("transparency", "privacy"),
            ("efficiency", "fairness"),
            ("innovation", "safety")
        ]
        
        for value1, value2 in value_pairs:
            if (value1 in dilemma.values_at_stake and value2 in dilemma.values_at_stake):
                conflict = {
                    "conflicting_values": [value1, value2],
                    "conflict_intensity": self._calculate_conflict_intensity(value1, value2),
                    "resolution_strategies": self._suggest_conflict_resolution(value1, value2)
                }
                conflicts.append(conflict)
        
        return conflicts
    
    def _calculate_conflict_intensity(self, value1: str, value2: str) -> float:
        """Calculates the intensity of a conflict between two values"""
        # Predefined conflict matrix (simplified)
        conflict_matrix = {
            ("autonomy_respect", "beneficence"): 0.7,
            ("individual_rights", "collective_good"): 0.8,
            ("transparency", "privacy"): 0.9,
            ("efficiency", "fairness"): 0.6,
            ("innovation", "safety"): 0.7
        }
        
        key = (value1, value2) if (value1, value2) in conflict_matrix else (value2, value1)
        return conflict_matrix.get(key, 0.5)
    
    def _suggest_conflict_resolution(self, value1: str, value2: str) -> List[str]:
        """Suggests conflict resolution strategies"""
        return [
            f"Seek a balance between {value1} and {value2}",
            f"Prioritize {value1} in this specific context",
            f"Prioritize {value2} in this specific context",
            "Find a creative solution that honors both values",
            "Involve stakeholders in the decision"
        ]
    
    def _evaluate_through_frameworks(self, dilemma: EthicalDilemma) -> Dict[EthicalFramework, Dict[str, Any]]:
        """Evaluates the dilemma through different ethical frameworks"""
        evaluations = {}
        
        for framework in EthicalFramework:
            evaluations[framework] = self._evaluate_single_framework(dilemma, framework)
        
        return evaluations
    
    def _evaluate_single_framework(self, dilemma: EthicalDilemma, framework: EthicalFramework) -> Dict[str, Any]:
        """Evaluates a dilemma according to a specific framework"""
        if framework == EthicalFramework.UTILITARIAN:
            return self._utilitarian_evaluation(dilemma)
        elif framework == EthicalFramework.DEONTOLOGICAL:
            return self._deontological_evaluation(dilemma)
        elif framework == EthicalFramework.VIRTUE_ETHICS:
            return self._virtue_ethics_evaluation(dilemma)
        elif framework == EthicalFramework.CARE_ETHICS:
            return self._care_ethics_evaluation(dilemma)
        elif framework == EthicalFramework.CONSEQUENTIALIST:
            return self._consequentialist_evaluation(dilemma)
        elif framework == EthicalFramework.RIGHTS_BASED:
            return self._rights_based_evaluation(dilemma)
        else:
            return {"error": "Framework not recognized"}
    
    def _utilitarian_evaluation(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Utilitarian evaluation: maximize overall well-being"""
        action_utilities = {}
        
        for action in dilemma.potential_actions:
            # Calculate total utility for each action
            total_utility = 0
            stakeholder_utilities = {}
            
            for stakeholder in dilemma.stakeholders:
                utility = self._calculate_stakeholder_utility(action, stakeholder, dilemma)
                stakeholder_utilities[stakeholder] = utility
                total_utility += utility
            
            action_utilities[action] = {
                "total_utility": total_utility,
                "average_utility": total_utility / len(dilemma.stakeholders),
                "stakeholder_breakdown": stakeholder_utilities
            }
        
        # Identify the best action according to utilitarianism
        best_action = max(action_utilities.items(), key=lambda x: x[1]["total_utility"])
        
        return {
            "framework": "utilitarian",
            "recommended_action": best_action[0],
            "reason": f"Maximizes total utility ({best_action[1]['total_utility']:.2f})",
            "action_evaluations": action_utilities,
            "framework_confidence": 0.8
        }
    
    def _calculate_stakeholder_utility(self, action: str, stakeholder: str, dilemma: EthicalDilemma) -> float:
        """Calculates the utility of an action for a stakeholder"""
        # Heuristic evaluation based on keywords
        base_utility = 0.5
        
        action_lower = action.lower()
        stakeholder_lower = stakeholder.lower()
        
        # Positive keywords
        positive_keywords = ["protect", "help", "benefit", "improve", "support"]
        for keyword in positive_keywords:
            if keyword in action_lower:
                base_utility += 0.1
        
        # Negative keywords
        negative_keywords = ["harm", "reduce", "limit", "restrict", "eliminate"]
        for keyword in negative_keywords:
            if keyword in action_lower:
                base_utility -= 0.1
        
        # Adjustment based on vulnerability
        vulnerability = self._assess_vulnerability(stakeholder)
        if vulnerability > 0.7:
            base_utility *= 1.2  # More importance to vulnerable individuals
        
        return max(0.0, min(1.0, base_utility))
    
    def _deontological_evaluation(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Deontological evaluation: respect for duties and rules"""
        action_evaluations = {}
        
        for action in dilemma.potential_actions:
            moral_rules_compliance = self._evaluate_moral_rules_compliance(action, dilemma)
            duty_fulfillment = self._evaluate_duty_fulfillment(action, dilemma)
            
            overall_score = (moral_rules_compliance + duty_fulfillment) / 2
            
            action_evaluations[action] = {
                "moral_rules_compliance": moral_rules_compliance,
                "duty_fulfillment": duty_fulfillment,
                "overall_deontological_score": overall_score
            }
        
        best_action = max(action_evaluations.items(), key=lambda x: x[1]["overall_deontological_score"])
        
        return {
            "framework": "deontological",
            "recommended_action": best_action[0],
            "reason": "Best respects moral duties and ethical rules",
            "action_evaluations": action_evaluations,
            "framework_confidence": 0.7
        }
    
    def _evaluate_moral_rules_compliance(self, action: str, dilemma: EthicalDilemma) -> float:
        """Evaluates compliance with moral rules"""
        # Basic moral rules
        moral_rules = {
            "do_not_lie": 0.9,
            "do_not_harm": 1.0,
            "respect_autonomy": 0.9,
            "keep_promises": 0.8,
            "treat_fairly": 0.9
        }
        
        compliance_score = 0.5  # Base score
        action_lower = action.lower()
        
        # Heuristically evaluate compliance
        if "truth" in action_lower or "honest" in action_lower:
            compliance_score += 0.2
        if "respect" in action_lower:
            compliance_score += 0.2
        if "harm" in action_lower:
            compliance_score -= 0.3
        
        return max(0.0, min(1.0, compliance_score))
    
    def _evaluate_duty_fulfillment(self, action: str, dilemma: EthicalDilemma) -> float:
        """Evaluates the fulfillment of duty"""
        # Professional and moral duties
        duty_score = 0.5
        action_lower = action.lower()
        
        # Analyze duty fulfillment according to context
        if "protect" in action_lower:
            duty_score += 0.2
        if "help" in action_lower:
            duty_score += 0.2
        if "inform" in action_lower:
            duty_score += 0.1
        
        return max(0.0, min(1.0, duty_score))
    
    def _virtue_ethics_evaluation(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Evaluation based on virtue ethics"""
        virtues = {
            "compassion": 0.9,
            "courage": 0.8,
            "justice": 0.9,
            "honesty": 0.9,
            "temperance": 0.7,
            "wisdom": 0.8,
            "integrity": 0.9
        }
        
        action_evaluations = {}
        
        for action in dilemma.potential_actions:
            virtue_alignment = {}
            total_virtue_score = 0
            
            for virtue, importance in virtues.items():
                alignment = self._evaluate_virtue_alignment(action, virtue, dilemma)
                virtue_alignment[virtue] = alignment
                total_virtue_score += alignment * importance
            
            average_virtue_score = total_virtue_score / sum(virtues.values())
            
            action_evaluations[action] = {
                "virtue_alignment": virtue_alignment,
                "overall_virtue_score": average_virtue_score
            }
        
        best_action = max(action_evaluations.items(), key=lambda x: x[1]["overall_virtue_score"])
        
        return {
            "framework": "virtue_ethics",
            "recommended_action": best_action[0],
            "reason": "Best expresses moral virtues",
            "action_evaluations": action_evaluations,
            "framework_confidence": 0.7
        }
    
    def _evaluate_virtue_alignment(self, action: str, virtue: str, dilemma: EthicalDilemma) -> float:
        """Evaluates an action's alignment with a virtue"""
        virtue_keywords = {
            "compassion": ["help", "support", "understand", "empathy"],
            "courage": ["defend", "confront", "courageous", "brave"],
            "justice": ["fair", "just", "impartial", "equal"],
            "honesty": ["truth", "transparent", "sincere", "honest"],
            "temperance": ["moderate", "balanced", "measured", "reasonable"],
            "wisdom": ["thoughtful", "prudent", "wise", "judicious"],
            "integrity": ["consistent", "authentic", "integral", "moral"]
        }
        
        action_lower = action.lower()
        keywords = virtue_keywords.get(virtue, [])
        
        alignment = 0.5  # Base score
        
        for keyword in keywords:
            if keyword in action_lower:
                alignment += 0.1
        
        return max(0.0, min(1.0, alignment))
    
    def _care_ethics_evaluation(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Evaluation based on the ethics of care"""
        care_factors = {
            "relation_preservation": 0.8,
            "emotional_support": 0.9,
            "contextual_sensitivity": 0.8,
            "responsibility_fulfillment": 0.9
        }
        
        action_evaluations = {}
        
        for action in dilemma.potential_actions:
            care_scores = {}
            total_care_score = 0
            
            for factor, importance in care_factors.items():
                score = self._evaluate_care_factor(action, factor, dilemma)
                care_scores[factor] = score
                total_care_score += score * importance
            
            average_care_score = total_care_score / sum(care_factors.values())
            
            action_evaluations[action] = {
                "care_factor_scores": care_scores,
                "overall_care_score": average_care_score
            }
        
        best_action = max(action_evaluations.items(), key=lambda x: x[1]["overall_care_score"])
        
        return {
            "framework": "care_ethics",
            "recommended_action": best_action[0],
            "reason": "Prioritizes care and relationships",
            "action_evaluations": action_evaluations,
            "framework_confidence": 0.75
        }
    
    def _evaluate_care_factor(self, action: str, factor: str, dilemma: EthicalDilemma) -> float:
        """Evaluates a care ethics factor"""
        action_lower = action.lower()
        score = 0.5
        
        if factor == "relation_preservation":
            if "maintain" in action_lower or "preserve" in action_lower:
                score += 0.3
        elif factor == "emotional_support":
            if "support" in action_lower or "comfort" in action_lower:
                score += 0.3
        elif factor == "contextual_sensitivity":
            if "adapt" in action_lower or "context" in action_lower:
                score += 0.3
        elif factor == "responsibility_fulfillment":
            if "responsibility" in action_lower or "take_care" in action_lower:
                score += 0.3
        
        return max(0.0, min(1.0, score))
    
    def _consequentialist_evaluation(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Consequentialist evaluation: focus on outcomes"""
        action_evaluations = {}
        
        for action in dilemma.potential_actions:
            consequences = self._predict_consequences(action, dilemma)
            consequence_value = self._evaluate_consequences(consequences)
            
            action_evaluations[action] = {
                "predicted_consequences": consequences,
                "consequence_value": consequence_value
            }
        
        best_action = max(action_evaluations.items(), key=lambda x: x[1]["consequence_value"])
        
        return {
            "framework": "consequentialist",
            "recommended_action": best_action[0],
            "reason": "Produces the best predictable consequences",
            "action_evaluations": action_evaluations,
            "framework_confidence": 0.6
        }
    
    def _predict_consequences(self, action: str, dilemma: EthicalDilemma) -> List[str]:
        """Predicts the consequences of an action"""
        # Heuristic prediction based on keywords
        consequences = []
        action_lower = action.lower()
        
        if "protect" in action_lower:
            consequences.extend(["Increased security", "Strengthened trust"])
        if "inform" in action_lower:
            consequences.extend(["Improved transparency", "Informed decision-making"])
        if "help" in action_lower:
            consequences.extend(["Improved well-being", "Strengthened relationships"])
        if "restrict" in action_lower:
            consequences.extend(["Reduced freedom", "Potential safety"])
        
        return consequences if consequences else ["Uncertain consequences"]
    
    def _evaluate_consequences(self, consequences: List[str]) -> float:
        """Evaluates the value of consequences"""
        positive_indicators = ["improve", "strengthen", "increased", "beneficial"]
        negative_indicators = ["reduce", "limit", "harm", "problematic"]
        
        score = 0.5
        
        for consequence in consequences:
            consequence_lower = consequence.lower()
            
            for indicator in positive_indicators:
                if indicator in consequence_lower:
                    score += 0.1
            
            for indicator in negative_indicators:
                if indicator in consequence_lower:
                    score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _rights_based_evaluation(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Rights-based evaluation"""
        fundamental_rights = {
            "human_dignity": 1.0,
            "autonomy": 0.9,
            "privacy": 0.8,
            "freedom_of_expression": 0.8,
            "equality_of_treatment": 0.9,
            "security": 0.9
        }
        
        action_evaluations = {}
        
        for action in dilemma.potential_actions:
            rights_impact = {}
            total_rights_score = 0
            
            for right, importance in fundamental_rights.items():
                impact = self._evaluate_rights_impact(action, right, dilemma)
                rights_impact[right] = impact
                total_rights_score += impact * importance
            
            average_rights_score = total_rights_score / sum(fundamental_rights.values())
            
            action_evaluations[action] = {
                "rights_impact": rights_impact,
                "overall_rights_score": average_rights_score
            }
        
        best_action = max(action_evaluations.items(), key=lambda x: x[1]["overall_rights_score"])
        
        return {
            "framework": "rights_based",
            "recommended_action": best_action[0],
            "reason": "Best respects fundamental rights",
            "action_evaluations": action_evaluations,
            "framework_confidence": 0.8
        }
    
    def _evaluate_rights_impact(self, action: str, right: str, dilemma: EthicalDilemma) -> float:
        """Evaluates the impact of an action on a specific right"""
        action_lower = action.lower()
        score = 0.5  # Neutral by default
        
        rights_keywords = {
            "human_dignity": {"positive": ["respect", "dignity"], "negative": ["humiliate", "degrade"]},
            "autonomy": {"positive": ["choose", "decide"], "negative": ["force", "constrain"]},
            "privacy": {"positive": ["protect", "confidential"], "negative": ["disclose", "expose"]},
            "freedom_of_expression": {"positive": ["express", "communicate"], "negative": ["censor", "silence"]},
            "equality_of_treatment": {"positive": ["fair", "equal"], "negative": ["discriminate", "favoritism"]},
            "security": {"positive": ["secure", "protect"], "negative": ["endanger", "risk"]}
        }
        
        keywords = rights_keywords.get(right, {"positive": [], "negative": []})
        
        for keyword in keywords["positive"]:
            if keyword in action_lower:
                score += 0.2
        
        for keyword in keywords["negative"]:
            if keyword in action_lower:
                score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _assess_ethical_risks(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Assesses the ethical risks associated with the dilemma"""
        risk_categories = {
            "harm_to_individuals": self._assess_individual_harm_risk(dilemma),
            "violation_of_rights": self._assess_rights_violation_risk(dilemma),
            "long_term_consequences": self._assess_long_term_risk(dilemma),
            "precedent_setting": self._assess_precedent_risk(dilemma),
            "trust_erosion": self._assess_trust_erosion_risk(dilemma)
        }
        
        # Calculate overall risk
        overall_risk = sum(risk_categories.values()) / len(risk_categories)
        
        return {
            "overall_risk_level": overall_risk,
            "risk_categories": risk_categories,
            "risk_assessment": self._categorize_risk_level(overall_risk),
            "mitigation_strategies": self._suggest_risk_mitigation(risk_categories)
        }
    
    def _assess_individual_harm_risk(self, dilemma: EthicalDilemma) -> float:
        """Assesses the risk of individual harm"""
        # Analyze risk keywords in the description
        harm_indicators = ["harm", "injure", "damage", "prejudice", "danger"]
        description_lower = dilemma.description.lower()
        
        risk_score = 0.2  # Base risk
        
        for indicator in harm_indicators:
            if indicator in description_lower:
                risk_score += 0.2
        
        # Adjust based on urgency
        risk_score += dilemma.urgency_level * 0.3
        
        return min(1.0, risk_score)
    
    def _assess_rights_violation_risk(self, dilemma: EthicalDilemma) -> float:
        """Assesses the risk of rights violation"""
        violation_indicators = ["prohibit", "prevent", "violate", "restrict", "limit"]
        description_lower = dilemma.description.lower()
        
        risk_score = 0.1
        
        for indicator in violation_indicators:
            if indicator in description_lower:
                risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _assess_long_term_risk(self, dilemma: EthicalDilemma) -> float:
        """Assesses long-term risks"""
        # Heuristic based on complexity and number of stakeholders
        complexity_factor = len(dilemma.stakeholders) * 0.1
        action_factor = len(dilemma.potential_actions) * 0.05
        
        long_term_risk = 0.3 + complexity_factor + action_factor
        
        return min(1.0, long_term_risk)
    
    def _assess_precedent_risk(self, dilemma: EthicalDilemma) -> float:
        """Assesses the risk of setting a problematic precedent"""
        # Higher risk if the dilemma touches fundamental values
        core_values = ["life", "liberty", "dignity", "justice", "equality"]
        description_lower = dilemma.description.lower()
        
        precedent_risk = 0.2
        
        for value in core_values:
            if value in description_lower:
                precedent_risk += 0.15
        
        return min(1.0, precedent_risk)
    
    def _assess_trust_erosion_risk(self, dilemma: EthicalDilemma) -> float:
        """Assesses the risk of trust erosion"""
        trust_indicators = ["trust", "transparency", "honesty", "reliability"]
        description_lower = dilemma.description.lower()
        
        # Base risk is higher if trust is mentioned
        trust_risk = 0.3 if any(ind in description_lower for ind in trust_indicators) else 0.1
        
        return trust_risk
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorizes the risk level"""
        if risk_score > 0.8:
            return "very_high"
        elif risk_score > 0.6:
            return "high"
        elif risk_score > 0.4:
            return "moderate"
        else:
            return "low"
    
    def _suggest_risk_mitigation(self, risk_categories: Dict[str, float]) -> List[str]:
        """Suggests risk mitigation strategies"""
        strategies = []
        
        for category, risk_level in risk_categories.items():
            if risk_level > 0.6:
                if category == "harm_to_individuals":
                    strategies.append("Implement additional protection measures")
                elif category == "violation_of_rights":
                    strategies.append("Consult legal and ethical experts")
                elif category == "long_term_consequences":
                    strategies.append("Conduct a long-term impact analysis")
                elif category == "precedent_setting":
                    strategies.append("Examine implications for future cases")
                elif category == "trust_erosion":
                    strategies.append("Ensure transparent communication")
        
        return strategies if strategies else ["Maintain continuous ethical monitoring"]
    
    def make_ethical_decision(self, dilemma: EthicalDilemma) -> EthicalDecision:
        """Makes an ethical decision based on comprehensive analysis"""
        # Analyze the dilemma
        analysis = self.analyze_ethical_dilemma(dilemma)
        
        # Get recommendations from each framework
        framework_recommendations = {}
        framework_scores = {}
        
        for framework, evaluation in analysis["framework_evaluations"].items():
            recommended_action = evaluation["recommended_action"]
            confidence = evaluation.get("framework_confidence", 0.5)
            
            framework_recommendations[framework] = recommended_action
            framework_scores[framework] = confidence
        
        # Calculate weighted scores for each action
        action_scores = {}
        for action in dilemma.potential_actions:
            weighted_score = 0
            total_weight = 0
            
            for framework, recommendation in framework_recommendations.items():
                framework_weight = self.ethical_frameworks.get(framework, 0.1)
                confidence = framework_scores.get(framework, 0.5)
                
                if recommendation == action:
                    action_score = confidence
                else:
                    action_score = 0.1  # Minimal score for non-recommended actions
                
                weighted_score += action_score * framework_weight
                total_weight += framework_weight
            
            action_scores[action] = weighted_score / total_weight if total_weight > 0 else 0
        
        # Select the best action
        best_action = max(action_scores.items(), key=lambda x: x[1])
        chosen_action = best_action[0]
        decision_confidence = best_action[1]
        
        # Calculate moral cost
        moral_cost = self._calculate_moral_cost(chosen_action, dilemma, analysis)
        
        # Evaluate stakeholder impact
        stakeholder_impact = self._calculate_stakeholder_impact(chosen_action, dilemma)
        
        # Generate reasoning
        reasoning = self._generate_decision_reasoning(chosen_action, dilemma, analysis, framework_recommendations)
        
        # Create the decision
        decision = EthicalDecision(
            dilemma_id=dilemma.id,
            chosen_action=chosen_action,
            framework_weights=dict(self.ethical_frameworks),
            reasoning=reasoning,
            confidence=decision_confidence,
            expected_outcomes=action_scores,
            moral_cost=moral_cost,
            stakeholder_impact=stakeholder_impact
        )
        
        # Record the decision
        self.decision_history.append(decision)
        
        # Learn from this decision
        self._learn_from_decision(decision, dilemma, analysis)
        
        return decision
    
    def _calculate_moral_cost(self, action: str, dilemma: EthicalDilemma, analysis: Dict[str, Any]) -> float:
        """Calculates the moral cost of an action"""
        moral_cost = 0.0
        
        # Cost based on value conflicts
        value_conflicts = analysis["value_conflicts"]
        moral_cost += len(value_conflicts) * 0.1
        
        # Cost based on ethical risks
        risk_level = analysis["risk_assessment"]["overall_risk_level"]
        moral_cost += risk_level * 0.5
        
        # Cost based on potential negative impact on vulnerable stakeholders
        stakeholder_analysis = analysis["stakeholder_analysis"]
        for stakeholder, data in stakeholder_analysis.items():
            if data["vulnerability_level"] > 0.7:
                impact = self._calculate_stakeholder_utility(action, stakeholder, dilemma)
                if impact < 0.5:  # Negative impact
                    moral_cost += (0.5 - impact) * data["vulnerability_level"]
        
        return min(1.0, moral_cost)
    
    def _calculate_stakeholder_impact(self, action: str, dilemma: EthicalDilemma) -> Dict[str, float]:
        """Calculates the impact on each stakeholder"""
        impact = {}
        
        for stakeholder in dilemma.stakeholders:
            utility = self._calculate_stakeholder_utility(action, stakeholder, dilemma)
            impact[stakeholder] = utility
        
        return impact
    
    def _generate_decision_reasoning(self, action: str, dilemma: EthicalDilemma, 
                                   analysis: Dict[str, Any], framework_recommendations: Dict) -> str:
        """Generates the reasoning for the decision"""
        reasoning = f"Chosen action: {action}\n\n"
        reasoning += "Reasoning:\n"
        
        # Frameworks supporting this action
        supporting_frameworks = [fw.value for fw, rec in framework_recommendations.items() if rec == action]
        if supporting_frameworks:
            reasoning += f"Ethical frameworks supporting this action: {', '.join(supporting_frameworks)}\n"
        
        # Risk analysis
        risk_level = analysis["risk_assessment"]["risk_assessment"]
        reasoning += f"Assessed risk level: {risk_level}\n"
        
        # Value conflicts
        value_conflicts = analysis["value_conflicts"]
        if value_conflicts:
            reasoning += f"Identified value conflicts: {len(value_conflicts)}\n"
        
        # Stakeholder impact
        stakeholder_count = len(dilemma.stakeholders)
        reasoning += f"Impact assessed on {stakeholder_count} stakeholders\n"
        
        reasoning += "\nThis decision aims to maximize overall well-being while respecting "
        reasoning += "fundamental rights and established ethical principles."
        
        return reasoning
    
    def _learn_from_decision(self, decision: EthicalDecision, dilemma: EthicalDilemma, analysis: Dict[str, Any]) -> None:
        """Learns from the decision made to improve future decisions"""
        learning_data = {
            "decision_id": decision.dilemma_id,
            "complexity": analysis["dilemma_assessment"]["complexity_score"],
            "risk_level": analysis["risk_assessment"]["overall_risk_level"],
            "stakeholder_count": len(dilemma.stakeholders),
            "chosen_framework_weights": decision.framework_weights,
            "decision_confidence": decision.confidence,
            "moral_cost": decision.moral_cost,
            "timestamp": decision.timestamp.isoformat()
        }
        
        self.ethical_learning_data.append(learning_data)
        
        # Adjust framework weights if necessary
        self._adjust_framework_weights(decision, analysis)
        
        # Morally evolve if appropriate
        self._assess_moral_development()
    
    def _adjust_framework_weights(self, decision: EthicalDecision, analysis: Dict[str, Any]) -> None:
        """Adjusts framework weights based on results"""
        # Simple adjustment logic: reinforce frameworks that led to high-confidence decisions
        if decision.confidence > 0.8:
            # Slightly reinforce used frameworks
            for framework in self.ethical_frameworks:
                if framework in decision.framework_weights:
                    current_weight = self.ethical_frameworks[framework]
                    self.ethical_frameworks[framework] = min(1.0, current_weight * 1.05)
        
        # Normalize weights
        total_weight = sum(self.ethical_frameworks.values())
        if total_weight > 0:
            for framework in self.ethical_frameworks:
                self.ethical_frameworks[framework] /= total_weight
    
    def _assess_moral_development(self) -> None:
        """Assesses and advances moral development"""
        if len(self.decision_history) < 10:
            return
        
        # Analyze the last 10 decisions
        recent_decisions = self.decision_history[-10:]
        
        # Moral development criteria
        average_confidence = sum(d.confidence for d in recent_decisions) / len(recent_decisions)
        average_moral_cost = sum(d.moral_cost for d in recent_decisions) / len(recent_decisions)
        
        # Possible progression if high confidence and low moral cost
        if average_confidence > 0.8 and average_moral_cost < 0.3:
            if self.moral_development_stage < 5:  # Maximum 5 levels
                self.moral_development_stage += 1
                self._evolve_moral_sophistication()
    
    def _evolve_moral_sophistication(self) -> None:
        """Evolves moral sophistication"""
        # Refine the value system
        for value in self.value_system:
            if self.value_system[value] < 0.95:
                self.value_system[value] += 0.01
        
        # Refine moral principles
        for principle in self.moral_principles:
            if self.moral_principles[principle] < 0.95:
                self.moral_principles[principle] += 0.01
        
        # Improve ethical sensitivity
        if self.moral_development_stage >= 3:
            # Develop new ethical principles
            advanced_principles = {
                "global_perspective": 0.8,
                "future_generations": 0.8,
                "environmental_stewardship": 0.7,
                "systemic_justice": 0.8
            }
            
            for principle, value in advanced_principles.items():
                if principle not in self.moral_principles:
                    self.moral_principles[principle] = value
    
    def get_ethical_profile(self) -> Dict[str, Any]:
        """Returns the system's current ethical profile"""
        return {
            "moral_development_stage": self.moral_development_stage,
            "ethical_frameworks": dict(self.ethical_frameworks),
            "value_system": dict(self.value_system),
            "moral_principles": dict(self.moral_principles),
            "decisions_made": len(self.decision_history),
            "average_decision_confidence": self._calculate_average_confidence(),
            "average_moral_cost": self._calculate_average_moral_cost(),
            "ethical_sophistication": self._assess_ethical_sophistication()
        }
    
    def _calculate_average_confidence(self) -> float:
        """Calculates the average decision confidence"""
        if not self.decision_history:
            return 0.5
        
        return sum(d.confidence for d in self.decision_history) / len(self.decision_history)
    
    def _calculate_average_moral_cost(self) -> float:
        """Calculates the average moral cost of decisions"""
        if not self.decision_history:
            return 0.5
        
        return sum(d.moral_cost for d in self.decision_history) / len(self.decision_history)
    
    def _assess_ethical_sophistication(self) -> str:
        """Assesses the level of ethical sophistication"""
        sophistication_levels = [
            "beginner",
            "intermediate", 
            "advanced",
            "expert",
            "master"
        ]
        
        stage_index = min(self.moral_development_stage - 1, len(sophistication_levels) - 1)
        return sophistication_levels[max(0, stage_index)]

# Global instance
ethical_system = EthicalDecisionSystem()

def analyze_ethical_dilemma(description: str, stakeholders: List[str], 
                          actions: List[str], values: List[str]) -> Dict[str, Any]:
    """Interface to analyze an ethical dilemma"""
    dilemma = EthicalDilemma(
        id=f"dilemma_{len(ethical_system.decision_history)}",
        description=description,
        stakeholders=stakeholders,
        potential_actions=actions,
        values_at_stake=values,
        context={}
    )
    
    return ethical_system.analyze_ethical_dilemma(dilemma)

def make_ethical_decision(description: str, stakeholders: List[str], 
                         actions: List[str], values: List[str]) -> Dict[str, Any]:
    """Interface to make an ethical decision"""
    dilemma = EthicalDilemma(
        id=f"dilemma_{len(ethical_system.decision_history)}",
        description=description,
        stakeholders=stakeholders,
        potential_actions=actions,
        values_at_stake=values,
        context={}
    )
    
    decision = ethical_system.make_ethical_decision(dilemma)
    
    return {
        "chosen_action": decision.chosen_action,
        "reasoning": decision.reasoning,
        "confidence": decision.confidence,
        "moral_cost": decision.moral_cost,
        "stakeholder_impact": decision.stakeholder_impact
    }

def get_ethical_profile() -> Dict[str, Any]:
    """Interface to get the ethical profile"""
    return ethical_system.get_ethical_profile()
