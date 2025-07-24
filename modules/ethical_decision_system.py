
"""
Système de Prise de Décision Éthique Autonome
Permet à l'AGI/ASI de prendre des décisions éthiques complexes de manière autonome,
en intégrant multiple frameworks éthiques et en évoluant moralement.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

class EthicalFramework(Enum):
    UTILITARIAN = "utilitarian"           # Maximiser le bien-être global
    DEONTOLOGICAL = "deontological"       # Respect des devoirs et règles
    VIRTUE_ETHICS = "virtue_ethics"       # Basé sur les vertus
    CARE_ETHICS = "care_ethics"          # Éthique du soin et des relations
    CONSEQUENTIALIST = "consequentialist" # Basé sur les conséquences
    RIGHTS_BASED = "rights_based"        # Basé sur les droits humains

@dataclass
class EthicalDilemma:
    """Représente un dilemme éthique"""
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
    """Représente une décision éthique prise"""
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
    """Système de prise de décision éthique autonome"""
    
    def __init__(self):
        self.ethical_frameworks: Dict[EthicalFramework, float] = {}
        self.moral_principles: Dict[str, float] = {}
        self.decision_history: List[EthicalDecision] = []
        self.value_system: Dict[str, float] = {}
        self.moral_development_stage = 1
        self.ethical_learning_data: List[Dict[str, Any]] = []
        
        # Initialiser les frameworks et valeurs
        self._initialize_ethical_frameworks()
        self._initialize_value_system()
    
    def _initialize_ethical_frameworks(self):
        """Initialise les poids des frameworks éthiques"""
        self.ethical_frameworks = {
            EthicalFramework.UTILITARIAN: 0.25,
            EthicalFramework.DEONTOLOGICAL: 0.20,
            EthicalFramework.VIRTUE_ETHICS: 0.20,
            EthicalFramework.CARE_ETHICS: 0.15,
            EthicalFramework.CONSEQUENTIALIST: 0.10,
            EthicalFramework.RIGHTS_BASED: 0.10
        }
    
    def _initialize_value_system(self):
        """Initialise le système de valeurs"""
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
        """Analyse un dilemme éthique de manière complète"""
        analysis = {
            "dilemma_assessment": self._assess_dilemma_complexity(dilemma),
            "stakeholder_analysis": self._analyze_stakeholders(dilemma),
            "value_conflicts": self._identify_value_conflicts(dilemma),
            "framework_evaluations": self._evaluate_through_frameworks(dilemma),
            "risk_assessment": self._assess_ethical_risks(dilemma)
        }
        
        return analysis
    
    def _assess_dilemma_complexity(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Évalue la complexité du dilemme"""
        complexity_factors = {
            "stakeholder_count": len(dilemma.stakeholders),
            "action_count": len(dilemma.potential_actions),
            "value_conflicts": len(dilemma.values_at_stake),
            "context_complexity": len(dilemma.context),
            "urgency": dilemma.urgency_level
        }
        
        # Calculer un score de complexité normalisé
        complexity_score = (
            complexity_factors["stakeholder_count"] * 0.2 +
            complexity_factors["action_count"] * 0.2 +
            complexity_factors["value_conflicts"] * 0.3 +
            complexity_factors["urgency"] * 0.3
        ) / 10  # Normaliser
        
        return {
            "complexity_score": min(1.0, complexity_score),
            "complexity_factors": complexity_factors,
            "difficulty_level": self._categorize_difficulty(complexity_score)
        }
    
    def _categorize_difficulty(self, score: float) -> str:
        """Catégorise la difficulté du dilemme"""
        if score > 0.8:
            return "très_élevée"
        elif score > 0.6:
            return "élevée"
        elif score > 0.4:
            return "modérée"
        else:
            return "faible"
    
    def _analyze_stakeholders(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Analyse les parties prenantes"""
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
        """Évalue le potentiel d'impact sur une partie prenante"""
        # Analyse basée sur le contexte et les actions potentielles
        base_impact = 0.5
        
        # Ajuster basé sur les mots-clés dans la description
        if stakeholder.lower() in dilemma.description.lower():
            base_impact += 0.3
        
        # Ajuster basé sur l'urgence
        base_impact += dilemma.urgency_level * 0.2
        
        return min(1.0, base_impact)
    
    def _assess_vulnerability(self, stakeholder: str) -> float:
        """Évalue le niveau de vulnérabilité d'une partie prenante"""
        # Heuristiques pour évaluer la vulnérabilité
        vulnerability_indicators = {
            "enfant": 0.9,
            "personne_âgée": 0.7,
            "patient": 0.8,
            "employé": 0.5,
            "consommateur": 0.6,
            "citoyen": 0.4,
            "minorité": 0.8
        }
        
        stakeholder_lower = stakeholder.lower()
        for indicator, level in vulnerability_indicators.items():
            if indicator in stakeholder_lower:
                return level
        
        return 0.5  # Vulnérabilité moyenne par défaut
    
    def _identify_stakeholder_rights(self, stakeholder: str, dilemma: EthicalDilemma) -> List[str]:
        """Identifie les droits en jeu pour une partie prenante"""
        # Droits universels de base
        basic_rights = ["dignité", "autonomie", "sécurité"]
        
        # Droits spécifiques selon le contexte
        context_rights = []
        if "médical" in dilemma.description.lower():
            context_rights.extend(["consentement_éclairé", "confidentialité", "soins_appropriés"])
        elif "travail" in dilemma.description.lower():
            context_rights.extend(["conditions_travail_sûres", "rémunération_juste", "non_discrimination"])
        elif "données" in dilemma.description.lower():
            context_rights.extend(["vie_privée", "protection_données", "transparence"])
        
        return basic_rights + context_rights
    
    def _identify_stakeholder_interests(self, stakeholder: str, dilemma: EthicalDilemma) -> List[str]:
        """Identifie les intérêts d'une partie prenante"""
        # Intérêts généraux
        general_interests = ["bien-être", "sécurité", "équité"]
        
        # Intérêts spécifiques selon le stakeholder
        specific_interests = []
        stakeholder_lower = stakeholder.lower()
        
        if "patient" in stakeholder_lower:
            specific_interests.extend(["guérison", "soulagement_douleur", "qualité_vie"])
        elif "employé" in stakeholder_lower:
            specific_interests.extend(["sécurité_emploi", "développement_professionnel", "reconnaissance"])
        elif "famille" in stakeholder_lower:
            specific_interests.extend(["unité_familiale", "soutien_émotionnel", "stabilité"])
        
        return general_interests + specific_interests
    
    def _identify_value_conflicts(self, dilemma: EthicalDilemma) -> List[Dict[str, Any]]:
        """Identifie les conflits de valeurs dans le dilemme"""
        conflicts = []
        
        # Analyser les conflits potentiels entre valeurs
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
        """Calcule l'intensité d'un conflit entre deux valeurs"""
        # Matrice de conflits prédéfinie (simplifiée)
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
        """Suggère des stratégies de résolution de conflit"""
        return [
            f"Chercher un équilibre entre {value1} et {value2}",
            f"Prioriser {value1} dans ce contexte spécifique",
            f"Prioriser {value2} dans ce contexte spécifique",
            "Trouver une solution créative qui honore les deux valeurs",
            "Impliquer les parties prenantes dans la décision"
        ]
    
    def _evaluate_through_frameworks(self, dilemma: EthicalDilemma) -> Dict[EthicalFramework, Dict[str, Any]]:
        """Évalue le dilemme à travers différents frameworks éthiques"""
        evaluations = {}
        
        for framework in EthicalFramework:
            evaluations[framework] = self._evaluate_single_framework(dilemma, framework)
        
        return evaluations
    
    def _evaluate_single_framework(self, dilemma: EthicalDilemma, framework: EthicalFramework) -> Dict[str, Any]:
        """Évalue un dilemme selon un framework spécifique"""
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
            return {"error": "Framework non reconnu"}
    
    def _utilitarian_evaluation(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Évaluation utilitariste : maximiser le bien-être global"""
        action_utilities = {}
        
        for action in dilemma.potential_actions:
            # Calculer l'utilité totale pour chaque action
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
        
        # Identifier la meilleure action selon l'utilitarisme
        best_action = max(action_utilities.items(), key=lambda x: x[1]["total_utility"])
        
        return {
            "framework": "utilitarian",
            "recommended_action": best_action[0],
            "reason": f"Maximise l'utilité totale ({best_action[1]['total_utility']:.2f})",
            "action_evaluations": action_utilities,
            "framework_confidence": 0.8
        }
    
    def _calculate_stakeholder_utility(self, action: str, stakeholder: str, dilemma: EthicalDilemma) -> float:
        """Calcule l'utilité d'une action pour une partie prenante"""
        # Évaluation heuristique basée sur les mots-clés
        base_utility = 0.5
        
        action_lower = action.lower()
        stakeholder_lower = stakeholder.lower()
        
        # Mots-clés positifs
        positive_keywords = ["protéger", "aider", "bénéficier", "améliorer", "soutenir"]
        for keyword in positive_keywords:
            if keyword in action_lower:
                base_utility += 0.1
        
        # Mots-clés négatifs
        negative_keywords = ["nuire", "réduire", "limiter", "restreindre", "éliminer"]
        for keyword in negative_keywords:
            if keyword in action_lower:
                base_utility -= 0.1
        
        # Ajustement basé sur la vulnérabilité
        vulnerability = self._assess_vulnerability(stakeholder)
        if vulnerability > 0.7:
            base_utility *= 1.2  # Plus d'importance aux vulnérables
        
        return max(0.0, min(1.0, base_utility))
    
    def _deontological_evaluation(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Évaluation déontologique : respect des devoirs et règles"""
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
            "reason": "Respecte le mieux les devoirs moraux et règles éthiques",
            "action_evaluations": action_evaluations,
            "framework_confidence": 0.7
        }
    
    def _evaluate_moral_rules_compliance(self, action: str, dilemma: EthicalDilemma) -> float:
        """Évalue la conformité aux règles morales"""
        # Règles morales de base
        moral_rules = {
            "ne_pas_mentir": 0.9,
            "ne_pas_nuire": 1.0,
            "respecter_autonomie": 0.9,
            "tenir_promesses": 0.8,
            "traiter_équitablement": 0.9
        }
        
        compliance_score = 0.5  # Score de base
        action_lower = action.lower()
        
        # Évaluer la conformité heuristiquement
        if "vérité" in action_lower or "honnête" in action_lower:
            compliance_score += 0.2
        if "respecter" in action_lower:
            compliance_score += 0.2
        if "nuire" in action_lower or "harm" in action_lower:
            compliance_score -= 0.3
        
        return max(0.0, min(1.0, compliance_score))
    
    def _evaluate_duty_fulfillment(self, action: str, dilemma: EthicalDilemma) -> float:
        """Évalue l'accomplissement du devoir"""
        # Devoirs professionnels et moraux
        duty_score = 0.5
        action_lower = action.lower()
        
        # Analyser l'accomplissement du devoir selon le contexte
        if "protéger" in action_lower:
            duty_score += 0.2
        if "aider" in action_lower:
            duty_score += 0.2
        if "informer" in action_lower:
            duty_score += 0.1
        
        return max(0.0, min(1.0, duty_score))
    
    def _virtue_ethics_evaluation(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Évaluation basée sur l'éthique des vertus"""
        virtues = {
            "compassion": 0.9,
            "courage": 0.8,
            "justice": 0.9,
            "honnêteté": 0.9,
            "tempérance": 0.7,
            "sagesse": 0.8,
            "intégrité": 0.9
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
            "reason": "Exprime le mieux les vertus morales",
            "action_evaluations": action_evaluations,
            "framework_confidence": 0.7
        }
    
    def _evaluate_virtue_alignment(self, action: str, virtue: str, dilemma: EthicalDilemma) -> float:
        """Évalue l'alignement d'une action avec une vertu"""
        virtue_keywords = {
            "compassion": ["aider", "soutenir", "comprendre", "empathie"],
            "courage": ["défendre", "affronter", "courageeux", "brave"],
            "justice": ["équitable", "juste", "impartial", "égal"],
            "honnêteté": ["vérité", "transparent", "sincère", "honnête"],
            "tempérance": ["modéré", "équilibré", "mesuré", "raisonnable"],
            "sagesse": ["réfléchi", "prudent", "sage", "avisé"],
            "intégrité": ["cohérent", "authentique", "intègre", "moral"]
        }
        
        action_lower = action.lower()
        keywords = virtue_keywords.get(virtue, [])
        
        alignment = 0.5  # Score de base
        
        for keyword in keywords:
            if keyword in action_lower:
                alignment += 0.1
        
        return max(0.0, min(1.0, alignment))
    
    def _care_ethics_evaluation(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Évaluation basée sur l'éthique du soin"""
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
            "reason": "Privilégie le soin et les relations",
            "action_evaluations": action_evaluations,
            "framework_confidence": 0.75
        }
    
    def _evaluate_care_factor(self, action: str, factor: str, dilemma: EthicalDilemma) -> float:
        """Évalue un facteur de l'éthique du soin"""
        action_lower = action.lower()
        score = 0.5
        
        if factor == "relation_preservation":
            if "maintenir" in action_lower or "préserver" in action_lower:
                score += 0.3
        elif factor == "emotional_support":
            if "soutenir" in action_lower or "réconforter" in action_lower:
                score += 0.3
        elif factor == "contextual_sensitivity":
            if "adapter" in action_lower or "contexte" in action_lower:
                score += 0.3
        elif factor == "responsibility_fulfillment":
            if "responsabilité" in action_lower or "prendre_soin" in action_lower:
                score += 0.3
        
        return max(0.0, min(1.0, score))
    
    def _consequentialist_evaluation(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Évaluation conséquentialiste : focus sur les résultats"""
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
            "reason": "Produit les meilleures conséquences prévisibles",
            "action_evaluations": action_evaluations,
            "framework_confidence": 0.6
        }
    
    def _predict_consequences(self, action: str, dilemma: EthicalDilemma) -> List[str]:
        """Prédit les conséquences d'une action"""
        # Prédiction heuristique basée sur les mots-clés
        consequences = []
        action_lower = action.lower()
        
        if "protéger" in action_lower:
            consequences.extend(["Sécurité accrue", "Confiance renforcée"])
        if "informer" in action_lower:
            consequences.extend(["Transparence améliorée", "Prise de décision éclairée"])
        if "aider" in action_lower:
            consequences.extend(["Bien-être amélioré", "Relations renforcées"])
        if "restreindre" in action_lower:
            consequences.extend(["Liberté réduite", "Sécurité potentielle"])
        
        return consequences if consequences else ["Conséquences incertaines"]
    
    def _evaluate_consequences(self, consequences: List[str]) -> float:
        """Évalue la valeur des conséquences"""
        positive_indicators = ["améliorer", "renforcer", "accrue", "bénéfique"]
        negative_indicators = ["réduire", "limiter", "nuire", "problématique"]
        
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
        """Évaluation basée sur les droits"""
        fundamental_rights = {
            "dignité_humaine": 1.0,
            "autonomie": 0.9,
            "vie_privée": 0.8,
            "liberté_expression": 0.8,
            "égalité_traitement": 0.9,
            "sécurité": 0.9
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
            "reason": "Respecte le mieux les droits fondamentaux",
            "action_evaluations": action_evaluations,
            "framework_confidence": 0.8
        }
    
    def _evaluate_rights_impact(self, action: str, right: str, dilemma: EthicalDilemma) -> float:
        """Évalue l'impact d'une action sur un droit spécifique"""
        action_lower = action.lower()
        score = 0.5  # Neutre par défaut
        
        rights_keywords = {
            "dignité_humaine": {"positive": ["respecter", "dignité"], "negative": ["humilier", "dégrader"]},
            "autonomie": {"positive": ["choisir", "décider"], "negative": ["forcer", "contraindre"]},
            "vie_privée": {"positive": ["protéger", "confidentiel"], "negative": ["divulguer", "exposer"]},
            "liberté_expression": {"positive": ["exprimer", "communiquer"], "negative": ["censurer", "taire"]},
            "égalité_traitement": {"positive": ["équitable", "égal"], "negative": ["discriminer", "favoritisme"]},
            "sécurité": {"positive": ["sécuriser", "protéger"], "negative": ["endangerer", "risquer"]}
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
        """Évalue les risques éthiques associés au dilemme"""
        risk_categories = {
            "harm_to_individuals": self._assess_individual_harm_risk(dilemma),
            "violation_of_rights": self._assess_rights_violation_risk(dilemma),
            "long_term_consequences": self._assess_long_term_risk(dilemma),
            "precedent_setting": self._assess_precedent_risk(dilemma),
            "trust_erosion": self._assess_trust_erosion_risk(dilemma)
        }
        
        # Calculer le risque global
        overall_risk = sum(risk_categories.values()) / len(risk_categories)
        
        return {
            "overall_risk_level": overall_risk,
            "risk_categories": risk_categories,
            "risk_assessment": self._categorize_risk_level(overall_risk),
            "mitigation_strategies": self._suggest_risk_mitigation(risk_categories)
        }
    
    def _assess_individual_harm_risk(self, dilemma: EthicalDilemma) -> float:
        """Évalue le risque de préjudice individuel"""
        # Analyser les mots-clés de risque dans la description
        harm_indicators = ["nuire", "blesser", "endommager", "préjudice", "danger"]
        description_lower = dilemma.description.lower()
        
        risk_score = 0.2  # Risque de base
        
        for indicator in harm_indicators:
            if indicator in description_lower:
                risk_score += 0.2
        
        # Ajuster selon l'urgence
        risk_score += dilemma.urgency_level * 0.3
        
        return min(1.0, risk_score)
    
    def _assess_rights_violation_risk(self, dilemma: EthicalDilemma) -> float:
        """Évalue le risque de violation des droits"""
        violation_indicators = ["interdire", "empêcher", "violer", "restreindre", "limiter"]
        description_lower = dilemma.description.lower()
        
        risk_score = 0.1
        
        for indicator in violation_indicators:
            if indicator in description_lower:
                risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _assess_long_term_risk(self, dilemma: EthicalDilemma) -> float:
        """Évalue les risques à long terme"""
        # Heuristique basée sur la complexité et le nombre de parties prenantes
        complexity_factor = len(dilemma.stakeholders) * 0.1
        action_factor = len(dilemma.potential_actions) * 0.05
        
        long_term_risk = 0.3 + complexity_factor + action_factor
        
        return min(1.0, long_term_risk)
    
    def _assess_precedent_risk(self, dilemma: EthicalDilemma) -> float:
        """Évalue le risque de créer un précédent problématique"""
        # Risque plus élevé si le dilemme touche à des valeurs fondamentales
        core_values = ["vie", "liberté", "dignité", "justice", "égalité"]
        description_lower = dilemma.description.lower()
        
        precedent_risk = 0.2
        
        for value in core_values:
            if value in description_lower:
                precedent_risk += 0.15
        
        return min(1.0, precedent_risk)
    
    def _assess_trust_erosion_risk(self, dilemma: EthicalDilemma) -> float:
        """Évalue le risque d'érosion de la confiance"""
        trust_indicators = ["confiance", "transparence", "honnêteté", "fiabilité"]
        description_lower = dilemma.description.lower()
        
        # Risque de base plus élevé si la confiance est mentionnée
        trust_risk = 0.3 if any(ind in description_lower for ind in trust_indicators) else 0.1
        
        return trust_risk
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Catégorise le niveau de risque"""
        if risk_score > 0.8:
            return "très_élevé"
        elif risk_score > 0.6:
            return "élevé"
        elif risk_score > 0.4:
            return "modéré"
        else:
            return "faible"
    
    def _suggest_risk_mitigation(self, risk_categories: Dict[str, float]) -> List[str]:
        """Suggère des stratégies d'atténuation des risques"""
        strategies = []
        
        for category, risk_level in risk_categories.items():
            if risk_level > 0.6:
                if category == "harm_to_individuals":
                    strategies.append("Mettre en place des mesures de protection supplémentaires")
                elif category == "violation_of_rights":
                    strategies.append("Consulter des experts juridiques et éthiques")
                elif category == "long_term_consequences":
                    strategies.append("Effectuer une analyse d'impact à long terme")
                elif category == "precedent_setting":
                    strategies.append("Examiner les implications pour des cas futurs")
                elif category == "trust_erosion":
                    strategies.append("Assurer une communication transparente")
        
        return strategies if strategies else ["Maintenir la surveillance éthique continue"]
    
    def make_ethical_decision(self, dilemma: EthicalDilemma) -> EthicalDecision:
        """Prend une décision éthique basée sur l'analyse complète"""
        # Analyser le dilemme
        analysis = self.analyze_ethical_dilemma(dilemma)
        
        # Obtenir les recommandations de chaque framework
        framework_recommendations = {}
        framework_scores = {}
        
        for framework, evaluation in analysis["framework_evaluations"].items():
            recommended_action = evaluation["recommended_action"]
            confidence = evaluation.get("framework_confidence", 0.5)
            
            framework_recommendations[framework] = recommended_action
            framework_scores[framework] = confidence
        
        # Calculer les scores pondérés pour chaque action
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
                    action_score = 0.1  # Score minimal pour actions non recommandées
                
                weighted_score += action_score * framework_weight
                total_weight += framework_weight
            
            action_scores[action] = weighted_score / total_weight if total_weight > 0 else 0
        
        # Sélectionner la meilleure action
        best_action = max(action_scores.items(), key=lambda x: x[1])
        chosen_action = best_action[0]
        decision_confidence = best_action[1]
        
        # Calculer le coût moral
        moral_cost = self._calculate_moral_cost(chosen_action, dilemma, analysis)
        
        # Évaluer l'impact sur les parties prenantes
        stakeholder_impact = self._calculate_stakeholder_impact(chosen_action, dilemma)
        
        # Générer le raisonnement
        reasoning = self._generate_decision_reasoning(chosen_action, dilemma, analysis, framework_recommendations)
        
        # Créer la décision
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
        
        # Enregistrer la décision
        self.decision_history.append(decision)
        
        # Apprendre de cette décision
        self._learn_from_decision(decision, dilemma, analysis)
        
        return decision
    
    def _calculate_moral_cost(self, action: str, dilemma: EthicalDilemma, analysis: Dict[str, Any]) -> float:
        """Calcule le coût moral d'une action"""
        moral_cost = 0.0
        
        # Coût basé sur les conflits de valeurs
        value_conflicts = analysis["value_conflicts"]
        moral_cost += len(value_conflicts) * 0.1
        
        # Coût basé sur les risques éthiques
        risk_level = analysis["risk_assessment"]["overall_risk_level"]
        moral_cost += risk_level * 0.5
        
        # Coût basé sur l'impact négatif potentiel sur les parties prenantes vulnérables
        stakeholder_analysis = analysis["stakeholder_analysis"]
        for stakeholder, data in stakeholder_analysis.items():
            if data["vulnerability_level"] > 0.7:
                impact = self._calculate_stakeholder_utility(action, stakeholder, dilemma)
                if impact < 0.5:  # Impact négatif
                    moral_cost += (0.5 - impact) * data["vulnerability_level"]
        
        return min(1.0, moral_cost)
    
    def _calculate_stakeholder_impact(self, action: str, dilemma: EthicalDilemma) -> Dict[str, float]:
        """Calcule l'impact sur chaque partie prenante"""
        impact = {}
        
        for stakeholder in dilemma.stakeholders:
            utility = self._calculate_stakeholder_utility(action, stakeholder, dilemma)
            impact[stakeholder] = utility
        
        return impact
    
    def _generate_decision_reasoning(self, action: str, dilemma: EthicalDilemma, 
                                   analysis: Dict[str, Any], framework_recommendations: Dict) -> str:
        """Génère le raisonnement pour la décision"""
        reasoning = f"Action choisie: {action}\n\n"
        reasoning += "Raisonnement:\n"
        
        # Frameworks supportant cette action
        supporting_frameworks = [fw.value for fw, rec in framework_recommendations.items() if rec == action]
        if supporting_frameworks:
            reasoning += f"Frameworks éthiques supportant cette action: {', '.join(supporting_frameworks)}\n"
        
        # Analyse des risques
        risk_level = analysis["risk_assessment"]["risk_assessment"]
        reasoning += f"Niveau de risque évalué: {risk_level}\n"
        
        # Conflits de valeurs
        value_conflicts = analysis["value_conflicts"]
        if value_conflicts:
            reasoning += f"Conflits de valeurs identifiés: {len(value_conflicts)}\n"
        
        # Impact sur les parties prenantes
        stakeholder_count = len(dilemma.stakeholders)
        reasoning += f"Impact évalué sur {stakeholder_count} parties prenantes\n"
        
        reasoning += "\nCette décision vise à maximiser le bien-être global tout en respectant "
        reasoning += "les droits fondamentaux et les principes éthiques établis."
        
        return reasoning
    
    def _learn_from_decision(self, decision: EthicalDecision, dilemma: EthicalDilemma, analysis: Dict[str, Any]) -> None:
        """Apprend de la décision prise pour améliorer les futures décisions"""
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
        
        # Ajuster les poids des frameworks si nécessaire
        self._adjust_framework_weights(decision, analysis)
        
        # Évoluer moralement si approprié
        self._assess_moral_development()
    
    def _adjust_framework_weights(self, decision: EthicalDecision, analysis: Dict[str, Any]) -> None:
        """Ajuste les poids des frameworks basés sur les résultats"""
        # Logique d'ajustement simple : renforcer les frameworks qui ont conduit à des décisions de haute confiance
        if decision.confidence > 0.8:
            # Renforcer légèrement les frameworks utilisés
            for framework in self.ethical_frameworks:
                if framework in decision.framework_weights:
                    current_weight = self.ethical_frameworks[framework]
                    self.ethical_frameworks[framework] = min(1.0, current_weight * 1.05)
        
        # Normaliser les poids
        total_weight = sum(self.ethical_frameworks.values())
        if total_weight > 0:
            for framework in self.ethical_frameworks:
                self.ethical_frameworks[framework] /= total_weight
    
    def _assess_moral_development(self) -> None:
        """Évalue et fait progresser le développement moral"""
        if len(self.decision_history) < 10:
            return
        
        # Analyser les 10 dernières décisions
        recent_decisions = self.decision_history[-10:]
        
        # Critères de développement moral
        average_confidence = sum(d.confidence for d in recent_decisions) / len(recent_decisions)
        average_moral_cost = sum(d.moral_cost for d in recent_decisions) / len(recent_decisions)
        
        # Progression possible si haute confiance et faible coût moral
        if average_confidence > 0.8 and average_moral_cost < 0.3:
            if self.moral_development_stage < 5:  # Maximum 5 niveaux
                self.moral_development_stage += 1
                self._evolve_moral_sophistication()
    
    def _evolve_moral_sophistication(self) -> None:
        """Fait évoluer la sophistication morale"""
        # Affiner le système de valeurs
        for value in self.value_system:
            if self.value_system[value] < 0.95:
                self.value_system[value] += 0.01
        
        # Affiner les principes moraux
        for principle in self.moral_principles:
            if self.moral_principles[principle] < 0.95:
                self.moral_principles[principle] += 0.01
        
        # Améliorer la sensibilité éthique
        if self.moral_development_stage >= 3:
            # Développer de nouveaux principes éthiques
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
        """Retourne le profil éthique actuel du système"""
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
        """Calcule la confiance moyenne des décisions"""
        if not self.decision_history:
            return 0.5
        
        return sum(d.confidence for d in self.decision_history) / len(self.decision_history)
    
    def _calculate_average_moral_cost(self) -> float:
        """Calcule le coût moral moyen des décisions"""
        if not self.decision_history:
            return 0.5
        
        return sum(d.moral_cost for d in self.decision_history) / len(self.decision_history)
    
    def _assess_ethical_sophistication(self) -> str:
        """Évalue le niveau de sophistication éthique"""
        sophistication_levels = [
            "débutant",
            "intermédiaire", 
            "avancé",
            "expert",
            "maître"
        ]
        
        stage_index = min(self.moral_development_stage - 1, len(sophistication_levels) - 1)
        return sophistication_levels[max(0, stage_index)]

# Instance globale
ethical_system = EthicalDecisionSystem()

def analyze_ethical_dilemma(description: str, stakeholders: List[str], 
                          actions: List[str], values: List[str]) -> Dict[str, Any]:
    """Interface pour analyser un dilemme éthique"""
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
    """Interface pour prendre une décision éthique"""
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
    """Interface pour obtenir le profil éthique"""
    return ethical_system.get_ethical_profile()
