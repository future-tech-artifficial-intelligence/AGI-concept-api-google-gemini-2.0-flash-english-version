
"""
Système d'Apprentissage Continu et d'Auto-Amélioration
Permet à l'AGI/ASI d'apprendre de manière autonome et de s'améliorer continuellement.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pickle
import hashlib

@dataclass
class LearningExperience:
    """Représente une expérience d'apprentissage"""
    id: str
    context: Dict[str, Any]
    action_taken: str
    outcome: Dict[str, Any]
    feedback_score: float
    timestamp: datetime
    learning_type: str = "experiential"
    confidence: float = 0.5
    generalization_potential: float = 0.0

class ContinuousLearningSystem:
    """Système d'apprentissage continu pour AGI/ASI"""
    
    def __init__(self):
        self.experiences: List[LearningExperience] = []
        self.learned_patterns: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, List[float]] = {}
        self.adaptation_strategies: Dict[str, callable] = {}
        self.meta_learning_data: Dict[str, Any] = {}
        
    def record_experience(self, context: Dict[str, Any], action: str, 
                         outcome: Dict[str, Any], feedback_score: float) -> str:
        """Enregistre une nouvelle expérience d'apprentissage"""
        experience_id = hashlib.md5(
            f"{datetime.now().isoformat()}_{action}".encode()
        ).hexdigest()[:8]
        
        experience = LearningExperience(
            id=experience_id,
            context=context,
            action_taken=action,
            outcome=outcome,
            feedback_score=feedback_score,
            timestamp=datetime.now(),
            confidence=self._calculate_confidence(context, outcome),
            generalization_potential=self._assess_generalization_potential(context, action)
        )
        
        self.experiences.append(experience)
        
        # Déclencher l'apprentissage
        self._trigger_learning_from_experience(experience)
        
        return experience_id
    
    def _calculate_confidence(self, context: Dict[str, Any], outcome: Dict[str, Any]) -> float:
        """Calcule la confiance dans l'expérience"""
        # Facteurs de confiance
        context_clarity = len(context) / 10  # Plus de contexte = plus de confiance
        outcome_clarity = 1.0 if outcome.get("success") else 0.5
        
        return min(1.0, (context_clarity + outcome_clarity) / 2)
    
    def _assess_generalization_potential(self, context: Dict[str, Any], action: str) -> float:
        """Évalue le potentiel de généralisation d'une expérience"""
        # Rechercher des expériences similaires
        similar_experiences = self._find_similar_experiences(context, action)
        
        if len(similar_experiences) > 3:
            return 0.8  # Haute généralisation si beaucoup d'exemples similaires
        elif len(similar_experiences) > 1:
            return 0.6  # Généralisation modérée
        else:
            return 0.3  # Faible généralisation (expérience unique)
    
    def _find_similar_experiences(self, context: Dict[str, Any], action: str) -> List[LearningExperience]:
        """Trouve des expériences similaires"""
        similar = []
        
        for exp in self.experiences:
            # Calculer la similarité de contexte
            context_similarity = self._calculate_context_similarity(context, exp.context)
            action_similarity = 1.0 if action == exp.action_taken else 0.0
            
            overall_similarity = (context_similarity + action_similarity) / 2
            
            if overall_similarity > 0.7:
                similar.append(exp)
        
        return similar
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calcule la similarité entre deux contextes"""
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = context1[key], context2[key]
            if isinstance(val1, str) and isinstance(val2, str):
                # Similarité textuelle simple
                sim = len(set(val1.split()) & set(val2.split())) / len(set(val1.split()) | set(val2.split()))
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Similarité numérique
                sim = 1.0 - abs(val1 - val2) / max(abs(val1), abs(val2), 1)
            else:
                sim = 1.0 if val1 == val2 else 0.0
            
            similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _trigger_learning_from_experience(self, experience: LearningExperience) -> None:
        """Déclenche l'apprentissage à partir d'une expérience"""
        # Extraire des patterns
        patterns = self._extract_patterns([experience])
        
        # Mettre à jour les patterns appris
        for pattern_id, pattern_data in patterns.items():
            if pattern_id in self.learned_patterns:
                # Renforcer le pattern existant
                self.learned_patterns[pattern_id]["confidence"] = min(1.0, 
                    self.learned_patterns[pattern_id]["confidence"] + 0.1)
                self.learned_patterns[pattern_id]["occurrences"] += 1
            else:
                # Nouveau pattern
                self.learned_patterns[pattern_id] = pattern_data
        
        # Meta-apprentissage
        self._update_meta_learning(experience)
    
    def _extract_patterns(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Extrait des patterns d'apprentissage"""
        patterns = {}
        
        for exp in experiences:
            # Pattern contexte -> action
            context_key = self._create_context_signature(exp.context)
            action_pattern_id = f"context_action_{context_key}_{exp.action_taken}"
            
            patterns[action_pattern_id] = {
                "type": "context_action",
                "context_signature": context_key,
                "recommended_action": exp.action_taken,
                "expected_outcome": exp.outcome,
                "confidence": exp.confidence,
                "feedback_score": exp.feedback_score,
                "occurrences": 1,
                "last_seen": exp.timestamp
            }
            
            # Pattern action -> outcome
            outcome_pattern_id = f"action_outcome_{exp.action_taken}"
            if outcome_pattern_id not in patterns:
                patterns[outcome_pattern_id] = {
                    "type": "action_outcome",
                    "action": exp.action_taken,
                    "typical_outcomes": [exp.outcome],
                    "success_rate": 1.0 if exp.feedback_score > 0.5 else 0.0,
                    "confidence": exp.confidence,
                    "occurrences": 1
                }
        
        return patterns
    
    def _create_context_signature(self, context: Dict[str, Any]) -> str:
        """Crée une signature unique pour un contexte"""
        # Simplifier le contexte en gardant les éléments clés
        key_elements = []
        for key, value in context.items():
            if isinstance(value, str):
                key_elements.append(f"{key}:{value[:20]}")
            elif isinstance(value, (int, float)):
                key_elements.append(f"{key}:{value}")
            else:
                key_elements.append(f"{key}:complex")
        
        signature = "_".join(sorted(key_elements))
        return hashlib.md5(signature.encode()).hexdigest()[:8]
    
    def _update_meta_learning(self, experience: LearningExperience) -> None:
        """Met à jour les données de méta-apprentissage"""
        # Analyser l'efficacité des stratégies d'apprentissage
        learning_strategy = experience.learning_type
        
        if learning_strategy not in self.meta_learning_data:
            self.meta_learning_data[learning_strategy] = {
                "total_experiences": 0,
                "average_feedback": 0.0,
                "success_rate": 0.0,
                "adaptation_speed": 0.0
            }
        
        meta_data = self.meta_learning_data[learning_strategy]
        meta_data["total_experiences"] += 1
        
        # Mise à jour de la moyenne mobile du feedback
        alpha = 0.1  # Facteur d'oubli
        meta_data["average_feedback"] = (
            (1 - alpha) * meta_data["average_feedback"] + 
            alpha * experience.feedback_score
        )
        
        # Mise à jour du taux de succès
        success = 1.0 if experience.feedback_score > 0.5 else 0.0
        meta_data["success_rate"] = (
            (1 - alpha) * meta_data["success_rate"] + alpha * success
        )
    
    def suggest_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Suggère une action basée sur l'apprentissage"""
        context_signature = self._create_context_signature(context)
        
        # Rechercher des patterns correspondants
        relevant_patterns = []
        for pattern_id, pattern_data in self.learned_patterns.items():
            if pattern_data["type"] == "context_action":
                if pattern_data["context_signature"] == context_signature:
                    relevant_patterns.append((pattern_id, pattern_data))
        
        if relevant_patterns:
            # Sélectionner le pattern avec la meilleure confiance
            best_pattern = max(relevant_patterns, key=lambda x: x[1]["confidence"])
            
            return {
                "suggested_action": best_pattern[1]["recommended_action"],
                "confidence": best_pattern[1]["confidence"],
                "expected_outcome": best_pattern[1]["expected_outcome"],
                "based_on_pattern": best_pattern[0],
                "learning_source": "experience"
            }
        else:
            # Pas de pattern exact, utiliser la généralisation
            return self._generalize_suggestion(context)
    
    def _generalize_suggestion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Génère une suggestion par généralisation"""
        # Trouver les expériences les plus similaires
        similar_experiences = []
        for exp in self.experiences:
            similarity = self._calculate_context_similarity(context, exp.context)
            if similarity > 0.4:  # Seuil de similarité
                similar_experiences.append((exp, similarity))
        
        if similar_experiences:
            # Pondérer par similarité et performance
            weighted_actions = {}
            for exp, similarity in similar_experiences:
                action = exp.action_taken
                weight = similarity * exp.feedback_score
                
                if action in weighted_actions:
                    weighted_actions[action] += weight
                else:
                    weighted_actions[action] = weight
            
            # Sélectionner l'action avec le meilleur score pondéré
            best_action = max(weighted_actions.items(), key=lambda x: x[1])
            
            return {
                "suggested_action": best_action[0],
                "confidence": min(1.0, best_action[1]),
                "expected_outcome": {"success": True, "method": "generalization"},
                "based_on_pattern": "generalization",
                "learning_source": "similar_experiences",
                "similarity_count": len(similar_experiences)
            }
        else:
            # Aucune expérience similaire, suggestion exploratoire
            return {
                "suggested_action": "explore_new_approach",
                "confidence": 0.3,
                "expected_outcome": {"success": "unknown", "method": "exploration"},
                "based_on_pattern": "exploration",
                "learning_source": "no_prior_experience"
            }
    
    def self_improve(self) -> Dict[str, Any]:
        """Processus d'auto-amélioration"""
        improvements = {
            "performance_analysis": self._analyze_performance(),
            "strategy_optimization": self._optimize_strategies(),
            "knowledge_consolidation": self._consolidate_knowledge(),
            "capability_enhancement": self._enhance_capabilities()
        }
        
        return improvements
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyse la performance du système"""
        if len(self.experiences) < 10:
            return {"status": "insufficient_data"}
        
        recent_experiences = self.experiences[-20:]
        recent_feedback = [exp.feedback_score for exp in recent_experiences]
        
        average_performance = np.mean(recent_feedback)
        performance_trend = np.polyfit(range(len(recent_feedback)), recent_feedback, 1)[0]
        
        return {
            "average_performance": average_performance,
            "performance_trend": performance_trend,
            "improvement_needed": average_performance < 0.6,
            "trend_direction": "improving" if performance_trend > 0 else "declining"
        }
    
    def _optimize_strategies(self) -> Dict[str, Any]:
        """Optimise les stratégies d'apprentissage"""
        strategy_performance = {}
        
        for strategy, meta_data in self.meta_learning_data.items():
            strategy_performance[strategy] = meta_data["average_feedback"]
        
        if strategy_performance:
            best_strategy = max(strategy_performance.items(), key=lambda x: x[1])
            worst_strategy = min(strategy_performance.items(), key=lambda x: x[1])
            
            return {
                "best_strategy": best_strategy[0],
                "worst_strategy": worst_strategy[0],
                "performance_gap": best_strategy[1] - worst_strategy[1],
                "recommendation": f"Focus more on {best_strategy[0]} approach"
            }
        
        return {"status": "no_strategies_evaluated"}
    
    def _consolidate_knowledge(self) -> Dict[str, Any]:
        """Consolide les connaissances apprises"""
        # Identifier les patterns redondants
        pattern_groups = {}
        for pattern_id, pattern_data in self.learned_patterns.items():
            key = (pattern_data["type"], pattern_data.get("action", ""))
            if key not in pattern_groups:
                pattern_groups[key] = []
            pattern_groups[key].append((pattern_id, pattern_data))
        
        consolidated = 0
        for group in pattern_groups.values():
            if len(group) > 1:
                # Fusionner les patterns similaires
                consolidated += self._merge_similar_patterns(group)
        
        return {
            "patterns_consolidated": consolidated,
            "total_patterns": len(self.learned_patterns),
            "consolidation_rate": consolidated / max(len(self.learned_patterns), 1)
        }
    
    def _merge_similar_patterns(self, pattern_group: List[Tuple[str, Dict[str, Any]]]) -> int:
        """Fusionne des patterns similaires"""
        # Sélectionner le pattern avec la meilleure confiance comme base
        base_pattern_id, base_pattern = max(pattern_group, key=lambda x: x[1]["confidence"])
        
        merged_count = 0
        for pattern_id, pattern_data in pattern_group:
            if pattern_id != base_pattern_id:
                # Fusionner dans le pattern de base
                base_pattern["confidence"] = min(1.0, base_pattern["confidence"] + 0.05)
                base_pattern["occurrences"] += pattern_data.get("occurrences", 1)
                
                # Supprimer le pattern redondant
                del self.learned_patterns[pattern_id]
                merged_count += 1
        
        return merged_count
    
    def _enhance_capabilities(self) -> Dict[str, Any]:
        """Améliore les capacités du système"""
        enhancements = []
        
        # Analyser les échecs récents pour identifier les améliorations
        recent_failures = [exp for exp in self.experiences[-50:] if exp.feedback_score < 0.4]
        
        if len(recent_failures) > 5:
            failure_patterns = self._analyze_failure_patterns(recent_failures)
            enhancements.append(f"Address failure pattern: {failure_patterns}")
        
        # Identifier les domaines sous-explorés
        action_diversity = len(set(exp.action_taken for exp in self.experiences))
        if action_diversity < 10:
            enhancements.append("Increase action space exploration")
        
        return {
            "identified_enhancements": enhancements,
            "priority_enhancement": enhancements[0] if enhancements else "continue_current_approach"
        }
    
    def _analyze_failure_patterns(self, failures: List[LearningExperience]) -> str:
        """Analyse les patterns d'échec"""
        # Identifier les contextes ou actions qui mènent souvent à l'échec
        failure_contexts = [exp.context for exp in failures]
        failure_actions = [exp.action_taken for exp in failures]
        
        # Analyser la fréquence
        from collections import Counter
        action_failures = Counter(failure_actions)
        
        most_problematic = action_failures.most_common(1)
        if most_problematic:
            return f"Action '{most_problematic[0][0]}' often leads to failure"
        
        return "No clear failure pattern identified"

# Instance globale
continuous_learner = ContinuousLearningSystem()

def record_learning_experience(context: Dict[str, Any], action: str, 
                             outcome: Dict[str, Any], feedback_score: float) -> str:
    """Interface pour enregistrer une expérience d'apprentissage"""
    return continuous_learner.record_experience(context, action, outcome, feedback_score)

def get_action_suggestion(context: Dict[str, Any]) -> Dict[str, Any]:
    """Interface pour obtenir une suggestion d'action"""
    return continuous_learner.suggest_action(context)

def trigger_self_improvement() -> Dict[str, Any]:
    """Interface pour déclencher l'auto-amélioration"""
    return continuous_learner.self_improve()
