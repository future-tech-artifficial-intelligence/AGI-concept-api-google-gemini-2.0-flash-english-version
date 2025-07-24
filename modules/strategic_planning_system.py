
"""
Système de Planification Stratégique Multi-Niveaux pour AGI/ASI
Permet la planification à long terme, la décomposition d'objectifs complexes
et l'adaptation dynamique des stratégies.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import networkx as nx
import numpy as np

class PlanningHorizon(Enum):
    IMMEDIATE = "immediate"      # Seconde à minute
    SHORT_TERM = "short_term"    # Minute à heure  
    MEDIUM_TERM = "medium_term"  # Heure à jour
    LONG_TERM = "long_term"      # Jour à semaine
    STRATEGIC = "strategic"      # Semaine à mois+

@dataclass
class Goal:
    """Représente un objectif dans la hiérarchie"""
    id: str
    description: str
    horizon: PlanningHorizon
    priority: float
    deadline: Optional[datetime] = None
    parent_goal: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    completion_status: float = 0.0
    adaptive_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Action:
    """Représente une action concrète"""
    id: str
    description: str
    goal_id: str
    estimated_duration: timedelta
    required_capabilities: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    expected_outcomes: List[str] = field(default_factory=list)
    execution_status: str = "planned"
    confidence_level: float = 0.7

class StrategicPlanningSystem:
    """Système de planification stratégique pour AGI/ASI"""
    
    def __init__(self):
        self.goals_hierarchy = nx.DiGraph()
        self.goals: Dict[str, Goal] = {}
        self.actions: Dict[str, Action] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.adaptation_rules: Dict[str, callable] = {}
        
    def create_goal_hierarchy(self, root_objective: str) -> str:
        """Crée une hiérarchie d'objectifs à partir d'un objectif racine"""
        root_id = f"goal_{len(self.goals)}"
        
        # Créer l'objectif racine
        root_goal = Goal(
            id=root_id,
            description=root_objective,
            horizon=PlanningHorizon.STRATEGIC,
            priority=1.0
        )
        
        self.goals[root_id] = root_goal
        self.goals_hierarchy.add_node(root_id)
        
        # Décomposer récursivement
        self._decompose_goal(root_id, depth=0, max_depth=4)
        
        return root_id
    
    def _decompose_goal(self, goal_id: str, depth: int, max_depth: int):
        """Décompose un objectif en sous-objectifs"""
        if depth >= max_depth:
            return
            
        goal = self.goals[goal_id]
        
        # Générer des sous-objectifs intelligemment
        sub_objectives = self._generate_sub_objectives(goal)
        
        for sub_desc in sub_objectives:
            sub_id = f"goal_{len(self.goals)}"
            
            # Déterminer l'horizon temporel du sous-objectif
            sub_horizon = self._determine_sub_horizon(goal.horizon, depth)
            
            sub_goal = Goal(
                id=sub_id,
                description=sub_desc,
                horizon=sub_horizon,
                priority=goal.priority * (1 - depth * 0.1),
                parent_goal=goal_id
            )
            
            self.goals[sub_id] = sub_goal
            self.goals[goal_id].sub_goals.append(sub_id)
            
            self.goals_hierarchy.add_node(sub_id)
            self.goals_hierarchy.add_edge(goal_id, sub_id)
            
            # Décomposer récursivement
            self._decompose_goal(sub_id, depth + 1, max_depth)
    
    def _generate_sub_objectives(self, goal: Goal) -> List[str]:
        """Génère des sous-objectifs intelligents"""
        # Analyse du texte de l'objectif pour identifier les composants
        description = goal.description.lower()
        
        sub_objectives = []
        
        # Patterns de décomposition
        if "développer" in description or "créer" in description:
            sub_objectives.extend([
                "Analyser les exigences et contraintes",
                "Concevoir l'architecture et la solution",
                "Implémenter les composants principaux",
                "Tester et valider la solution",
                "Déployer et optimiser"
            ])
        elif "apprendre" in description or "comprendre" in description:
            sub_objectives.extend([
                "Identifier les sources d'information pertinentes",
                "Acquérir les connaissances de base",
                "Approfondir la compréhension",
                "Appliquer les connaissances acquises",
                "Évaluer et consolider l'apprentissage"
            ])
        elif "résoudre" in description or "problème" in description:
            sub_objectives.extend([
                "Analyser et définir le problème",
                "Identifier les solutions possibles",
                "Évaluer les alternatives",
                "Implémenter la solution choisie",
                "Vérifier l'efficacité de la solution"
            ])
        else:
            # Décomposition générique
            sub_objectives.extend([
                f"Phase de préparation pour {goal.description}",
                f"Exécution principale de {goal.description}",
                f"Finalisation et validation de {goal.description}"
            ])
        
        return sub_objectives[:3]  # Limiter à 3 sous-objectifs
    
    def _determine_sub_horizon(self, parent_horizon: PlanningHorizon, depth: int) -> PlanningHorizon:
        """Détermine l'horizon temporel d'un sous-objectif"""
        horizons = [
            PlanningHorizon.STRATEGIC,
            PlanningHorizon.LONG_TERM,
            PlanningHorizon.MEDIUM_TERM,
            PlanningHorizon.SHORT_TERM,
            PlanningHorizon.IMMEDIATE
        ]
        
        parent_index = horizons.index(parent_horizon)
        sub_index = min(parent_index + depth, len(horizons) - 1)
        
        return horizons[sub_index]
    
    def generate_action_plan(self, goal_id: str) -> List[Action]:
        """Génère un plan d'actions pour un objectif"""
        goal = self.goals[goal_id]
        actions = []
        
        # Si l'objectif a des sous-objectifs, générer des actions pour chacun
        if goal.sub_goals:
            for sub_goal_id in goal.sub_goals:
                sub_actions = self.generate_action_plan(sub_goal_id)
                actions.extend(sub_actions)
        else:
            # Générer des actions concrètes pour les objectifs feuilles
            concrete_actions = self._generate_concrete_actions(goal)
            actions.extend(concrete_actions)
        
        return actions
    
    def _generate_concrete_actions(self, goal: Goal) -> List[Action]:
        """Génère des actions concrètes pour un objectif feuille"""
        actions = []
        
        # Analyse du type d'objectif pour générer des actions appropriées
        description = goal.description.lower()
        
        if "analyser" in description:
            actions.append(Action(
                id=f"action_{len(self.actions)}",
                description=f"Collecter les données nécessaires pour {goal.description}",
                goal_id=goal.id,
                estimated_duration=timedelta(hours=2),
                required_capabilities=["data_collection", "analysis"]
            ))
            actions.append(Action(
                id=f"action_{len(self.actions) + 1}",
                description=f"Effectuer l'analyse pour {goal.description}",
                goal_id=goal.id,
                estimated_duration=timedelta(hours=4),
                required_capabilities=["analysis", "reasoning"]
            ))
        elif "implémenter" in description:
            actions.append(Action(
                id=f"action_{len(self.actions)}",
                description=f"Concevoir la solution pour {goal.description}",
                goal_id=goal.id,
                estimated_duration=timedelta(hours=3),
                required_capabilities=["design", "architecture"]
            ))
            actions.append(Action(
                id=f"action_{len(self.actions) + 1}",
                description=f"Coder la solution pour {goal.description}",
                goal_id=goal.id,
                estimated_duration=timedelta(hours=6),
                required_capabilities=["coding", "implementation"]
            ))
        else:
            # Action générique
            actions.append(Action(
                id=f"action_{len(self.actions)}",
                description=f"Exécuter {goal.description}",
                goal_id=goal.id,
                estimated_duration=timedelta(hours=2),
                required_capabilities=["general_execution"]
            ))
        
        # Enregistrer les actions
        for action in actions:
            self.actions[action.id] = action
        
        return actions
    
    def adaptive_replanning(self, execution_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Replanification adaptive basée sur les résultats d'exécution"""
        results = {
            "replanning_triggered": False,
            "modifications": [],
            "new_actions": [],
            "updated_priorities": {}
        }
        
        # Analyser les écarts de performance
        if "performance_gap" in execution_feedback:
            gap = execution_feedback["performance_gap"]
            
            if gap > 0.3:  # Écart significatif
                results["replanning_triggered"] = True
                
                # Identifier les objectifs affectés
                affected_goals = execution_feedback.get("affected_goals", [])
                
                for goal_id in affected_goals:
                    if goal_id in self.goals:
                        # Adapter la priorité
                        old_priority = self.goals[goal_id].priority
                        new_priority = min(1.0, old_priority * 1.2)
                        self.goals[goal_id].priority = new_priority
                        
                        results["updated_priorities"][goal_id] = {
                            "old": old_priority,
                            "new": new_priority
                        }
                        
                        # Générer de nouvelles actions si nécessaire
                        if gap > 0.5:
                            new_actions = self._generate_recovery_actions(goal_id, gap)
                            results["new_actions"].extend(new_actions)
        
        return results
    
    def _generate_recovery_actions(self, goal_id: str, performance_gap: float) -> List[Action]:
        """Génère des actions de récupération pour un objectif en difficulté"""
        goal = self.goals[goal_id]
        recovery_actions = []
        
        # Actions de récupération basées sur l'ampleur du problème
        if performance_gap > 0.7:
            # Problème majeur - revoir l'approche
            recovery_actions.append(Action(
                id=f"recovery_{len(self.actions)}",
                description=f"Réviser complètement l'approche pour {goal.description}",
                goal_id=goal_id,
                estimated_duration=timedelta(hours=4),
                required_capabilities=["strategic_thinking", "problem_solving"]
            ))
        elif performance_gap > 0.5:
            # Problème modéré - ajuster la méthode
            recovery_actions.append(Action(
                id=f"recovery_{len(self.actions)}",
                description=f"Ajuster la méthode d'exécution pour {goal.description}",
                goal_id=goal_id,
                estimated_duration=timedelta(hours=2),
                required_capabilities=["adaptation", "optimization"]
            ))
        else:
            # Problème mineur - optimiser
            recovery_actions.append(Action(
                id=f"recovery_{len(self.actions)}",
                description=f"Optimiser l'exécution de {goal.description}",
                goal_id=goal_id,
                estimated_duration=timedelta(hours=1),
                required_capabilities=["optimization"]
            ))
        
        # Enregistrer les actions
        for action in recovery_actions:
            self.actions[action.id] = action
        
        return recovery_actions
    
    def get_strategic_status(self) -> Dict[str, Any]:
        """Retourne le statut stratégique du système"""
        total_goals = len(self.goals)
        completed_goals = sum(1 for g in self.goals.values() if g.completion_status >= 1.0)
        
        # Calculer la progression par horizon
        horizon_progress = {}
        for horizon in PlanningHorizon:
            horizon_goals = [g for g in self.goals.values() if g.horizon == horizon]
            if horizon_goals:
                avg_progress = sum(g.completion_status for g in horizon_goals) / len(horizon_goals)
                horizon_progress[horizon.value] = avg_progress
        
        return {
            "total_goals": total_goals,
            "completed_goals": completed_goals,
            "completion_rate": completed_goals / max(total_goals, 1),
            "horizon_progress": horizon_progress,
            "active_actions": len([a for a in self.actions.values() if a.execution_status == "active"]),
            "pending_actions": len([a for a in self.actions.values() if a.execution_status == "planned"])
        }

# Instance globale
strategic_planner = StrategicPlanningSystem()

def create_strategic_plan(objective: str) -> Dict[str, Any]:
    """Interface pour créer un plan stratégique"""
    goal_id = strategic_planner.create_goal_hierarchy(objective)
    actions = strategic_planner.generate_action_plan(goal_id)
    
    return {
        "root_goal_id": goal_id,
        "total_goals": len(strategic_planner.goals),
        "action_plan": [{"id": a.id, "description": a.description} for a in actions[:10]],
        "estimated_duration": sum([a.estimated_duration for a in actions], timedelta()).days
    }
