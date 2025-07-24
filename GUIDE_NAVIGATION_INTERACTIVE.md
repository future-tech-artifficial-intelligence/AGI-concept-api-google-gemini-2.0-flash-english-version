# Guide d'Utilisation - Système de Navigation Interactive Gemini

## 🎯 Vue d'ensemble

Le système de navigation interactive permet à l'API Gemini d'interagir directement avec les éléments des sites web (onglets, boutons, liens, formulaires) pour une navigation plus intelligente et automatisée.

## 📋 Table des matières

1. [Installation et Configuration](#installation-et-configuration)
2. [Démarrage rapide](#démarrage-rapide)
3. [Types d'interactions supportées](#types-dinteractions-supportées)
4. [Exemples d'utilisation](#exemples-dutilisation)
5. [Configuration avancée](#configuration-avancée)
6. [Dépannage](#dépannage)
7. [API de développeur](#api-de-développeur)

---

## 🚀 Installation et Configuration

### Prérequis

1. **Python 3.8+** avec les dépendances existantes du projet
2. **Selenium WebDriver** pour l'automatisation du navigateur
3. **ChromeDriver** ou **EdgeDriver** installé

### Installation automatique

```bash
# Exécuter le script d'installation des dépendances
python install_dependencies.py

# Installer les dépendances Selenium spécifiques
pip install selenium webdriver-manager
```

### Installation manuelle de ChromeDriver

**Windows :**
```bash
# Télécharger et placer ChromeDriver dans le PATH
# Ou utiliser webdriver-manager (recommandé)
pip install webdriver-manager
```

**Vérification de l'installation :**
```python
python -c "from selenium import webdriver; print('Selenium OK')"
```

---

## ⚡ Démarrage rapide

### 1. Test du système

```bash
# Tester l'installation complète
python test_interactive_navigation.py

# Voir une démonstration
python demo_interactive_navigation.py
```

### 2. Première utilisation avec Gemini

```python
from gemini_api_adapter import GeminiAPI

# Créer une instance Gemini avec le système interactif
gemini = GeminiAPI()

# Exemples de prompts interactifs
prompts = [
    "Clique sur l'onglet 'Services' de https://example.com",
    "Explore tous les onglets de ce site web",
    "Parcours toutes les sections pour voir les options disponibles"
]

# Utilisation normale
for prompt in prompts:
    response = gemini.get_response(prompt, user_id=1)
    print(response['response'])
```

### 3. Vérification des fonctionnalités

```python
# Vérifier que le système interactif est actif
from gemini_interactive_adapter import get_gemini_interactive_adapter

adapter = get_gemini_interactive_adapter()
if adapter:
    print("✅ Système interactif opérationnel")
    stats = adapter.get_interaction_statistics()
    print(f"📊 Statistiques: {stats}")
else:
    print("❌ Système interactif non disponible")
```

---

## 🎯 Types d'interactions supportées

### 1. Interaction directe
**Description :** Cliquer sur un élément spécifique mentionné par l'utilisateur.

**Exemples de prompts :**
- `"Clique sur le bouton 'Suivant'"`
- `"Appuie sur l'onglet 'Produits'"`
- `"Sélectionne le lien 'En savoir plus'"`

**Mots-clés détectés :**
- `clique sur`, `cliquer sur`
- `appuie sur`, `appuyer sur`
- `sélectionne`, `sélectionner`

### 2. Navigation par onglets
**Description :** Explorer systématiquement tous les onglets d'une page.

**Exemples de prompts :**
- `"Explore tous les onglets de ce site"`
- `"Parcours toutes les sections disponibles"`
- `"Va dans tous les onglets pour voir le contenu"`

**Fonctionnalités :**
- Détection automatique des onglets
- Navigation séquentielle
- Extraction du contenu de chaque onglet
- Résumé des découvertes

### 3. Exploration complète
**Description :** Navigation automatique et exhaustive d'un site.

**Exemples de prompts :**
- `"Explore toutes les options de ce site web"`
- `"Parcours tous les menus et sections"`
- `"Analyse complète de toutes les fonctionnalités"`

**Actions automatiques :**
- Identification des éléments interactifs
- Clics sur les éléments importants
- Navigation dans les sous-sections
- Compilation des informations trouvées

### 4. Interaction avec formulaires
**Description :** Analyse et interaction avec les formulaires web.

**Exemples de prompts :**
- `"Analyse le formulaire de contact"`
- `"Trouve les champs de recherche"`
- `"Montre-moi les options de filtrage"`

**Note de sécurité :** Le système identifie les formulaires mais ne saisit pas automatiquement de données pour des raisons de sécurité.

---

## 📖 Exemples d'utilisation

### Exemple 1 : E-commerce - Explorer les catégories

```python
from gemini_api_adapter import GeminiAPI

gemini = GeminiAPI()

prompt = """
Explore tous les onglets de catégories sur https://example-shop.com 
et donne-moi un résumé des produits disponibles dans chaque section.
"""

response = gemini.get_response(prompt, user_id=1)
print(response['response'])
```

**Résultat attendu :**
```
✅ J'ai exploré 5 onglets sur le site.

📋 Contenu des onglets découverts:
• Électronique: 150+ produits incluant smartphones, ordinateurs, accessoires
• Vêtements: Collection homme/femme avec 200+ articles de mode
• Maison & Jardin: Meubles, décoration, outils de jardinage (80+ items)
• Sports: Équipements sportifs, vêtements techniques (120+ produits)
• Livres: Large sélection de livres numériques et papier (500+ titres)

💡 Suggestions d'interaction:
• Explorer les sous-catégories d'électronique
• Consulter les promotions en cours
• Analyser les avis clients
```

### Exemple 2 : Site institutionnel - Services

```python
prompt = """
Clique sur l'onglet 'Services' de https://company-website.com 
et liste-moi tous les services proposés.
"""

response = gemini.get_response(prompt, user_id=1)
```

**Résultat attendu :**
```
✅ J'ai cliqué sur 'Services' et analysé le contenu.

📄 La page a changé suite à cette interaction.

🏢 Services proposés par l'entreprise:
• Conseil en stratégie digitale
• Développement d'applications web
• Formation en nouvelles technologies  
• Support technique 24/7
• Audit de sécurité informatique

📍 Page actuelle: https://company-website.com/services
```

### Exemple 3 : Recherche d'informations

```python
prompt = """
Sur le site de cette université, trouve la section pour les inscriptions 
et montre-moi les étapes à suivre.
"""

response = gemini.get_response(prompt, user_id=1, session_id="university_search")
```

---

## ⚙️ Configuration avancée

### 1. Configuration du navigateur

```python
from interactive_web_navigator import get_interactive_navigator

navigator = get_interactive_navigator()

# Modifier la configuration
navigator.config.update({
    'max_interactions_per_session': 100,  # Limite d'interactions
    'interaction_timeout': 45,            # Timeout en secondes
    'page_load_timeout': 20,              # Timeout de chargement
    'screenshot_on_interaction': True     # Captures d'écran automatiques
})
```

### 2. Configuration des sélecteurs CSS

```python
from interactive_web_navigator import InteractiveElementAnalyzer

analyzer = InteractiveElementAnalyzer()

# Ajouter des sélecteurs personnalisés
analyzer.element_selectors['custom_buttons'] = [
    '.my-custom-button',
    '[data-action="submit"]',
    '.special-interactive-element'
]

# Modifier les mots-clés d'importance
analyzer.importance_keywords['high'].extend(['acheter', 'commander', 'réserver'])
```

### 3. Configuration des statistiques

```python
from gemini_interactive_adapter import get_gemini_interactive_adapter

adapter = get_gemini_interactive_adapter()

# Afficher les statistiques détaillées
stats = adapter.get_interaction_statistics()
print(f"📊 Statistiques complètes:")
print(f"   🔢 Requêtes totales: {stats['stats']['total_requests']}")
print(f"   🎯 Sessions créées: {stats['stats']['interactive_sessions_created']}")
print(f"   ✅ Interactions réussies: {stats['stats']['successful_interactions']}")
print(f"   📂 Onglets explorés: {stats['stats']['tabs_explored']}")

# Nettoyer les anciennes sessions
adapter.cleanup_sessions(max_age_hours=1)
```

---

## 🔧 Dépannage

### Problèmes courants

#### 1. ChromeDriver non trouvé
**Erreur :**
```
selenium.common.exceptions.WebDriverException: 'chromedriver' executable needs to be in PATH
```

**Solutions :**
```bash
# Option 1: Installer webdriver-manager
pip install webdriver-manager

# Option 2: Télécharger manuellement ChromeDriver
# https://chromedriver.chromium.org/
# Placer dans le PATH système
```

#### 2. Éléments non cliquables
**Erreur :**
```
Element not clickable at point (x, y)
```

**Solutions :**
- Le système tente automatiquement un clic JavaScript
- Vérifier que la page est complètement chargée
- Augmenter les timeouts dans la configuration

#### 3. Détection d'interaction échoue
**Symptôme :** L'API ne détecte pas qu'une interaction est nécessaire.

**Solutions :**
```python
# Tester la détection manuellement
from gemini_interactive_adapter import detect_interactive_need

result = detect_interactive_need("Votre prompt ici")
print(f"Détection: {result}")

# Ajuster les mots-clés si nécessaire
# Voir section "Configuration avancée"
```

#### 4. Sessions bloquées
**Symptôme :** Sessions qui ne se ferment pas correctement.

**Solution :**
```python
# Forcer le nettoyage
from gemini_interactive_adapter import get_gemini_interactive_adapter

adapter = get_gemini_interactive_adapter()
adapter.cleanup_sessions(max_age_hours=0)  # Nettoie toutes les sessions
```

### Logs et débogage

```python
import logging

# Activer les logs détaillés
logging.getLogger('InteractiveWebNavigator').setLevel(logging.DEBUG)
logging.getLogger('GeminiInteractiveIntegration').setLevel(logging.DEBUG)

# Voir les logs en temps réel
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
```

---

## 🔌 API de développeur

### 1. Utilisation directe du navigateur

```python
from interactive_web_navigator import initialize_interactive_navigator

# Initialiser le navigateur
navigator = initialize_interactive_navigator()

# Créer une session
session_id = "my_custom_session"
session = navigator.create_interactive_session(
    session_id=session_id,
    start_url="https://example.com",
    navigation_goals=['explore_tabs', 'find_content']
)

# Naviguer vers une URL
result = navigator.navigate_to_url(session_id, "https://example.com")
print(f"Éléments trouvés: {result['elements_found']}")

# Interagir avec un élément
elements = result['interactive_elements']
if elements:
    element_id = elements[0]['id']
    interaction_result = navigator.interact_with_element(session_id, element_id, 'click')
    print(f"Interaction réussie: {interaction_result.success}")

# Fermer la session
navigator.close_session(session_id)
```

### 2. Créer des détecteurs personnalisés

```python
from gemini_interactive_adapter import GeminiInteractiveWebAdapter

class CustomInteractiveAdapter(GeminiInteractiveWebAdapter):
    def detect_custom_interaction(self, prompt):
        """Détecteur personnalisé pour des interactions spécifiques"""
        if "mon_mot_cle_special" in prompt.lower():
            return {
                'requires_interaction': True,
                'interaction_type': 'custom_action',
                'confidence': 0.95
            }
        return {'requires_interaction': False}
    
    def handle_custom_interaction(self, prompt, session_id):
        """Gestionnaire personnalisé"""
        # Votre logique personnalisée ici
        return {
            'success': True,
            'custom_action_performed': True,
            'details': 'Action personnalisée réalisée'
        }

# Utiliser l'adaptateur personnalisé
custom_adapter = CustomInteractiveAdapter()
```

### 3. Analyseur d'éléments personnalisé

```python
from interactive_web_navigator import InteractiveElementAnalyzer

class CustomElementAnalyzer(InteractiveElementAnalyzer):
    def __init__(self):
        super().__init__()
        
        # Ajouter des sélecteurs personnalisés
        self.element_selectors['my_custom_elements'] = [
            '.my-special-button',
            '[data-custom="interactive"]'
        ]
    
    def custom_scoring_logic(self, element_text, attributes):
        """Logique de scoring personnalisée"""
        score = 0.5  # Score de base
        
        # Votre logique personnalisée
        if 'important' in element_text.lower():
            score += 0.3
        
        return min(score, 1.0)

# Utiliser l'analyseur personnalisé
analyzer = CustomElementAnalyzer()
```

---

## 📊 Métriques et monitoring

### Statistiques disponibles

```python
from gemini_interactive_adapter import get_gemini_interactive_adapter
from interactive_web_navigator import get_interactive_navigator

# Statistiques de l'adaptateur Gemini
adapter = get_gemini_interactive_adapter()
adapter_stats = adapter.get_interaction_statistics()

print("📈 Statistiques de l'adaptateur:")
print(f"   Total requêtes: {adapter_stats['stats']['total_requests']}")
print(f"   Sessions créées: {adapter_stats['stats']['interactive_sessions_created']}")
print(f"   Interactions réussies: {adapter_stats['stats']['successful_interactions']}")
print(f"   Onglets explorés: {adapter_stats['stats']['tabs_explored']}")
print(f"   Formulaires interagis: {adapter_stats['stats']['forms_interacted']}")

# Statistiques du navigateur
navigator = get_interactive_navigator()
nav_stats = navigator.get_statistics()

print("\n🔍 Statistiques du navigateur:")
print(f"   Sessions actives: {nav_stats['active_sessions']}")
print(f"   Interactions réalisées: {nav_stats['stats']['interactions_performed']}")
print(f"   Éléments découverts: {nav_stats['stats']['elements_discovered']}")
print(f"   Pages naviguées: {nav_stats['stats']['pages_navigated']}")
```

### Surveillance en temps réel

```python
import time
from datetime import datetime

def monitor_interactive_system():
    """Surveillance continue du système"""
    adapter = get_gemini_interactive_adapter()
    
    while True:
        stats = adapter.get_interaction_statistics()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        print(f"[{timestamp}] Interactions: {stats['stats']['successful_interactions']}, "
              f"Sessions: {stats['stats']['interactive_sessions_created']}")
        
        time.sleep(30)  # Vérifier toutes les 30 secondes

# Lancer la surveillance
# monitor_interactive_system()
```

---

## 🛡️ Bonnes pratiques et sécurité

### 1. Respect des sites web

```python
# Ajouter des délais entre les interactions
navigator.config['interaction_delay'] = 2.0  # 2 secondes entre chaque action

# Limiter le nombre d'interactions par session
navigator.config['max_interactions_per_session'] = 20

# Respecter les robots.txt (à implémenter selon les besoins)
```

### 2. Gestion des erreurs

```python
try:
    result = navigator.interact_with_element(session_id, element_id, 'click')
    if not result.success:
        print(f"Interaction échouée: {result.error_message}")
        # Logique de fallback
except Exception as e:
    print(f"Erreur critique: {e}")
    # Nettoyage et récupération
```

### 3. Utilisation responsable

- **Fréquence des requêtes :** Éviter de surcharger les serveurs
- **Données personnelles :** Ne jamais saisir d'informations sensibles automatiquement
- **Respect des CGU :** Vérifier que l'automatisation est autorisée
- **Monitoring :** Surveiller les performances et erreurs

---

## 📞 Support et contribution

### Signaler un problème

1. **Créer un rapport de test :**
   ```bash
   python test_interactive_navigation.py
   ```

2. **Inclure les logs :**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   # Reproduire le problème
   ```

3. **Informations système :**
   - Version Python
   - Version Selenium
   - Navigateur utilisé (Chrome/Edge)
   - Système d'exploitation

### Contribution au projet

1. **Tests :** Ajouter des tests pour nouveaux cas d'usage
2. **Sélecteurs :** Améliorer la détection d'éléments
3. **Documentation :** Enrichir ce guide avec vos retours d'expérience

---

## 🎉 Conclusion

Le système de navigation interactive transforme l'API Gemini en un assistant capable d'interagir physiquement avec les sites web. Cette fonctionnalité ouvre de nouvelles possibilités pour :

- **L'automatisation de tâches web**
- **L'exploration intelligente de contenu**
- **L'assistance utilisateur avancée**
- **L'analyse de sites complexes**

**Prochaines étapes recommandées :**
1. Tester le système avec `python demo_interactive_navigation.py`
2. Commencer par des interactions simples
3. Expérimenter avec vos propres cas d'usage
4. Contribuer aux améliorations du système

---

*Guide mis à jour le 24 juillet 2025 - Version 1.0*
