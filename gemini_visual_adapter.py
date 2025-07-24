"""
Adaptateur Vision Avancé pour l'API Gemini
Permet à Gemini de "voir" et analyser visuellement l'intérieur des sites web
"""

import base64
import json
import logging
import requests
import io
from PIL import Image
from typing import Dict, List, Any, Optional, Union, Tuple
import os
from datetime import datetime

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiVisualAdapter:
    """Adaptateur pour les capacités visuelles avancées de Gemini"""
    
    def __init__(self, api_key: str = None):
        """
        Initialise l'adaptateur vision Gemini
        
        Args:
            api_key: Clé API Gemini (utilise la clé par défaut si non spécifiée)
        """
        self.api_key = api_key or "AIzaSyDdWKdpPqgAVLet6_mchFxmG_GXnfPx2aQ"
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        
        # Configuration pour l'optimisation d'images
        self.max_image_size = (1024, 1024)  # Taille max pour l'IA
        self.image_quality = 85  # Qualité JPEG pour optimiser
        
        # Statistiques
        self.stats = {
            'images_processed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_processing_time': 0
        }
        
        logger.info("🤖 Adaptateur Vision Gemini initialisé")
    
    def encode_image_for_gemini(self, image_path: str) -> Optional[str]:
        """
        Encode une image pour l'API Gemini multimodale
        
        Args:
            image_path: Chemin vers l'image à encoder
            
        Returns:
            Image encodée en base64 ou None si erreur
        """
        try:
            # Ouvrir et optimiser l'image
            with Image.open(image_path) as img:
                # Convertir en RGB si nécessaire
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Redimensionner si trop grande
                if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                    img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                    logger.info(f"📏 Image redimensionnée: {img.size}")
                
                # Sauvegarder en mémoire
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=self.image_quality, optimize=True)
                
                # Encoder en base64
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                logger.info(f"✅ Image encodée: {len(image_data)} caractères")
                return image_data
                
        except Exception as e:
            logger.error(f"❌ Erreur encodage image {image_path}: {e}")
            return None
    
    def analyze_website_screenshot(self, 
                                 image_path: str, 
                                 analysis_prompt: str,
                                 context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyse une capture d'écran de site web avec Gemini Vision
        
        Args:
            image_path: Chemin vers la capture d'écran
            analysis_prompt: Prompt d'analyse spécifique
            context: Contexte textuel additionnel
            
        Returns:
            Résultat de l'analyse avec métadonnées
        """
        start_time = datetime.now()
        
        try:
            # Encoder l'image
            encoded_image = self.encode_image_for_gemini(image_path)
            if not encoded_image:
                return {
                    'success': False,
                    'error': 'Impossible d\'encoder l\'image',
                    'analysis': None
                }
            
            # Construire le prompt d'analyse visuelle
            context_text = context or "Analyse générale d'un site web"
            visual_prompt = f"""🤖 ANALYSE VISUELLE D'UN SITE WEB

**CONTEXTE**: {context_text}

**INSTRUCTIONS D'ANALYSE**:
{analysis_prompt}

**TÂCHES SPÉCIFIQUES**:
1. 📋 **Structure et Layout**: Décrivez la structure générale, navigation, zones principales
2. 🎨 **Design et UX**: Analysez les couleurs, polices, espacement, lisibilité
3. 📝 **Contenu Visible**: Identifiez et résumez le contenu textuel principal
4. 🔗 **Éléments Interactifs**: Boutons, liens, formulaires, menus visibles
5. 📱 **Responsive Design**: Indices sur l'adaptation mobile/desktop
6. ⚡ **Problèmes Potentiels**: Erreurs, éléments cassés, problèmes d'accessibilité
7. 🎯 **Objectif du Site**: Déterminez le but principal de la page
8. 💡 **Recommandations**: Suggestions d'amélioration UX/UI

**FORMAT DE RÉPONSE**: Structurez votre analyse avec ces sections et utilisez des emojis pour la lisibilité.
"""

            # Préparer la requête multimodale
            headers = {
                'Content-Type': 'application/json'
            }
            
            data = {
                "contents": [{
                    "parts": [
                        {
                            "text": visual_prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": encoded_image
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.4,  # Plus bas pour analyses précises
                    "topK": 32,
                    "topP": 0.8,
                    "maxOutputTokens": 3000,  # Plus élevé pour analyses détaillées
                }
            }
            
            # Envoyer la requête
            url = f"{self.api_url}?key={self.api_key}"
            logger.info("📤 Envoi requête d'analyse visuelle à Gemini...")
            
            response = requests.post(url, headers=headers, json=data, timeout=120)
            
            # Traiter la réponse
            if response.status_code == 200:
                response_data = response.json()
                
                if 'candidates' in response_data and response_data['candidates']:
                    analysis = response_data['candidates'][0]['content']['parts'][0]['text']
                    
                    # Calculer les métriques
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    # Mettre à jour les statistiques
                    self.stats['images_processed'] += 1
                    self.stats['successful_analyses'] += 1
                    self.stats['total_processing_time'] += processing_time
                    
                    logger.info(f"✅ Analyse visuelle réussie en {processing_time:.2f}s")
                    
                    return {
                        'success': True,
                        'analysis': analysis,
                        'image_path': image_path,
                        'processing_time': processing_time,
                        'image_size': os.path.getsize(image_path) if os.path.exists(image_path) else 0,
                        'analysis_length': len(analysis),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    error_msg = "Aucune réponse valide de Gemini"
                    logger.error(f"❌ {error_msg}")
                    self.stats['failed_analyses'] += 1
                    
                    return {
                        'success': False,
                        'error': error_msg,
                        'analysis': None
                    }
            else:
                error_msg = f"Erreur API Gemini: {response.status_code} - {response.text}"
                logger.error(f"❌ {error_msg}")
                self.stats['failed_analyses'] += 1
                
                return {
                    'success': False,
                    'error': error_msg,
                    'analysis': None
                }
                
        except Exception as e:
            error_msg = f"Erreur analyse visuelle: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.stats['failed_analyses'] += 1
            
            return {
                'success': False,
                'error': error_msg,
                'analysis': None
            }
    
    def compare_website_changes(self, 
                              image_path_before: str,
                              image_path_after: str,
                              comparison_context: str = "") -> Dict[str, Any]:
        """
        Compare deux captures d'écran pour détecter les changements
        
        Args:
            image_path_before: Capture avant
            image_path_after: Capture après
            comparison_context: Contexte de la comparaison
            
        Returns:
            Analyse comparative
        """
        try:
            # Encoder les deux images
            encoded_before = self.encode_image_for_gemini(image_path_before)
            encoded_after = self.encode_image_for_gemini(image_path_after)
            
            if not encoded_before or not encoded_after:
                return {
                    'success': False,
                    'error': 'Impossible d\'encoder une ou plusieurs images',
                    'comparison': None
                }
            
            # Prompt de comparaison
            comparison_prompt = f"""🔍 COMPARAISON VISUELLE DE SITES WEB

**CONTEXTE**: {comparison_context}

**INSTRUCTIONS**:
Comparez ces deux captures d'écran et identifiez:

1. 🆚 **Différences Visuelles**: Changements de layout, couleurs, éléments
2. ➕ **Nouveaux Éléments**: Ce qui a été ajouté
3. ➖ **Éléments Supprimés**: Ce qui a disparu
4. 🔄 **Modifications**: Éléments modifiés (texte, position, style)
5. 📊 **Impact UX**: Comment ces changements affectent l'expérience utilisateur
6. ⚖️ **Évaluation Globale**: Les changements sont-ils positifs ou négatifs?

**PREMIÈRE IMAGE (AVANT)**:
"""

            # Construire la requête avec les deux images
            data = {
                "contents": [{
                    "parts": [
                        {"text": comparison_prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": encoded_before
                            }
                        },
                        {"text": "\n\n**DEUXIÈME IMAGE (APRÈS)**:"},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg", 
                                "data": encoded_after
                            }
                        },
                        {"text": "\n\nVeuillez maintenant effectuer la comparaison détaillée."}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "topK": 32,
                    "topP": 0.8,
                    "maxOutputTokens": 3000,
                }
            }
            
            headers = {'Content-Type': 'application/json'}
            url = f"{self.api_url}?key={self.api_key}"
            
            logger.info("🔍 Envoi requête de comparaison visuelle...")
            response = requests.post(url, headers=headers, json=data, timeout=120)
            
            if response.status_code == 200:
                response_data = response.json()
                
                if 'candidates' in response_data and response_data['candidates']:
                    comparison = response_data['candidates'][0]['content']['parts'][0]['text']
                    
                    logger.info("✅ Comparaison visuelle réussie")
                    
                    return {
                        'success': True,
                        'comparison': comparison,
                        'image_before': image_path_before,
                        'image_after': image_path_after,
                        'timestamp': datetime.now().isoformat()
                    }
            
            return {
                'success': False,
                'error': f"Erreur API: {response.status_code}",
                'comparison': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Erreur comparaison: {str(e)}",
                'comparison': None
            }
    
    def analyze_ui_elements(self, image_path: str, element_types: List[str] = None) -> Dict[str, Any]:
        """
        Analyse spécifique des éléments UI dans une capture d'écran
        
        Args:
            image_path: Chemin vers la capture
            element_types: Types d'éléments à analyser (buttons, forms, navigation, etc.)
            
        Returns:
            Analyse détaillée des éléments UI
        """
        if element_types is None:
            element_types = ['buttons', 'forms', 'navigation', 'content', 'images', 'links']
        
        elements_list = ", ".join(element_types)
        
        ui_prompt = f"""🎯 ANALYSE SPÉCIALISÉE DES ÉLÉMENTS UI

**FOCUS SUR**: {elements_list}

**INSTRUCTIONS DÉTAILLÉES**:
1. 🔘 **Boutons**: Identifiez tous les boutons (CTA, navigation, action)
2. 📝 **Formulaires**: Champs, labels, validation, accessibilité
3. 🧭 **Navigation**: Menus, breadcrumbs, liens de navigation
4. 📄 **Contenu**: Hiérarchie, lisibilité, organisation
5. 🖼️ **Images**: Pertinence, qualité, optimisation
6. 🔗 **Liens**: Visibilité, différenciation, call-to-action

**POUR CHAQUE ÉLÉMENT**:
- Position et visibilité
- État (actif, hover, disabled)
- Accessibilité (contraste, taille)
- Cohérence avec le design system
- Recommandations d'amélioration

**FORMAT**: Organisez par type d'élément avec évaluation de 1-5 ⭐
"""

        return self.analyze_website_screenshot(
            image_path=image_path,
            analysis_prompt=ui_prompt,
            context=f"Analyse UI spécialisée - Focus sur: {elements_list}"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne les statistiques d'utilisation de l'adaptateur
        
        Returns:
            Dictionnaire avec les statistiques
        """
        avg_processing_time = (
            self.stats['total_processing_time'] / max(self.stats['images_processed'], 1)
        )
        
        success_rate = (
            self.stats['successful_analyses'] / max(self.stats['images_processed'], 1) * 100
        )
        
        return {
            'images_processed': self.stats['images_processed'],
            'successful_analyses': self.stats['successful_analyses'],
            'failed_analyses': self.stats['failed_analyses'],
            'success_rate': round(success_rate, 2),
            'average_processing_time': round(avg_processing_time, 2),
            'total_processing_time': round(self.stats['total_processing_time'], 2)
        }
    
    def reset_statistics(self):
        """Remet à zéro les statistiques"""
        self.stats = {
            'images_processed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'total_processing_time': 0
        }
        logger.info("📊 Statistiques remises à zéro")

# Instance globale pour utilisation facile
gemini_visual_adapter = None

def initialize_gemini_visual_adapter(api_key: str = None) -> GeminiVisualAdapter:
    """
    Initialise l'adaptateur vision Gemini global
    
    Args:
        api_key: Clé API optionnelle
        
    Returns:
        Instance de l'adaptateur
    """
    global gemini_visual_adapter
    
    if gemini_visual_adapter is None:
        gemini_visual_adapter = GeminiVisualAdapter(api_key)
        logger.info("🚀 Adaptateur Vision Gemini global initialisé")
    
    return gemini_visual_adapter

def get_gemini_visual_adapter() -> Optional[GeminiVisualAdapter]:
    """
    Retourne l'instance globale de l'adaptateur vision
    
    Returns:
        Instance de l'adaptateur ou None si non initialisé
    """
    global gemini_visual_adapter
    return gemini_visual_adapter

if __name__ == "__main__":
    # Test de l'adaptateur
    adapter = initialize_gemini_visual_adapter()
    print("🧪 Adaptateur Vision Gemini prêt pour les tests")
    print(f"📊 Statistiques: {adapter.get_statistics()}")
