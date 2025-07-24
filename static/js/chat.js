document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const imageUpload = document.getElementById('image-upload');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const documentUpload = document.getElementById('document-upload');
    const filePreviewContainer = document.getElementById('file-preview-container');
    const emotionValue = document.getElementById('emotion-value');
    
    let currentImageData = null;
    let currentFileData = null;
    
    // Fonction pour mettre à jour l'affichage de l'état émotionnel
    function updateEmotionalStateDisplay(emotionalState) {
        if (!emotionalState || !emotionValue) return;
        
        // Mettre à jour l'étiquette d'émotion uniquement
        emotionValue.textContent = emotionalState.base_state || 'neutre';
    }

    // Fonction pour ajouter un message à la conversation
    function addMessage(content, isUser = false, imageData = null, fileData = null, emotionalState = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Si c'est un message de l'IA et qu'on a un état émotionnel significatif
        if (!isUser && emotionalState && emotionalState.base_state && 
            !['calm', 'neutral'].includes(emotionalState.base_state.toLowerCase())) {
            const emotionTag = document.createElement('span');
            emotionTag.className = 'emotion-tag';
            emotionTag.dataset.emotion = emotionalState.base_state; // Ajouter l'attribut data pour le CSS
            emotionTag.textContent = emotionalState.base_state;
            messageContent.appendChild(emotionTag);
        }
        
        // Si un message texte est présent
        if (content) {
            // Créer un élément pour le texte du message
            const textElem = document.createElement('span');
            textElem.textContent = content;
            messageContent.appendChild(textElem);
        }
        
        // Si une image est présente
        if (imageData && isUser) {
            const imageElement = document.createElement('img');
            imageElement.src = imageData;
            imageElement.className = 'message-image';
            messageContent.appendChild(imageElement);
        }
        
        // Si un fichier est présent
        if (fileData && isUser) {
            const fileAttachment = document.createElement('div');
            fileAttachment.className = 'file-attachment';
            
            const fileIcon = document.createElement('i');
            fileIcon.className = `fas ${getFileIcon(fileData.file_type)}`;
            fileAttachment.appendChild(fileIcon);
            
            const fileInfo = document.createElement('div');
            fileInfo.className = 'file-attachment-info';
            fileInfo.textContent = `${fileData.filename} (${fileData.file_size})`;
            fileAttachment.appendChild(fileInfo);
            
            messageContent.appendChild(fileAttachment);
        }
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll vers le bas pour voir le nouveau message
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Fonction pour afficher l'animation d'écriture
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'thinking-animation';
        typingDiv.id = 'typing-indicator';
        
        const dotsContainer = document.createElement('div');
        dotsContainer.className = 'thinking-dots';
        
        // Ajouter 3 points d'animation
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'thinking-dot';
            dotsContainer.appendChild(dot);
        }
        
        const textSpan = document.createElement('span');
        textSpan.className = 'thinking-text';
        textSpan.textContent = 'Gemini réfléchit';
        
        typingDiv.appendChild(dotsContainer);
        typingDiv.appendChild(textSpan);
        
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return typingDiv;
    }
    
    // Fonction pour supprimer l'animation d'écriture
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            chatMessages.removeChild(typingIndicator);
        }
    }
    
    // Fonction pour déterminer l'icône de fichier en fonction du type
    function getFileIcon(fileType) {
        const icons = {
            'PDF': 'fa-file-pdf',
            'Word': 'fa-file-word',
            'Text': 'fa-file-alt',
            'CSV': 'fa-file-csv',
            'Excel': 'fa-file-excel',
            'JSON': 'fa-file-code',
            'HTML': 'fa-file-code',
            'XML': 'fa-file-code',
            'Markdown': 'fa-file-alt',
            'Rich Text': 'fa-file-alt'
        };
        
        return icons[fileType] || 'fa-file';
    }

    // Fonction pour récupérer l'état émotionnel actuel
    async function fetchEmotionalState() {
        try {
            const response = await fetch('/api/emotional_state', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching emotional state:', error);
            return null;
        }
    }

    // Fonction pour envoyer un message à l'API
    async function sendMessage(message, imageData = null, fileData = null) {
        try {
            const payload = { message };
            
            // Ajouter l'image en base64 si présente
            if (imageData) {
                payload.image = imageData;
                console.log("Image ajoutée à la requête");
            }
            
            // Ajouter l'ID du fichier si présent
            if (fileData) {
                payload.file_id = fileData.id;
            }
            
            // Ajouter l'ID de session si disponible via l'URL
            const urlParams = new URLSearchParams(window.location.search);
            const sessionId = urlParams.get('session_id');
            if (sessionId) {
                payload.session_id = sessionId;
            }
            
            // Ajouter le fuseau horaire de l'utilisateur
            payload.user_timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
            
            console.log("Envoi de la requête à l'API...");
            
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload),
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Réponse reçue de l'API");
            
            // Si l'état émotionnel est inclus dans la réponse, le mettre à jour
            if (data.emotional_state) {
                updateEmotionalStateDisplay(data.emotional_state);
            }
            
            // Si un ID de session est retourné et que nous n'avons pas d'ID de session dans l'URL
            if (data.session_id && !sessionId) {
                // Mettre à jour l'URL avec le nouvel ID de session sans recharger la page
                const newUrl = new URL(window.location);
                newUrl.searchParams.set('session_id', data.session_id);
                window.history.pushState({}, '', newUrl);
            }
            
            return {
                response: data.response,
                emotionalState: data.emotional_state
            };
        } catch (error) {
            console.error('Error:', error);
            return {
                response: "Désolé, une erreur s'est produite lors de la communication avec Gemini."
            };
        }
    }
    
    // Gestionnaire d'événement pour l'envoi de message
    async function handleSend() {
        const message = userInput.value.trim();
        if (!message && !currentImageData && !currentFileData) return;
        
        // Afficher le message de l'utilisateur
        addMessage(message, true, currentImageData, currentFileData);
        
        // Sauvegarder une copie des données actuelles
        const sentImageData = currentImageData;
        const sentFileData = currentFileData;
        
        // Effacer l'entrée utilisateur et réinitialiser l'image/fichier
        userInput.value = '';
        currentImageData = null;
        imagePreviewContainer.innerHTML = '';
        imageUpload.value = '';
        currentFileData = null;
        filePreviewContainer.innerHTML = '';
        documentUpload.value = '';
        
        // Afficher l'animation d'écriture moderne
        showTypingIndicator();
        
        // Envoyer le message à l'API
        const result = await sendMessage(message, sentImageData, sentFileData);
        
        // Supprimer l'indicateur de chargement
        removeTypingIndicator();
        
        // Afficher la réponse avec l'état émotionnel
        addMessage(result.response, false, null, null, result.emotionalState);
    }

    // Attachement des événements
    sendButton.addEventListener('click', handleSend);
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
    
    // Gestion de l'upload d'image
    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // Vérifier si c'est bien une image
        if (!file.type.match('image.*')) {
            alert('Veuillez sélectionner une image.');
            return;
        }
        
        const reader = new FileReader();
        
        reader.onload = function(e) {
            currentImageData = e.target.result; // Image en base64
            console.log("Image convertie en base64");
            
            // Afficher la prévisualisation
            imagePreviewContainer.innerHTML = '';
            const previewElement = document.createElement('div');
            previewElement.className = 'image-preview';
            
            const img = document.createElement('img');
            img.src = currentImageData;
            
            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-image-btn';
            removeBtn.innerHTML = '<i class="fas fa-times"></i>';
            removeBtn.addEventListener('click', function() {
                currentImageData = null;
                imagePreviewContainer.innerHTML = '';
                imageUpload.value = '';
            });
            
            previewElement.appendChild(img);
            previewElement.appendChild(removeBtn);
            imagePreviewContainer.appendChild(previewElement);
            
            // Réinitialiser le fichier si une image est sélectionnée
            if (currentFileData) {
                currentFileData = null;
                filePreviewContainer.innerHTML = '';
                documentUpload.value = '';
            }
        };
        
        reader.readAsDataURL(file);
    });
    
    // Gestion de l'upload de document
    documentUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // Vérifier l'extension du fichier
        const fileExt = file.name.split('.').pop().toLowerCase();
        const allowedExts = ['pdf', 'doc', 'docx', 'txt', 'csv', 'xls', 'xlsx', 'json', 'html', 'xml', 'md', 'rtf'];
        
        if (!allowedExts.includes(fileExt)) {
            alert('Type de fichier non pris en charge.');
            return;
        }
        
        // Créer un FormData pour l'upload
        const formData = new FormData();
        formData.append('file', file);
        
        // Afficher un indicateur de chargement
        filePreviewContainer.innerHTML = '<div class="file-loading">Chargement du fichier...</div>';
        
        // Envoyer le fichier au serveur
        fetch('/api/upload-file', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Erreur HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Stocker les données du fichier
            currentFileData = data;
            
            // Afficher la prévisualisation du fichier
            filePreviewContainer.innerHTML = '';
            const previewElement = document.createElement('div');
            previewElement.className = 'file-preview';
            
            const fileIcon = document.createElement('i');
            fileIcon.className = `fas ${getFileIcon(data.file_type)}`;
            previewElement.appendChild(fileIcon);
            
            const fileInfo = document.createElement('div');
            fileInfo.className = 'file-info';
            
            const fileName = document.createElement('div');
            fileName.className = 'file-name';
            fileName.textContent = data.original_filename;
            fileInfo.appendChild(fileName);
            
            const fileSize = document.createElement('div');
            fileSize.className = 'file-size';
            fileSize.textContent = data.file_size;
            fileInfo.appendChild(fileSize);
            
            previewElement.appendChild(fileInfo);
            
            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-file-btn';
            removeBtn.innerHTML = '<i class="fas fa-times"></i>';
            removeBtn.addEventListener('click', function() {
                currentFileData = null;
                filePreviewContainer.innerHTML = '';
                documentUpload.value = '';
            });
            
            previewElement.appendChild(removeBtn);
            filePreviewContainer.appendChild(previewElement);
            
            // Réinitialiser l'image si un fichier est sélectionné
            if (currentImageData) {
                currentImageData = null;
                imagePreviewContainer.innerHTML = '';
                imageUpload.value = '';
            }
        })
        .catch(error => {
            console.error('Erreur:', error);
            filePreviewContainer.innerHTML = `<div class="file-error">Erreur: ${error.message}</div>`;
        });
    });
    
    // Fonction pour optimiser l'espacement des messages longs sur mobile
    function optimizeMessageSpacing() {
        // Vérifier si on est sur mobile
        if (window.innerWidth <= 768) {
            // Cibler tous les messages
            const messages = document.querySelectorAll('.message');
            
            messages.forEach(msg => {
                // Si le contenu est long (+ de 200 caractères), ajouter une classe spéciale
                if (msg.textContent.length > 200) {
                    msg.classList.add('long-content');
                }
                
                // Améliorer l'espacement dans les listes
                const lists = msg.querySelectorAll('ul, ol');
                lists.forEach(list => {
                    list.style.marginTop = '1rem';
                    list.style.marginBottom = '1rem';
                });
                
                // Améliorer l'espacement des paragraphes
                const paragraphs = msg.querySelectorAll('p');
                if (paragraphs.length > 3) {
                    paragraphs.forEach(p => {
                        p.style.marginBottom = '1.2rem';
                    });
                }
            });
        }
    }
    
    // Exécuter la fonction au chargement et au resize
    optimizeMessageSpacing();
    window.addEventListener('resize', optimizeMessageSpacing);
    
    // Observer les nouveaux messages pour leur appliquer l'optimisation
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                optimizeMessageSpacing();
            }
        });
    });
    
    // Observer les changements dans le conteneur de messages
    if (chatMessages) {
        observer.observe(chatMessages, { childList: true });
    }
    
    // Configuration des voix de synthèse
    let voiceConfig = {
        language: 'fr-FR',
        voice: null,
        rate: 0.9,
        pitch: 1.0,
        volume: 1.0
    };

    // Charger la configuration des voix depuis le localStorage
    function loadVoiceConfig() {
        const saved = localStorage.getItem('voiceConfig');
        if (saved) {
            try {
                const savedConfig = JSON.parse(saved);
                voiceConfig = { ...voiceConfig, ...savedConfig };
            } catch (e) {
                console.error('Erreur lors du chargement de la configuration des voix:', e);
            }
        }
    }

    // Sauvegarder la configuration des voix dans le localStorage
    function saveVoiceConfig() {
        try {
            localStorage.setItem('voiceConfig', JSON.stringify(voiceConfig));
        } catch (e) {
            console.error('Erreur lors de la sauvegarde de la configuration des voix:', e);
        }
    }

    // Fonction de synthèse vocale améliorée avec configuration
    function speakText(text) {
        // Vérifier si la synthèse vocale est supportée
        if ('speechSynthesis' in window && typeof speechSynthesis !== 'undefined') {
            // Arrêter toute synthèse en cours
            speechSynthesis.cancel();
            
            // Créer une nouvelle instance de SpeechSynthesisUtterance
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Appliquer la configuration
            utterance.lang = voiceConfig.language;
            utterance.rate = voiceConfig.rate;
            utterance.pitch = voiceConfig.pitch;
            utterance.volume = voiceConfig.volume;
            
            // Utiliser la voix configurée si disponible
            if (voiceConfig.voice) {
                const voices = speechSynthesis.getVoices();
                const selectedVoice = voices.find(voice => voice.name === voiceConfig.voice);
                if (selectedVoice) {
                    utterance.voice = selectedVoice;
                }
            } else {
                // Sélection automatique de la voix selon la langue
                const voices = speechSynthesis.getVoices();
                const languageVoices = voices.filter(voice => voice.lang === voiceConfig.language);
                if (languageVoices.length > 0) {
                    utterance.voice = languageVoices[0];
                }
            }
            
            // Indicateur visuel pendant la lecture
            const voiceBtn = document.getElementById('voice-synthesis-btn');
            if (voiceBtn) {
                const originalIcon = voiceBtn.innerHTML;
                
                utterance.onstart = function() {
                    voiceBtn.innerHTML = '<i class="fas fa-stop" style="color: #e74c3c;"></i>';
                    voiceBtn.title = 'Arrêter la lecture';
                    voiceBtn.style.borderColor = '#e74c3c';
                };
                
                utterance.onend = function() {
                    voiceBtn.innerHTML = originalIcon;
                    voiceBtn.title = 'Lire avec la synthèse vocale';
                    voiceBtn.style.borderColor = '#6c5ce7';
                };
                
                utterance.onerror = function() {
                    voiceBtn.innerHTML = originalIcon;
                    voiceBtn.title = 'Lire avec la synthèse vocale';
                    voiceBtn.style.borderColor = '#6c5ce7';
                };
            }
            
            // Lancer la synthèse
            speechSynthesis.speak(utterance);
        } else {
            console.error('La synthèse vocale n\'est pas supportée par ce navigateur');
            alert('La synthèse vocale n\'est pas supportée par votre navigateur');
        }
    }

    // Gestion de la modale de configuration des voix
    const voiceConfigBtn = document.getElementById('voice-config-btn');
    const voiceConfigModal = document.getElementById('voice-config-modal');
    const closeVoiceModal = document.getElementById('close-voice-modal');
    const voiceLanguageSelect = document.getElementById('voice-language-select');
    const voiceSelect = document.getElementById('voice-select');
    const voiceRateSlider = document.getElementById('voice-rate');
    const voicePitchSlider = document.getElementById('voice-pitch');
    const voiceVolumeSlider = document.getElementById('voice-volume');
    const testVoiceBtn = document.getElementById('test-voice-btn');
    const saveVoiceConfigBtn = document.getElementById('save-voice-config-btn');
    const resetVoiceConfigBtn = document.getElementById('reset-voice-config-btn');

    // Charger la configuration des voix au démarrage
    loadVoiceConfig();

    // Ouvrir la modale de configuration des voix
    if (voiceConfigBtn) {
        voiceConfigBtn.addEventListener('click', function() {
            voiceConfigModal.style.display = 'block';
            loadAvailableVoices();
            updateVoiceConfigUI();
        });
    }

    // Fermer la modale de configuration des voix
    if (closeVoiceModal) {
        closeVoiceModal.addEventListener('click', function() {
            voiceConfigModal.style.display = 'none';
        });
    }

    // Fermer la modale en cliquant en dehors
    window.addEventListener('click', function(event) {
        if (event.target === voiceConfigModal) {
            voiceConfigModal.style.display = 'none';
        }
    });

    // Charger les voix disponibles
    function loadAvailableVoices() {
        const voices = speechSynthesis.getVoices();
        const selectedLanguage = voiceLanguageSelect.value;
        const filteredVoices = voices.filter(voice => voice.lang === selectedLanguage);

        voiceSelect.innerHTML = '';
        
        if (filteredVoices.length === 0) {
            voiceSelect.innerHTML = '<option value="">Aucune voix disponible</option>';
            return;
        }

        filteredVoices.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice.name;
            option.textContent = `${voice.name} ${voice.localService ? '(Local)' : '(En ligne)'}`;
            voiceSelect.appendChild(option);
        });

        // Sélectionner la voix configurée si elle existe
        if (voiceConfig.voice) {
            voiceSelect.value = voiceConfig.voice;
        }
    }

    // Mettre à jour l'interface utilisateur avec la configuration actuelle
    function updateVoiceConfigUI() {
        if (voiceLanguageSelect) voiceLanguageSelect.value = voiceConfig.language;
        if (voiceRateSlider) voiceRateSlider.value = voiceConfig.rate;
        if (voicePitchSlider) voicePitchSlider.value = voiceConfig.pitch;
        if (voiceVolumeSlider) voiceVolumeSlider.value = voiceConfig.volume;
        
        const rateValue = document.getElementById('voice-rate-value');
        const pitchValue = document.getElementById('voice-pitch-value');
        const volumeValue = document.getElementById('voice-volume-value');
        
        if (rateValue) rateValue.textContent = voiceConfig.rate;
        if (pitchValue) pitchValue.textContent = voiceConfig.pitch;
        if (volumeValue) volumeValue.textContent = voiceConfig.volume;
    }

    // Gestionnaire de changement de langue
    if (voiceLanguageSelect) {
        voiceLanguageSelect.addEventListener('change', function() {
            voiceConfig.language = this.value;
            loadAvailableVoices();
        });
    }

    // Gestionnaire de changement de voix
    if (voiceSelect) {
        voiceSelect.addEventListener('change', function() {
            voiceConfig.voice = this.value;
        });
    }

    // Gestionnaires des sliders
    if (voiceRateSlider) {
        voiceRateSlider.addEventListener('input', function() {
            voiceConfig.rate = parseFloat(this.value);
            document.getElementById('voice-rate-value').textContent = this.value;
        });
    }

    if (voicePitchSlider) {
        voicePitchSlider.addEventListener('input', function() {
            voiceConfig.pitch = parseFloat(this.value);
            document.getElementById('voice-pitch-value').textContent = this.value;
        });
    }

    if (voiceVolumeSlider) {
        voiceVolumeSlider.addEventListener('input', function() {
            voiceConfig.volume = parseFloat(this.value);
            document.getElementById('voice-volume-value').textContent = this.value;
        });
    }

    // Tester la voix
    if (testVoiceBtn) {
        testVoiceBtn.addEventListener('click', function() {
            const testText = "Ceci est un test de la synthèse vocale. Hello, this is a voice synthesis test.";
            speakText(testText);
        });
    }

    // Sauvegarder la configuration
    if (saveVoiceConfigBtn) {
        saveVoiceConfigBtn.addEventListener('click', function() {
            saveVoiceConfig();
            alert('Configuration des voix sauvegardée !');
        });
    }

    // Réinitialiser la configuration
    if (resetVoiceConfigBtn) {
        resetVoiceConfigBtn.addEventListener('click', function() {
            voiceConfig = {
                language: 'fr-FR',
                voice: null,
                rate: 0.9,
                pitch: 1.0,
                volume: 1.0
            };
            updateVoiceConfigUI();
            loadAvailableVoices();
            saveVoiceConfig();
            alert('Configuration des voix réinitialisée !');
        });
    }

    // Charger les voix lors du changement de voix disponibles
    if ('speechSynthesis' in window) {
        speechSynthesis.onvoiceschanged = function() {
            if (voiceConfigModal && voiceConfigModal.style.display === 'block') {
                loadAvailableVoices();
            }
        };
        
        // Forcer le chargement des voix
        speechSynthesis.getVoices();
    }

    // Exposer des fonctions pour être utilisées par d'autres scripts
    window.addMessage = addMessage;
    window.getFileIcon = getFileIcon;
    window.updateEmotionalStateDisplay = updateEmotionalStateDisplay;
    window.sendMessage = sendMessage;
    window.showTypingIndicator = showTypingIndicator;
    window.removeTypingIndicator = removeTypingIndicator;
    window.speakText = speakText;
    window.loadVoiceConfig = loadVoiceConfig;
    window.saveVoiceConfig = saveVoiceConfig;
});
