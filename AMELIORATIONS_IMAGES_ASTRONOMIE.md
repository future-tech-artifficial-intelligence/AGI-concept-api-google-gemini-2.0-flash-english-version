**Improvements for Astronomical Image Analysis**

## Identified Issues

1.  **Repetitive descriptions**: The artificial intelligence API GOOGLE GEMINI 2.0 FLASH systematically begins its astronomical image analyses with "The image shows a map of the night sky with several constellations..." without considering the specificities of each image.

2.  **Lack of variety**: Descriptions of celestial maps are too similar from one image to another, which makes conversations monotonous.

3.  **Ignorance of question context**: The artificial intelligence API GOOGLE GEMINI 2.0 FLASH provides a general description instead of focusing on the specific element mentioned in the user's question.

## Implemented Solutions

### 1. `gemini_api.py` Module

*   Added specific instructions for astronomical images to the system prompt.
*   Instructions to avoid repetitive phrasing.
*   Directives to focus on specific and unique elements of each image.
*   Requirement to adapt the response to the question asked.

```python
ASTRO IMAGES: For images of celestial maps, constellations, or the night sky:
1. ABSOLUTELY AVOID repetitive phrasing such as "The image shows a map of the night sky with several constellations..."
2. Focus on the SPECIFIC ELEMENTS of THIS particular image (precise constellations, planets, moon position, etc.)
3. Adapt your response to the QUESTION ASKED rather than providing a generic description
4. Mention the unique or interesting characteristics of the image (particular alignments, visible phenomena, etc.)
5. Use your astronomical knowledge to provide relevant and varied explanations
```

### 2. `conversation_context_manager.py` Module

*   Added specific detection patterns for astronomical image analysis responses.
*   Improved the `detect_image_analysis` function to better identify astronomical content.

```python
# Keywords specific to astronomical images
astro_keywords = [
    r"(?i)(constellation[s]? (de|du|des))",
    r"(?i)(carte (du|céleste|du ciel))",
    r"(?i)(ciel nocturne)",
    r"(?i)(étoile[s]? (visible|brillante|nommée))",
    r"(?i)(position (de la|des) (lune|planète|étoile))",
    r"(?i)(trajectoire (de|des|du))",
]
```

### 3. `emotional_engine.py` Module

*   Improved the `is_image_analysis_request` function to better detect image analysis requests based on context.
*   Added keywords to identify image analysis requests.

```python
# Check for keywords in the request that suggest image analysis
if 'message' in request_data and isinstance(request_data['message'], str):
    image_request_keywords = [
        r"(?i)(analyse[r]? (cette|l'|l'|une|des|la) image)",
        r"(?i)(que (vois|voit|montre|représente)-tu (sur|dans) (cette|l'|l'|une|des|la) image)",
        r"(?i)(que peux-tu (me dire|dire) (sur|à propos de|de) (cette|l'|l'|une|des|la) image)",
        r"(?i)(décri[s|re|vez] (cette|l'|l'|une|des|la) image)",
        r"(?i)(explique[r|z]? (cette|l'|l'|une|des|la) image)",
    ]
```

## Expected Results

*   Astronomical image analyses will now be more varied and personalized.
*   The artificial intelligence API GOOGLE GEMINI 2.0 FLASH will focus on specific elements mentioned in the user's question.
*   Each image will receive a unique analysis adapted to its actual content.
*   Conversations will be more natural and interactive.

## Tests

To verify that the modifications are effective:
1.  Present several different astronomical images to the artificial intelligence API GOOGLE GEMINI 2.0 FLASH.
2.  Verify that the descriptions are not repetitive.
3.  Ask specific questions about certain elements to test if the artificial intelligence API GOOGLE GEMINI 2.0 FLASH adapts its response.
