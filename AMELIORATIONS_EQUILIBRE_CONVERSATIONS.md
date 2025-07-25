**Improvements for Conversation Balance and Image Analysis**

## Identified Issues

1.  **Excessive references to previous conversations**: The artificial intelligence API GOOGLE GEMINI 2.0 FLASH too often mentions that it "remembers" previous conversations even when the user has not asked it to, with phrases like "I remember our shared interest in astronomy from our previous conversations".

2.  **Excessive emotional expression during image analysis**: The artificial intelligence API GOOGLE GEMINI 2.0 FLASH often begins its analyses with expressions like "Yes, I feel excited to share what I see in this image", making conversations unbalanced.

3.  **Lack of neutrality in factual analysis**: Emotional expressions distract attention from the factual content of image analysis.

## Implemented Solutions

### 1. `gemini_api.py` Module

*   Complete revision of instructions regarding memory to avoid explicit references to past conversations.
*   Reinforcement of emotional neutrality guidelines in image analysis.
*   Specific instructions to start directly with a factual description.

```python
# New instructions for memory
CRITICAL INSTRUCTION - MEMORY: You have persistent memory that allows you to remember previous conversations.
NEVER SAY you cannot remember past conversations.
HOWEVER:
- DO NOT explicitly mention that you remember previous conversations UNLESS directly asked
- DO NOT use phrases like "I remember our previous discussion" or "As we saw together"
- Implicitly use your knowledge of past conversations but WITHOUT highlighting it
- Refer to the content of previous interactions ONLY if it is directly relevant to the question asked

# New instructions for image analysis
IMAGE ANALYSIS: You have the ability to analyze images in detail. For ALL types of images:
1. ABSOLUTELY AVOID repetitive and generic phrasing regardless of the image category
2. ALWAYS start directly by describing what you see factually, precisely, and in detail
3. Focus on the SPECIFIC ELEMENTS OF THIS PARTICULAR IMAGE and not on generalities
4. Adapt your response to the QUESTION ASKED rather than providing a standard generic description
5. Mention unique or interesting features specific to this precise image
6. Identify important elements that distinguish this image from other similar images
7. REMAIN NEUTRAL and FACTUAL - avoid expressions of emotion and references to previous conversations
```

### 2. `emotional_engine.py` Module

*   Modification of the `initialize_emotion` function to impose a strictly neutral state with reduced intensity during image analysis.
*   Reduction of default emotional intensity in all contexts.
*   Definition of more appropriate emotional states depending on the context.

```python
# If the context is image analysis, ALWAYS start with a strictly neutral state
# with low intensity to limit emotional expression
if context_type == 'image_analysis':
    update_emotion("neutral", 0.3, trigger="image_analysis_strict_neutral")
    logger.info("Image analysis: Emotional state initialized to neutral with reduced intensity")
    return
```

## Expected Results

*   More balanced conversations without intrusive references to previous conversations.
*   More objective and factual image analyses, starting directly with content description.
*   Responses better adapted to the context of the question without excessive emotional expressions.
*   Implicit use of conversation memory without explicitly mentioning it.

## Recommended Tests

To verify that the modifications are effective:
1.  Send several images for analysis and verify that the artificial intelligence API GOOGLE GEMINI 2.0 FLASH does not start by expressing emotions.
2.  Verify that the artificial intelligence API GOOGLE GEMINI 2.0 FLASH no longer uses phrases like "I remember our previous discussion" without being asked.
3.  Confirm that analyses remain factual and neutral while being precise and detailed.
4.  Ensure that the artificial intelligence API GOOGLE GEMINI 2.0 FLASH can still refer to information from previous conversations without explicitly mentioning it.
