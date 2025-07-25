# Multi-API Implementation

This extension allows the use of different artificial intelligence APIs within GeminiChat.

## API Compatibility

GeminiChat now supports multiple artificial intelligence API providers:

-   **Google Gemini** (default) - artificial intelligence API GOOGLE GEMINI 2.0 FLASH
-   **Claude by Anthropic** - Support for Claude-3 Opus

Other providers can be easily added by implementing the `AIApiInterface`.

## Architecture

The system uses a modular architecture with the following components:

-   `ai_api_interface.py` - Abstract interface that all API implementations must follow
-   `ai_api_manager.py` - Manager centralizing access to different artificial intelligence APIs
-   `gemini_api_adapter.py` - Interface implementation for Google Gemini
-   `claude_api_adapter.py` - Interface implementation for Claude by Anthropic

## Configuration

### User Interface

A user interface is available to configure the APIs:
1.  Log in to your GeminiChat account
2.  Click on "Config API" in the navigation menu
3.  For each API:
    -   Enter your API key
    -   Click on "Save Key"
    -   Click on "Activate this API" to use it

### File Configuration

You can also configure the APIs via the `ai_api_config.json` file:

```json
{
    "default_api": "gemini",
    "apis": {
        "gemini": {
            "api_key": "votre_clé_api_gemini",
            "api_url": null
        },
        "claude": {
            "api_key": "votre_clé_api_claude",
            "api_url": null
        }
    }
}
```

## Add a New API

To add support for a new artificial intelligence API:

1.  Create a new class implementing `AIApiInterface`
2.  Register this class with the `AIApiManager`
3.  Update the configuration to include the new API's parameters

Example of registering a new API:

```python
from my_new_api_adapter import MyNewAPI
from ai_api_manager import get_ai_api_manager

api_manager = get_ai_api_manager()
api_manager.add_api_implementation('my_new_api', MyNewAPI)
```

## REST API for API Management

The system exposes several REST endpoints to manage the APIs:

-   `GET /api/config/apis` - List of available APIs
-   `GET /api/config/apis/current` - Currently active API
-   `POST /api/config/apis/current` - Change the active API
-   `GET /api/keys` - List of configured API keys
-   `POST /api/keys/{api_name}` - Configure an API key
-   `DELETE /api/keys/{api_name}` - Delete an API key
