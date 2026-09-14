from unittest.mock import patch

from ollama import ChatResponse, Message

from slm_gerenciadordespesas.config import OllamaConfig
from slm_gerenciadordespesas.controller.ollama_classifier import (
    OllamaClassifier
)
from slm_gerenciadordespesas.model.category import Category

def test_ollama_classifier_retries_after_invalid_response():
    classifier = OllamaClassifier(
        OllamaConfig(
            model="qwen3:1.7b",
            temperature=0.0,
        ),
        max_retries=2,
    )

    invalid_response = ChatResponse(message=Message(
        role="assistant",
        content="""
    {
        "description": "Netflix",
        "category": "Cinema"
    }
    """,
    ))

    valid_response = ChatResponse(message=Message(
        role="assistant",
        content="""
    {
        "description": "Netflix",
        "category": "Streaming"
    }
    """,
    ))

    with patch(
        "slm_gerenciadordespesas.controller.ollama_classifier.ollama.chat",
        side_effect=[
            invalid_response,
            valid_response,
        ],
    ) as mock_chat:
        result = classifier.classify("Netflix")

    assert result.category == Category.STREAMING
    assert mock_chat.call_count == 2