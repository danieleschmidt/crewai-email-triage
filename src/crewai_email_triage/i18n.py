"""Internationalization and localization support."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

from .logging_utils import get_logger

logger = get_logger(__name__)

# Default translations
TRANSLATIONS = {
    "en": {
        "categories": {
            "urgent": "urgent",
            "work": "work",
            "spam": "spam",
            "general": "general",
            "empty": "empty",
            "unknown": "unknown"
        },
        "responses": {
            "default": "Thanks for your email",
            "urgent": "I understand this is urgent and will address it immediately with high priority.",
            "work": "Thank you for your work-related message. I will review and respond appropriately.",
            "spam": "This message has been identified as spam and filtered accordingly.",
            "empty": "No content to process"
        },
        "errors": {
            "processing_failed": "Processing failed",
            "unable_to_process": "Unable to process message",
            "validation_failed": "validation_failed",
            "sanitization_failed": "sanitization_failed"
        }
    },
    "es": {
        "categories": {
            "urgent": "urgente",
            "work": "trabajo",
            "spam": "spam",
            "general": "general",
            "empty": "vacío",
            "unknown": "desconocido"
        },
        "responses": {
            "default": "Gracias por tu correo electrónico",
            "urgent": "Entiendo que esto es urgente y lo abordaré inmediatamente con alta prioridad.",
            "work": "Gracias por tu mensaje relacionado con el trabajo. Lo revisaré y responderé apropiadamente.",
            "spam": "Este mensaje ha sido identificado como spam y filtrado en consecuencia.",
            "empty": "Sin contenido que procesar"
        },
        "errors": {
            "processing_failed": "Error en el procesamiento",
            "unable_to_process": "No se puede procesar el mensaje",
            "validation_failed": "validación_falló",
            "sanitization_failed": "sanitización_falló"
        }
    },
    "fr": {
        "categories": {
            "urgent": "urgent",
            "work": "travail",
            "spam": "spam",
            "general": "général",
            "empty": "vide",
            "unknown": "inconnu"
        },
        "responses": {
            "default": "Merci pour votre email",
            "urgent": "Je comprends que c'est urgent et je vais m'en occuper immédiatement avec une priorité élevée.",
            "work": "Merci pour votre message lié au travail. Je vais l'examiner et répondre de manière appropriée.",
            "spam": "Ce message a été identifié comme spam et filtré en conséquence.",
            "empty": "Aucun contenu à traiter"
        },
        "errors": {
            "processing_failed": "Échec du traitement",
            "unable_to_process": "Impossible de traiter le message",
            "validation_failed": "échec_validation",
            "sanitization_failed": "échec_assainissement"
        }
    },
    "de": {
        "categories": {
            "urgent": "dringend",
            "work": "arbeit",
            "spam": "spam",
            "general": "allgemein",
            "empty": "leer",
            "unknown": "unbekannt"
        },
        "responses": {
            "default": "Danke für Ihre E-Mail",
            "urgent": "Ich verstehe, dass dies dringend ist und werde es sofort mit hoher Priorität bearbeiten.",
            "work": "Vielen Dank für Ihre arbeitsbezogene Nachricht. Ich werde sie überprüfen und angemessen antworten.",
            "spam": "Diese Nachricht wurde als Spam identifiziert und entsprechend gefiltert.",
            "empty": "Kein Inhalt zu verarbeiten"
        },
        "errors": {
            "processing_failed": "Verarbeitung fehlgeschlagen",
            "unable_to_process": "Nachricht kann nicht verarbeitet werden",
            "validation_failed": "validierung_fehlgeschlagen",
            "sanitization_failed": "bereinigung_fehlgeschlagen"
        }
    },
    "ja": {
        "categories": {
            "urgent": "緊急",
            "work": "仕事",
            "spam": "スパム",
            "general": "一般",
            "empty": "空",
            "unknown": "不明"
        },
        "responses": {
            "default": "メールをありがとうございます",
            "urgent": "これが緊急であることを理解し、最優先で直ちに対処いたします。",
            "work": "お仕事関連のメッセージをありがとうございます。確認して適切に対応いたします。",
            "spam": "このメッセージはスパムとして識別され、それに応じてフィルタリングされました。",
            "empty": "処理するコンテンツがありません"
        },
        "errors": {
            "processing_failed": "処理に失敗しました",
            "unable_to_process": "メッセージを処理できません",
            "validation_failed": "検証_失敗",
            "sanitization_failed": "無害化_失敗"
        }
    },
    "zh": {
        "categories": {
            "urgent": "紧急",
            "work": "工作",
            "spam": "垃圾邮件",
            "general": "一般",
            "empty": "空",
            "unknown": "未知"
        },
        "responses": {
            "default": "感谢您的电子邮件",
            "urgent": "我理解这很紧急，将立即以最高优先级处理。",
            "work": "感谢您与工作相关的消息。我将审查并适当回复。",
            "spam": "此消息已被识别为垃圾邮件并相应过滤。",
            "empty": "没有内容需要处理"
        },
        "errors": {
            "processing_failed": "处理失败",
            "unable_to_process": "无法处理消息",
            "validation_failed": "验证_失败",
            "sanitization_failed": "清理_失败"
        }
    }
}


class I18nManager:
    """Internationalization and localization manager."""

    def __init__(self, default_language: str = "en"):
        self.default_language = default_language
        self.current_language = default_language
        self.translations = TRANSLATIONS.copy()

        # Try to load from environment
        env_lang = os.environ.get('TRIAGE_LANGUAGE', '').lower()
        if env_lang and env_lang in self.translations:
            self.current_language = env_lang

        logger.info("I18n manager initialized",
                   extra={'default_language': default_language,
                         'current_language': self.current_language,
                         'available_languages': list(self.translations.keys())})

    def set_language(self, language: str) -> bool:
        """Set the current language."""
        if language not in self.translations:
            logger.warning(f"Language '{language}' not supported, using '{self.current_language}'")
            return False

        self.current_language = language
        logger.info(f"Language changed to: {language}")
        return True

    def get_category_translation(self, category: str) -> str:
        """Get translated category name."""
        lang_dict = self.translations.get(self.current_language, {})
        categories = lang_dict.get("categories", {})
        return categories.get(category, category)

    def get_response_translation(self, response_type: str = "default", category: str = None) -> str:
        """Get translated response text."""
        lang_dict = self.translations.get(self.current_language, {})
        responses = lang_dict.get("responses", {})

        # Try category-specific response first
        if category and category in responses:
            return responses[category]

        # Fall back to response type
        return responses.get(response_type, responses.get("default", "Thanks for your email"))

    def get_error_translation(self, error_type: str) -> str:
        """Get translated error message."""
        lang_dict = self.translations.get(self.current_language, {})
        errors = lang_dict.get("errors", {})
        return errors.get(error_type, error_type)

    def add_custom_translations(self, language: str, translations: Dict[str, Dict[str, str]]) -> None:
        """Add custom translations for a language."""
        if language not in self.translations:
            self.translations[language] = {"categories": {}, "responses": {}, "errors": {}}

        for category, trans_dict in translations.items():
            if category in self.translations[language]:
                self.translations[language][category].update(trans_dict)
            else:
                self.translations[language][category] = trans_dict

        logger.info(f"Added custom translations for '{language}'",
                   extra={'categories_added': len(translations.keys())})

    def get_available_languages(self) -> list[str]:
        """Get list of available languages."""
        return list(self.translations.keys())

    def load_translations_from_file(self, file_path: str) -> bool:
        """Load translations from JSON file."""
        try:
            with open(file_path, encoding='utf-8') as f:
                translations = json.load(f)

            for language, trans_dict in translations.items():
                self.add_custom_translations(language, trans_dict)

            logger.info(f"Loaded translations from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load translations from {file_path}: {e}")
            return False


# Global I18n manager instance
_i18n_manager: Optional[I18nManager] = None


def get_i18n_manager(language: str = None) -> I18nManager:
    """Get the global I18n manager instance."""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager(language or "en")
    elif language:
        _i18n_manager.set_language(language)
    return _i18n_manager


def translate_category(category: str, language: str = None) -> str:
    """Translate category to specified language."""
    manager = get_i18n_manager(language)
    return manager.get_category_translation(category)


def translate_response(response_type: str = "default", category: str = None, language: str = None) -> str:
    """Translate response to specified language."""
    manager = get_i18n_manager(language)
    return manager.get_response_translation(response_type, category)


def translate_error(error_type: str, language: str = None) -> str:
    """Translate error message to specified language."""
    manager = get_i18n_manager(language)
    return manager.get_error_translation(error_type)


def get_supported_languages() -> list[str]:
    """Get list of supported languages."""
    return list(TRANSLATIONS.keys())


def detect_language_from_content(content: str) -> str:
    """Simple language detection based on common patterns."""
    if not content:
        return "en"

    content_lower = content.lower()

    # Simple heuristics for language detection
    if any(word in content_lower for word in ['gracias', 'señor', 'correo', 'mensaje']):
        return "es"
    elif any(word in content_lower for word in ['merci', 'monsieur', 'madame', 'bonjour']):
        return "fr"
    elif any(word in content_lower for word in ['danke', 'herr', 'frau', 'guten']):
        return "de"
    elif any(char in content for char in '你好谢谢请问'):
        return "zh"
    elif any(char in content for char in 'ありがとうございますこんにちは'):
        return "ja"
    else:
        return "en"  # Default to English
