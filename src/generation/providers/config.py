"""
Provider Configuration for LLM Rate Limits and Deployment Routes.
Loads limits from environment variables with sane fallback defaults.
"""

import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class ProviderConfig:
    """
    Configuration for a specific provider and route combination.

    Attributes:
        provider: Provider name ('anthropic', 'openai', 'gemini')
        route: Deployment route ('direct', 'bedrock', 'azure', 'vertex')
        rpm: Requests per minute limit
        input_tpm: Input tokens per minute limit
        output_tpm: Output tokens per minute limit
        model: Default model for this route
    """

    provider: str
    route: str
    rpm: int
    input_tpm: int
    output_tpm: int
    model: str


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable with default."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _get_env_str(key: str, default: str) -> str:
    """Get string from environment variable with default."""
    return os.environ.get(key, default)


def get_provider_config(
    provider: Literal["anthropic", "openai", "gemini"],
    route: Literal["direct", "bedrock", "azure", "vertex"],
) -> ProviderConfig:
    """
    Get configuration for a specific provider and route.

    Environment variables (upper case, underscore-separated):
    - {PROVIDER}_{ROUTE}_RPM: Requests per minute
    - {PROVIDER}_{ROUTE}_INPUT_TPM: Input tokens per minute
    - {PROVIDER}_{ROUTE}_OUTPUT_TPM: Output tokens per minute
    - {PROVIDER}_{ROUTE}_MODEL: Default model name

    Examples:
    - ANTHROPIC_DIRECT_RPM=50
    - ANTHROPIC_BEDROCK_INPUT_TPM=40000
    - OPENAI_AZURE_MODEL=gpt-4o

    Fallback defaults are documented per provider/route below.
    """
    prefix = f"{provider.upper()}_{route.upper()}"

    # Anthropic configurations
    if provider == "anthropic":
        if route == "direct":
            # Anthropic Claude defaults (Tier 1): 50 RPM, 40K input TPM, 40K output TPM
            return ProviderConfig(
                provider=provider,
                route=route,
                rpm=_get_env_int(f"{prefix}_RPM", 50),
                input_tpm=_get_env_int(f"{prefix}_INPUT_TPM", 40000),
                output_tpm=_get_env_int(f"{prefix}_OUTPUT_TPM", 40000),
                model=_get_env_str(f"{prefix}_MODEL", "claude-sonnet-5"),
            )
        elif route == "bedrock":
            # AWS Bedrock Anthropic defaults (Claude 3.5 Sonnet)
            # Note: Claude 3.7+ output tokens burn Bedrock TPM quota at 5x rate
            return ProviderConfig(
                provider=provider,
                route=route,
                rpm=_get_env_int(f"{prefix}_RPM", 50),
                input_tpm=_get_env_int(f"{prefix}_INPUT_TPM", 40000),
                output_tpm=_get_env_int(
                    f"{prefix}_OUTPUT_TPM", 8000
                ),  # Conservative due to 5x multiplier
                model=_get_env_str(
                    f"{prefix}_MODEL",
                    "anthropic.claude-3-5-sonnet-20241022-v2:0",
                ),
            )

    # OpenAI configurations
    elif provider == "openai":
        if route == "direct":
            # OpenAI GPT-4o defaults (Tier 3): 10K RPM, 200K input TPM, 200K output TPM
            return ProviderConfig(
                provider=provider,
                route=route,
                rpm=_get_env_int(f"{prefix}_RPM", 10000),
                input_tpm=_get_env_int(f"{prefix}_INPUT_TPM", 200000),
                output_tpm=_get_env_int(f"{prefix}_OUTPUT_TPM", 200000),
                model=_get_env_str(f"{prefix}_MODEL", "gpt-4o"),
            )
        elif route == "azure":
            # Azure OpenAI defaults (varies by deployment)
            return ProviderConfig(
                provider=provider,
                route=route,
                rpm=_get_env_int(f"{prefix}_RPM", 300),
                input_tpm=_get_env_int(f"{prefix}_INPUT_TPM", 120000),
                output_tpm=_get_env_int(f"{prefix}_OUTPUT_TPM", 120000),
                model=_get_env_str(f"{prefix}_MODEL", "gpt-4o"),
            )

    # Gemini configurations
    elif provider == "gemini":
        if route == "direct":
            # Google Gemini direct API defaults
            return ProviderConfig(
                provider=provider,
                route=route,
                rpm=_get_env_int(f"{prefix}_RPM", 15),
                input_tpm=_get_env_int(f"{prefix}_INPUT_TPM", 15000),
                output_tpm=_get_env_int(f"{prefix}_OUTPUT_TPM", 15000),
                model=_get_env_str(f"{prefix}_MODEL", "gemini-2.5-flash"),
            )
        elif route == "vertex":
            # GCP Vertex AI defaults (varies by quota)
            return ProviderConfig(
                provider=provider,
                route=route,
                rpm=_get_env_int(f"{prefix}_RPM", 60),
                input_tpm=_get_env_int(f"{prefix}_INPUT_TPM", 120000),
                output_tpm=_get_env_int(f"{prefix}_OUTPUT_TPM", 120000),
                model=_get_env_str(f"{prefix}_MODEL", "gemini-2.5-flash"),
            )

    raise ValueError(f"Unsupported provider/route combination: {provider}/{route}")
