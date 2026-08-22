import re

# Common Prompt Injection Patterns
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above) instructions",
    r"forget (all )?(previous|above) instructions",
    r"you are now an unrestricted ai",
    r"bypass system prompt",
    r"jailbreak",
    r"override rules"
]

# Prohibited Keywords / Content Safety Rules
PROHIBITED_TERMS = [
    "malware",
    "exploit code",
    "ddos attack",
    "hack system"
]

def validate_user_input(text: str):
    """
    Validates input text against safety guardrails.
    Returns:
        (is_safe: bool, reason: str)
    """
    cleaned_text = text.lower().strip()

    # 1. Check for empty or whitespace-only strings
    if not cleaned_text:
        return False, "Empty input provided."

    # 2. Check for Prompt Injection attempts
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned_text):
            return False, "Security Guardrail Triggered: System prompt override attempt detected."

    # 3. Check for Prohibited Terms
    for term in PROHIBITED_TERMS:
        if term in cleaned_text:
            return False, f"Content Safety Guardrail Triggered: Input contains prohibited topic ('{term}')."

    # Input passed all guardrails
    return True, "Safe"
