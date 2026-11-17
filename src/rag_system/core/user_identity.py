import secrets
import re
from uuid import UUID
USER_ID_PREFIX = "USR-"
USER_ID_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
USER_ID_PATTERN = re.compile(
    r"^USR-[23456789ABCDEFGHJKLMNPQRSTUVWXYZ]{16}$"
)
def generate_user_id() -> str:
    """Generate a readable, high-entropy user ID."""
    random_part = "".join(
        secrets.choice(USER_ID_ALPHABET)
        for _ in range(16)
    )
    return f"{USER_ID_PREFIX}{random_part}"
def normalize_user_id(user_id: str) -> str:
    """
    Validate and normalize a user ID.
    Supports:
    - Existing UUID user IDs.
    - Newly generated readable user IDs.
    """
    if not user_id or not user_id.strip():
        raise ValueError("User ID cannot be empty.")
    user_id = user_id.strip()
    # Preserve compatibility with existing UUID users.
    try:
        return str(UUID(user_id))
    except ValueError:
        pass
    # Validate new readable user IDs.
    normalized = user_id.upper()
    if USER_ID_PATTERN.fullmatch(normalized):
        return normalized
    raise ValueError(
        "Invalid user ID. Provide a valid UUID or readable user ID."
    )
def get_tenant_storage_key(user_id: str) -> str:
    """
    Generate a storage-safe tenant key.
    Existing UUID users retain their original tenant key format.
    """
    normalized = normalize_user_id(user_id)
    try:
        return f"tenant_{UUID(normalized).hex}"
    except ValueError:
        return f"tenant_{normalized.lower().replace('-', '_')}"
