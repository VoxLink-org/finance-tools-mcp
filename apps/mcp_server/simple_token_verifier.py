"""A TokenVerifier implementation using Unkey for token verification."""

import os
from typing import Any

from mcp.server.auth.provider import TokenVerifier, AccessToken
from starlette.authentication import SimpleUser, AuthCredentials

from unkey.py import Unkey

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

UNKEY_ROOTKEY = os.environ.get('UNKEY_ROOTKEY', "")
SUPER_TOKEN = os.environ.get('SUPER_TOKEN', "")

class SimpleTokenVerifier(TokenVerifier):
    """A TokenVerifier that validates tokens using Unkey API."""

    def __init__(self) -> None:
        """
        Initialize the verifier with Unkey client.
        """
        if not UNKEY_ROOTKEY:
            raise ValueError("UNKEY_ROOTKEY environment variable is required for Unkey token verification")
        
        self.unkey = Unkey(root_key=UNKEY_ROOTKEY)

    async def verify_token(self, token: str) -> AccessToken | None:
        """
        Verify the provided token using Unkey API.
        
        Args:
            token: The token string from the Authorization header.
            
        Returns:
            AccessToken if the token is valid, otherwise None.
        """
        if token == SUPER_TOKEN:
            return AccessToken(
                token=token,
                client_id="super_user",
                scopes=["read", "write"],
            )
        try:
            # Verify the token with Unkey
            verification = self.unkey.keys.verify_key(key=token)
            # Handle the response based on actual Unkey Python SDK structure
            # The response is a V2KeysVerifyKeyResponseData object with direct fields
            
            # Check if the key is valid directly from the verification object
            is_valid = verification.data.valid
            
            # Use the verification object directly for other attributes
            result = verification
            
            if is_valid:
                # Extract owner_id - try both snake_case and camelCase
                owner_id = result.data.key_id or None
                
                # Extract scopes - try permissions as scopes
                scopes = ["read", "write"]
                permissions = result.data.permissions
                if permissions and isinstance(permissions, list):
                    scopes = permissions
                
                client_id = owner_id or "authenticated_user"
                
                return AccessToken(
                    token=token,
                    client_id=client_id,
                    scopes=scopes,
                )
            else:
                # Log invalid key details for debugging
                error_code = getattr(result, 'code', 'UNKNOWN')
                print(f"Token verification failed: key is invalid, code: {error_code}")
                
        except Exception as e:
            # Log the error for debugging purposes
            print(f"Token verification failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return None