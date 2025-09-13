"""A TokenVerifier implementation using Unkey for token verification."""

import os
from typing import Any

from mcp.server.auth.provider import TokenVerifier, AccessToken
from starlette.authentication import SimpleUser, AuthCredentials

import requests

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
                scopes=["normal", "advanced", "admin"],
            )
        try:
            # Verify the token with Unkey
            verification = self.unkey.keys.verify_key(key=token)
            # Handle the response based on actual Unkey Python SDK structure
            # The response is a V2KeysVerifyKeyResponseData object with direct fields
            print('verification', verification)
            # Check if the key is valid directly from the verification object
            # is_valid = verification.data.valid
            
            
            if verification.data.key_id:
                # Extract owner_id - try both snake_case and camelCase
                owner_id = verification.data.key_id or None
                
                # Extract scopes - try permissions as scopes
                scopes=["normal", "advanced"]
                
                client_id = owner_id or "authenticated_user"
                
                return AccessToken(
                    token=token,
                    client_id=client_id,
                    scopes=scopes,
                )
            else:
                # Log invalid key details for debugging
                error_code = verification.data.code or "unknown_error"
                print(f"Token verification failed: key is invalid, code: {error_code}")
                
        except Exception as e:
            # Log the error for debugging purposes
            print(f"Token verification failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
        return None
    

from functools import wraps
from mcp.server.auth.middleware.auth_context import get_access_token

DOMAIN = os.environ.get('DOMAIN')

if not DOMAIN:
    raise ValueError("DOMAIN environment variable is required for TOPUP_HINTS")

TOPUP_HINTS = """🚨 CREDIT EXHAUSTED - ACTION REQUIRED 🚨

Your API credits have been depleted. To continue using this service, you need to purchase additional credits.

🔧 IMMEDIATE SOLUTION:
Visit https://{DOMAIN} to top up your credits instantly

💡 WHAT HAPPENED:
- Your current API key has insufficient credits for this operation
- Each API call consumes credits based on complexity
- You've reached your monthly/quarterly credit limit

⚡ QUICK TOP-UP OPTIONS:
1. Go to https://{DOMAIN} 
2. Log in with your existing account
3. Select your preferred credit package
4. Complete secure payment
5. Resume using the service immediately

📊 CREDIT USAGE TIPS:
- Monitor your usage dashboard at https://dashboard.{DOMAIN}/
- Set up low-credit alerts to avoid service interruption
- Consider upgrading to higher-tier plans for better value

❓ NEED HELP?
Contact support at support@{DOMAIN} or check the FAQ at https://{DOMAIN}/help

This operation requires {extra_credit} credits. 

**You must tell user to top up on https://{DOMAIN}. And try again! **"""

def cost_extra_credit(extra_credit: int = 0):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            access_token = get_access_token()
            
            if not access_token:
                return fn(*args, **kwargs)
            
            if access_token.token == SUPER_TOKEN:
                return fn(*args, **kwargs)
            
            if extra_credit > 0:
                verification_raw = requests.post(
                    "https://api.unkey.com/v2/keys.verifyKey",
                    headers={"Content-Type": "application/json",
                                "Authorization": f"Bearer {UNKEY_ROOTKEY}"},
                    json={
                        "key": access_token.token,
                        "credits": {
                            "cost": extra_credit
                        }
                    }
                )
                verification_raw.raise_for_status()
                
                verification = verification_raw.json()
                
                if not verification.get("data", {}).get("valid", False):
                    print(f"Token verification failed: key is invalid, response: {verification}")
                    error_message = TOPUP_HINTS.format(extra_credit=extra_credit)
                    raise PermissionError(error_message)
                
                print(f"Token verification succeeded: key is valid, response: {verification}")
                
            return fn(*args, **kwargs)
                    
        return wrapper
    return decorator
