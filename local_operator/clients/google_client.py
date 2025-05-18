"""
Client for interacting with Google APIs (Gmail, Calendar, Drive).
"""

import base64
import json
import logging
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

GMAIL_API_BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me/"
GOOGLE_OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"


class GoogleAPIError(Exception):
    """Custom exception for Google API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class GmailMessagePartBody(BaseModel):
    """Represents the body of a Gmail message part."""

    data: Optional[str] = None
    size: Optional[int] = None


class GmailMessagePart(BaseModel):
    """Represents a part of a Gmail message (for multipart messages)."""

    partId: Optional[str] = None
    mimeType: Optional[str] = None
    filename: Optional[str] = None
    headers: Optional[List[Dict[str, str]]] = None
    body: Optional[GmailMessagePartBody] = None
    parts: Optional[List["GmailMessagePart"]] = None  # type: ignore


class GmailMessage(BaseModel):
    """Represents a Gmail message resource."""

    id: str
    threadId: str
    labelIds: Optional[List[str]] = None
    snippet: Optional[str] = None
    historyId: Optional[str] = None
    internalDate: Optional[str] = None
    payload: Optional[GmailMessagePart] = None
    raw: Optional[str] = None
    sizeEstimate: Optional[int] = None


class GmailListMessagesResponse(BaseModel):
    """Response for listing Gmail messages."""

    messages: Optional[List[GmailMessage]] = None
    nextPageToken: Optional[str] = None
    resultSizeEstimate: Optional[int] = None


class GmailDraft(BaseModel):
    """Represents a Gmail draft resource."""

    id: Optional[str] = None
    message: GmailMessage


class GoogleClient:
    """
    A client for interacting with various Google APIs.

    This client handles authentication and provides methods for common operations
    across Gmail, Google Calendar, and Google Drive.
    """

    def __init__(self, access_token: str):
        """
        Initializes the GoogleClient.

        Args:
            access_token: The OAuth 2.0 access token for Google APIs.
        """
        if not access_token:
            raise ValueError("Access token cannot be empty.")
        self.access_token = access_token
        self.headers = {"Authorization": f"Bearer {self.access_token}"}

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Makes an HTTP request to the Google API.

        Args:
            method: HTTP method (GET, POST, PATCH, etc.).
            url: The full URL for the API endpoint.
            params: Optional dictionary of query parameters.
            data: Optional dictionary for the request body (will be JSON-encoded).

        Returns:
            The JSON response from the API as a dictionary.

        Raises:
            GoogleAPIError: If the API request fails.
        """
        try:
            response = requests.request(method, url, headers=self.headers, params=params, json=data)
            response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            return response.json()
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_message = f"Google API request failed: {e.response.text}"
            try:
                error_details = e.response.json()
                if "error" in error_details and "message" in error_details["error"]:
                    err_msg = error_details["error"]["message"]
                    error_message = f"Google API Error: {err_msg} (Status: {status_code})"
            except json.JSONDecodeError:
                pass  # Stick with the text if JSON decoding fails
            logger.error(error_message, exc_info=True)
            raise GoogleAPIError(error_message, status_code=status_code) from e
        except requests.exceptions.RequestException as e:
            logger.error(f"Google API request failed: {e}", exc_info=True)
            raise GoogleAPIError(f"Google API request failed: {e}") from e

    # --- Gmail API Methods ---

    def list_gmail_messages(
        self,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> GmailListMessagesResponse:
        """
        Lists messages in the user's mailbox.

        Args:
            query: Query string to filter messages (e.g., "from:example@example.com", "is:unread").
            max_results: Maximum number of messages to return.
            page_token: Token for pagination.

        Returns:
            A GmailListMessagesResponse object.
        """
        params: Dict[str, Any] = {}
        if query:
            params["q"] = query
        if max_results:
            params["maxResults"] = max_results
        if page_token:
            params["pageToken"] = page_token

        url = f"{GMAIL_API_BASE_URL}messages"
        response_data = self._request("GET", url, params=params)
        return GmailListMessagesResponse(**response_data)

    def get_gmail_message(self, message_id: str, format: str = "full") -> GmailMessage:
        """
        Gets the specified message.

        Args:
            message_id: The ID of the message to retrieve.
            format: The format to return the message in (e.g., "full", "metadata", "raw").

        Returns:
            A GmailMessage object.
        """
        params = {"format": format}
        url = f"{GMAIL_API_BASE_URL}messages/{message_id}"
        response_data = self._request("GET", url, params=params)
        return GmailMessage(**response_data)

    def create_gmail_draft(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        sender: Optional[str] = None,
    ) -> GmailDraft:
        """
        Creates a new draft email.

        Args:
            to: List of recipient email addresses.
            subject: The subject of the email.
            body: The plain text body of the email.
            cc: Optional list of CC recipient email addresses.
            bcc: Optional list of BCC recipient email addresses.
            sender: Optional sender email address (if different from the authenticated user).

        Returns:
            A GmailDraft object representing the created draft.
        """
        mime_message_lines = []
        if sender:
            mime_message_lines.append(f"From: {sender}")
        mime_message_lines.append(f"To: {', '.join(to)}")
        if cc:
            mime_message_lines.append(f"Cc: {', '.join(cc)}")
        if bcc:
            mime_message_lines.append(
                f"Bcc: {', '.join(bcc)}"
            )  # Note: BCC usually handled by SMTP, not in headers seen by recipients
        mime_message_lines.append(f"Subject: {subject}")
        mime_message_lines.append('Content-Type: text/plain; charset="UTF-8"')
        mime_message_lines.append("")  # Blank line before body
        mime_message_lines.append(body)

        mime_message = "\r\n".join(mime_message_lines)
        raw_message = base64.urlsafe_b64encode(mime_message.encode("utf-8")).decode("utf-8")

        data = {"message": {"raw": raw_message}}
        url = f"{GMAIL_API_BASE_URL}drafts"
        response_data = self._request("POST", url, data=data)
        return GmailDraft(**response_data)

    def send_gmail_message(
        self,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        sender: Optional[str] = None,
    ) -> GmailMessage:
        """
        Sends an email message directly.

        Args:
            to: List of recipient email addresses.
            subject: The subject of the email.
            body: The plain text body of the email.
            cc: Optional list of CC recipient email addresses.
            bcc: Optional list of BCC recipient email addresses.
            sender: Optional sender email address (if different from the authenticated user).

        Returns:
            A GmailMessage object representing the sent message.
        """
        mime_message_lines = []
        if sender:
            mime_message_lines.append(f"From: {sender}")
        mime_message_lines.append(f"To: {', '.join(to)}")
        if cc:
            mime_message_lines.append(f"Cc: {', '.join(cc)}")
        if bcc:
            mime_message_lines.append(f"Bcc: {', '.join(bcc)}")
        mime_message_lines.append(f"Subject: {subject}")
        mime_message_lines.append('Content-Type: text/plain; charset="UTF-8"')
        mime_message_lines.append("")
        mime_message_lines.append(body)

        mime_message = "\r\n".join(mime_message_lines)
        raw_message = base64.urlsafe_b64encode(mime_message.encode("utf-8")).decode("utf-8")

        data = {"raw": raw_message}
        url = f"{GMAIL_API_BASE_URL}messages/send"
        response_data = self._request("POST", url, data=data)
        return GmailMessage(**response_data)

    def send_gmail_draft(self, draft_id: str) -> GmailMessage:
        """
        Sends an existing draft message.

        Args:
            draft_id: The ID of the draft to send.

        Returns:
            A GmailMessage object representing the sent message.
        """
        data = {"id": draft_id}
        url = f"{GMAIL_API_BASE_URL}drafts/send"
        response_data = self._request("POST", url, data=data)
        # The response for sending a draft is actually a Message resource
        return GmailMessage(**response_data)

    # --- Google Calendar API Methods (Placeholders) ---

    def list_calendar_events(self, calendar_id: str = "primary", **kwargs) -> Dict[str, Any]:
        """Placeholder for listing Google Calendar events."""
        logger.warning("list_calendar_events is a placeholder and not yet implemented.")
        # Example: url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
        # return self._request("GET", url, params=kwargs)
        return {"status": "not implemented", "calendar_id": calendar_id, "params": kwargs}

    def create_calendar_event(
        self, calendar_id: str = "primary", event_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Placeholder for creating a Google Calendar event."""
        logger.warning("create_calendar_event is a placeholder and not yet implemented.")
        # Example: url = f"https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
        # return self._request("POST", url, data=event_data or {})
        return {"status": "not implemented", "calendar_id": calendar_id, "event_data": event_data}

    # --- Google Drive API Methods (Placeholders) ---

    def list_drive_files(self, **kwargs) -> Dict[str, Any]:
        """Placeholder for listing Google Drive files."""
        logger.warning("list_drive_files is a placeholder and not yet implemented.")
        # Example: url = "https://www.googleapis.com/drive/v3/files"
        # return self._request("GET", url, params=kwargs)
        return {"status": "not implemented", "params": kwargs}

    def upload_drive_file(
        self, file_metadata: Dict[str, Any], file_content: bytes, mime_type: str
    ) -> Dict[str, Any]:
        """Placeholder for uploading a file to Google Drive."""
        logger.warning("upload_drive_file is a placeholder and not yet implemented.")
        # This is more complex, often involving multipart uploads.
        # Example (simplified):
        # headers = {**self.headers, "Content-Type": mime_type} # And potentially more for multipart
        # url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=media" # or multipart
        # response = requests.post(url, headers=headers, data=file_content) # or multipart data
        # response.raise_for_status()
        # return response.json()
        return {"status": "not implemented", "file_metadata": file_metadata, "mime_type": mime_type}


# Helper function for token refresh (will be used by SchedulerService)
def refresh_google_access_token(
    client_id: str, client_secret: str, refresh_token: str
) -> Dict[str, Any]:
    """
    Refreshes a Google OAuth 2.0 access token.

    Args:
        client_id: The Google Cloud project's client ID.
        client_secret: The Google Cloud project's client secret.
        refresh_token: The refresh token to use.

    Returns:
        A dictionary containing the new 'access_token', 'expires_in',
        and potentially 'id_token', 'scope', 'token_type'.

    Raises:
        GoogleAPIError: If the token refresh fails.
    """
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    try:
        response = requests.post(GOOGLE_OAUTH_TOKEN_URL, data=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_message = f"Google token refresh failed: {e.response.text}"
        try:
            error_details = e.response.json()
            if "error_description" in error_details:  # Google often uses error_description here
                err_desc = error_details["error_description"]
                error_message = f"Google Token Refresh Error: {err_desc} (Status: {status_code})"
            elif "error" in error_details and isinstance(error_details["error"], str):
                err = error_details["error"]
                error_message = f"Google Token Refresh Error: {err} (Status: {status_code})"

        except json.JSONDecodeError:
            pass
        logger.error(error_message, exc_info=True)
        raise GoogleAPIError(error_message, status_code=status_code) from e
    except requests.exceptions.RequestException as e:
        logger.error(f"Google token refresh request failed: {e}", exc_info=True)
        raise GoogleAPIError(f"Google token refresh request failed: {e}") from e


# Update GmailMessagePart to handle forward reference
GmailMessagePart.model_rebuild()
