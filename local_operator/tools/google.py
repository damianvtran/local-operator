import base64
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from local_operator.clients.google_client import GoogleAPIError, GoogleClient
from local_operator.credentials import CredentialManager

# Constants for Google Credentials (matching scheduler_service.py)
GOOGLE_ACCESS_TOKEN_KEY = "GOOGLE_ACCESS_TOKEN"

# --- Gmail Tools ---


def list_gmail_messages_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the list_gmail_messages tool."""

    def list_gmail_messages(
        query: Optional[str] = None,
        max_results: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> str:
        """Lists messages in the user's Gmail mailbox.  This will provide a full message object for each message.  Be sure to query enough messages to be able to present a representative sample to the user based on their request.  Query the last 50 messages by default if asked for summaries unless otherwise specified.

        Args:
            query (Optional[str]): Query string to filter messages
            (e.g., "from:example@example.com", "is:unread").
            max_results (Optional[int]): Maximum number of messages to return.
            page_token (Optional[str]): Token for pagination.

        Returns:
            str: JSON string of GmailListMessagesResponse or error message.
        """  # noqa: E501
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            response = client.list_gmail_messages(
                query=query, max_results=max_results, page_token=page_token
            )
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return list_gmail_messages


def get_gmail_message_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the get_gmail_message tool."""

    def get_gmail_message(message_id: str, format: str = "full") -> str:
        """Gets the specified Gmail message.

        Args:
            message_id (str): The ID of the message to retrieve.
            format (str): The format to return the message in
            (e.g., "full", "metadata", "raw"). Defaults to "full".

        Returns:
            str: JSON string of GmailMessage or error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            response = client.get_gmail_message(message_id=message_id, format=format)
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return get_gmail_message


def create_gmail_draft_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the create_gmail_draft tool."""

    def create_gmail_draft(
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        sender: Optional[str] = None,
    ) -> str:
        """Creates a new draft email in Gmail.

        Args:
            to (List[str]): List of recipient email addresses.
            subject (str): The subject of the email.
            body (str): The plain text body of the email.
            cc (Optional[List[str]]): Optional list of CC recipient email addresses.
            bcc (Optional[List[str]]): Optional list of BCC recipient email addresses.
            sender (Optional[str]): Optional sender email address.

        Returns:
            str: JSON string of GmailDraft or error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            response = client.create_gmail_draft(
                to=to, subject=subject, body=body, cc=cc, bcc=bcc, sender=sender
            )
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return create_gmail_draft


def send_gmail_message_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the send_gmail_message tool."""

    def send_gmail_message(
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        sender: Optional[str] = None,
    ) -> str:
        """Sends an email message directly using Gmail.

        Args:
            to (List[str]): List of recipient email addresses.
            subject (str): The subject of the email.
            body (str): The plain text body of the email.
            cc (Optional[List[str]]): Optional list of CC recipient email addresses.
            bcc (Optional[List[str]]): Optional list of BCC recipient email addresses.
            sender (Optional[str]): Optional sender email address.

        Returns:
            str: JSON string of GmailMessage (representing the sent message) or error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            response = client.send_gmail_message(
                to=to, subject=subject, body=body, cc=cc, bcc=bcc, sender=sender
            )
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return send_gmail_message


def send_gmail_draft_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the send_gmail_draft tool."""

    def send_gmail_draft(draft_id: str) -> str:
        """Sends an existing Gmail draft message.

        Args:
            draft_id (str): The ID of the draft to send.

        Returns:
            str: JSON string of GmailMessage (representing the sent message) or error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            response = client.send_gmail_draft(draft_id=draft_id)
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return send_gmail_draft


def update_gmail_draft_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the update_gmail_draft tool."""

    def update_gmail_draft(
        draft_id: str,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        sender: Optional[str] = None,
    ) -> str:
        """Updates an existing Gmail draft.

        Args:
            draft_id (str): The ID of the draft to update.
            to (List[str]): List of recipient email addresses.
            subject (str): The subject of the email.
            body (str): The plain text body of the email.
            cc (Optional[List[str]]): Optional list of CC recipients.
            bcc (Optional[List[str]]): Optional list of BCC recipients.
            sender (Optional[str]): Optional sender email address.

        Returns:
            str: JSON string of the updated GmailDraft or error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            response = client.update_gmail_draft(
                draft_id=draft_id,
                to=to,
                subject=subject,
                body=body,
                cc=cc,
                bcc=bcc,
                sender=sender,
            )
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return update_gmail_draft


def delete_gmail_draft_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the delete_gmail_draft tool."""

    def delete_gmail_draft(draft_id: str) -> str:
        """Permanently deletes the specified Gmail draft.

        Args:
            draft_id (str): The ID of the draft to delete.

        Returns:
            str: JSON string confirming deletion or error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            client.delete_gmail_draft(draft_id=draft_id)
            return json.dumps({"status": "success", "message": f"Draft {draft_id} deleted."})
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return delete_gmail_draft


# --- Google Calendar Tools ---


def list_calendar_events_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the list_calendar_events tool."""

    def list_calendar_events(
        calendar_id: str = "primary",
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
        page_token: Optional[str] = None,
        single_events: Optional[bool] = None,
        order_by: Optional[str] = None,
    ) -> str:
        """Lists events on the specified Google Calendar.

        Args:
            calendar_id (str): Calendar identifier. Defaults to "primary".
            time_min (Optional[str]): Start of time range (ISO 8601).
            time_max (Optional[str]): End of time range (ISO 8601).
            query (Optional[str]): Free text search.
            max_results (Optional[int]): Maximum number of events to return.
            page_token (Optional[str]): Token for pagination.
            single_events (Optional[bool]): Whether to expand recurring events.
            order_by (Optional[str]): Order of events ("startTime" or "updated").

        Returns:
            str: JSON string of CalendarListEventsResponse or error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            response = client.list_calendar_events(
                calendar_id=calendar_id,
                time_min=time_min,
                time_max=time_max,
                query=query,
                max_results=max_results,
                page_token=page_token,
                single_events=single_events,
                order_by=order_by,
            )
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return list_calendar_events


def create_calendar_event_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the create_calendar_event tool."""

    def create_calendar_event(event_data: Dict[str, Any], calendar_id: str = "primary") -> str:
        """Creates an event on the specified Google Calendar. The event_data schema must match the full CalendarEvent model: {'summary': str, 'start': {'date': str (YYYY-MM-DD) or 'dateTime': str (ISO 8601), 'timeZone': Optional[str]}, 'end': {'date': str (YYYY-MM-DD) or 'dateTime': str (ISO 8601), 'timeZone': Optional[str]}, 'location': Optional[str], 'description': Optional[str], 'attendees': Optional[List[{'email': str, 'displayName': Optional[str], 'organizer': Optional[bool], 'self': Optional[bool], 'resource': Optional[bool], 'optional': Optional[bool], 'responseStatus': Optional[str], 'comment': Optional[str], 'additionalGuests': Optional[int]}]], 'reminders': Optional[{'useDefault': bool, 'overrides': Optional[List[{'method': str, 'minutes': int}]]}], 'status': Optional[str], 'htmlLink': Optional[str], 'created': Optional[str], 'updated': Optional[str], 'organizer': Optional[Dict[str, Any]], 'creator': Optional[Dict[str, Any]], 'recurringEventId': Optional[str]}. Must include 'summary', 'start', and 'end'.

        Args:
            event_data (Dict[str, Any]): Dictionary representing the event.
                                         Must include 'summary', 'start', and 'end'.
                                         Example: {"summary": "Meeting",
                                                  "start": {"dateTime": "2025-01-01T10:00", "timeZone": "America/New_York"},
                                                  "end": {"dateTime": "2025-01-01T11:00", "timeZone": "America/New_York"}}
            calendar_id (str): Calendar identifier. Defaults to "primary".

        Returns:
            str: JSON string of the created CalendarEvent or error message.
        """  # noqa: E501
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        # event_data is now a required argument, so no None check needed here.
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            response = client.create_calendar_event(calendar_id=calendar_id, event_data=event_data)
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except ValueError as ve:  # Catch validation errors from client
            return json.dumps({"error": f"Validation Error: {str(ve)}"})
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return create_calendar_event


def update_calendar_event_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the update_calendar_event tool."""

    def update_calendar_event(
        event_id: str,
        event_data: Dict[str, Any],
        calendar_id: str = "primary",
        send_updates: Optional[str] = None,
    ) -> str:
        """Updates an existing event on the specified Google Calendar. The event_data schema must match the  CalendarEvent model for the fields being updated: {'summary': str, 'start': {'date': str (YYYY-MM-DD) or 'dateTime': str (ISO 8601), 'timeZone': Optional[str]}, 'end': {'date': str (YYYY-MM-DD) or 'dateTime': str (ISO 8601), 'timeZone': Optional[str]}, 'location': Optional[str], 'description': Optional[str], 'attendees': Optional[List[{'email': str, 'displayName': Optional[str], 'organizer': Optional[bool], 'self': Optional[bool], 'resource': Optional[bool], 'optional': Optional[bool], 'responseStatus': Optional[str], 'comment': Optional[str], 'additionalGuests': Optional[int]}]], 'reminders': Optional[{'useDefault': bool, 'overrides': Optional[List[{'method': str, 'minutes': int}]]}], 'status': Optional[str], 'htmlLink': Optional[str], 'created': Optional[str], 'updated': Optional[str], 'organizer': Optional[Dict[str, Any]], 'creator': Optional[Dict[str, Any]], 'recurringEventId': Optional[str]}

        Args:
            event_id (str): Event identifier.
            event_data (Dict[str, Any]): Dictionary with fields to update.
                                         Example: {"summary": "Updated Meeting Title"}
            calendar_id (str): Calendar identifier. Defaults to "primary".
            send_updates (Optional[str]): Who gets notifications ("all", "externalOnly", "none").

        Returns:
            str: JSON string of the updated CalendarEvent or error message.
        """  # noqa: E501
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            # The client expects CalendarEvent or Dict, we pass Dict from tool input
            response = client.update_calendar_event(
                calendar_id=calendar_id,
                event_id=event_id,
                event_data=event_data,
                send_updates=send_updates,
            )
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except ValueError as ve:
            return json.dumps({"error": f"Validation Error: {str(ve)}"})
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return update_calendar_event


def delete_calendar_event_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the delete_calendar_event tool."""

    def delete_calendar_event(
        event_id: str, calendar_id: str = "primary", send_updates: Optional[str] = None
    ) -> str:
        """Deletes an event from the specified Google Calendar.

        Args:
            event_id (str): Event identifier.
            calendar_id (str): Calendar identifier. Defaults to "primary".
            send_updates (Optional[str]): Who gets notifications ("all", "externalOnly", "none").

        Returns:
            str: JSON string confirming deletion or error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            client.delete_calendar_event(
                calendar_id=calendar_id, event_id=event_id, send_updates=send_updates
            )
            return json.dumps({"status": "success", "message": f"Event {event_id} deleted."})
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return delete_calendar_event


# --- Google Drive Tools ---


def list_drive_files_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the list_drive_files tool."""

    def list_drive_files(
        query: Optional[str] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        fields: Optional[str] = None,
        corpora: Optional[str] = None,
        drive_id: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> str:
        """Lists or searches files in Google Drive.

        Args:
            query (Optional[str]): Query string (e.g., "name contains 'report'").
            page_size (Optional[int]): Number of files per page.
            page_token (Optional[str]): Token for pagination.
            fields (Optional[str]): Fields to include (e.g., "nextPageToken, files(id, name)").
            corpora (Optional[str]): Source: "user", "drive", "allDrives".
            drive_id (Optional[str]): ID of shared drive (if corpora="drive").
            order_by (Optional[str]): Sort keys (e.g., "createdTime desc").

        Returns:
            str: JSON string of DriveListFilesResponse or error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            response = client.list_drive_files(
                query=query,
                page_size=page_size,
                page_token=page_token,
                fields=fields,
                corpora=corpora,
                drive_id=drive_id,
                order_by=order_by,
            )
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except ValueError as ve:
            return json.dumps({"error": f"Validation Error: {str(ve)}"})
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return list_drive_files


def download_drive_file_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the download_drive_file tool.
    Note: This tool returns a base64 encoded string of the file content.
    """

    def download_drive_file(file_id: str, output_path: Optional[str] = None) -> str:
        """Downloads a file's content from Google Drive.

        Args:
            file_id (str): The ID of the file to download.
            output_path (Optional[str]): If provided, saves the file to this path.
                                         Otherwise, returns base64 encoded content.

        Returns:
            str: JSON string with "file_path" if saved, or "file_content_base64"
                 and "file_name" (if retrievable), or an error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            file_content_bytes = client.download_drive_file(file_id=file_id)

            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(file_content_bytes)
                return json.dumps({"status": "success", "file_path": output_path})
            else:
                # Try to get filename for context if not saving to path
                file_name = "downloaded_file"  # Default
                try:
                    meta = client.get_drive_file_metadata(file_id, fields="name")
                    if meta.name:
                        file_name = meta.name
                except Exception:  # pylint: disable=broad-except
                    pass  # Ignore error if metadata fetch fails, use default name

                encoded_content = base64.b64encode(file_content_bytes).decode("utf-8")
                return json.dumps(
                    {
                        "status": "success",
                        "file_name": file_name,
                        "file_content_base64": encoded_content,
                    }
                )
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return download_drive_file


def upload_drive_file_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the upload_drive_file tool."""

    def upload_drive_file(
        file_path: str,  # Path to local file to upload
        drive_file_name: Optional[str] = None,  # Name for the file in Drive
        mime_type: Optional[str] = None,
        parents: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """Uploads a local file to Google Drive.

        Args:
            file_path (str): Path to the local file to upload.
            drive_file_name (Optional[str]): Name for the file in Google Drive.
                                             Defaults to the local file's name.
            mime_type (Optional[str]): MIME type of the file. If None, attempts to guess.
            parents (Optional[List[str]]): Optional list of parent folder IDs in Drive.
            description (Optional[str]): Optional description for the file in Drive.

        Returns:
            str: JSON string of DriveFile (representing the uploaded file) or error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})

        if not os.path.exists(file_path):
            return json.dumps({"error": f"Local file not found: {file_path}"})

        try:
            with open(file_path, "rb") as f:
                file_content = f.read()

            actual_drive_file_name = drive_file_name or Path(file_path).name

            # Basic MIME type guessing if not provided
            actual_mime_type = mime_type
            if not actual_mime_type:
                import mimetypes

                guessed_type, _ = mimetypes.guess_type(file_path)
                actual_mime_type = guessed_type or "application/octet-stream"

            client = GoogleClient(access_token_secret.get_secret_value())
            response = client.upload_drive_file_multipart(
                file_name=actual_drive_file_name,
                file_content=file_content,
                mime_type=actual_mime_type,
                parents=parents,
                description=description,
            )
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return upload_drive_file


def update_drive_file_metadata_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the update_drive_file_metadata tool."""

    def update_drive_file_metadata(file_id: str, metadata_update: Dict[str, Any]) -> str:
        """Updates a Google Drive file's metadata.

        Args:
            file_id (str): The ID of the file to update.
            metadata_update (Dict[str, Any]): Metadata fields to update (e.g., {"name": "new.txt"}).

        Returns:
            str: JSON string of the updated DriveFile or error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})
        try:
            client = GoogleClient(access_token_secret.get_secret_value())
            response = client.update_drive_file_metadata(
                file_id=file_id, metadata_update=metadata_update
            )
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return update_drive_file_metadata


def update_drive_file_content_tool(credential_manager: CredentialManager) -> Callable[..., str]:
    """Factory to create the update_drive_file_content tool."""

    def update_drive_file_content(
        file_id: str, local_file_path: str, new_mime_type: Optional[str] = None
    ) -> str:
        """Updates a Google Drive file's content from a local file.

        Args:
            file_id (str): The ID of the Drive file to update.
            local_file_path (str): Path to the local file with the new content.
            new_mime_type (Optional[str]): New MIME type. Guesses if None.

        Returns:
            str: JSON string of the updated DriveFile or error message.
        """
        access_token_secret = credential_manager.get_credential(GOOGLE_ACCESS_TOKEN_KEY)
        if not access_token_secret or not access_token_secret.get_secret_value():
            return json.dumps({"error": "Google access token not found or empty."})

        if not os.path.exists(local_file_path):
            return json.dumps({"error": f"Local file not found: {local_file_path}"})

        try:
            with open(local_file_path, "rb") as f:
                new_content_bytes = f.read()

            actual_new_mime_type = new_mime_type
            if not actual_new_mime_type:
                import mimetypes

                guessed_type, _ = mimetypes.guess_type(local_file_path)
                actual_new_mime_type = guessed_type or "application/octet-stream"

            client = GoogleClient(access_token_secret.get_secret_value())
            response = client.update_drive_file_content(
                file_id=file_id, new_content=new_content_bytes, new_mime_type=actual_new_mime_type
            )
            return response.model_dump_json()
        except GoogleAPIError as e:
            return json.dumps(
                {"error": f"Google API Error: {str(e)}", "status_code": e.status_code}
            )
        except Exception as e:
            return json.dumps({"error": f"An unexpected error occurred: {str(e)}"})

    return update_drive_file_content
