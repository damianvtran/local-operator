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
CALENDAR_API_BASE_URL = "https://www.googleapis.com/calendar/v3/"
DRIVE_API_BASE_URL = "https://www.googleapis.com/drive/v3/"
DRIVE_API_UPLOAD_BASE_URL = "https://www.googleapis.com/upload/drive/v3/"
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


# --- Google Calendar API Models ---


class CalendarEventDateTime(BaseModel):
    """Represents date or dateTime for calendar events."""

    date: Optional[str] = None  # YYYY-MM-DD
    dateTime: Optional[str] = None  # ISO 8601 format
    timeZone: Optional[str] = None


class CalendarEventAttendee(BaseModel):
    """Represents an attendee of a calendar event."""

    email: str
    displayName: Optional[str] = None
    organizer: Optional[bool] = None
    self: Optional[bool] = None
    resource: Optional[bool] = None
    optional: Optional[bool] = None
    responseStatus: Optional[str] = None  # "needsAction", "declined", "tentative", "accepted"
    comment: Optional[str] = None
    additionalGuests: Optional[int] = None


class CalendarEventReminderOverride(BaseModel):
    """Represents a reminder override for a calendar event."""

    method: str  # "email", "popup"
    minutes: int


class CalendarEventReminders(BaseModel):
    """Represents reminders for a calendar event."""

    useDefault: bool
    overrides: Optional[List[CalendarEventReminderOverride]] = None


class CalendarEvent(BaseModel):
    """Represents a Google Calendar event resource."""

    id: Optional[str] = None
    summary: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    start: CalendarEventDateTime
    end: CalendarEventDateTime
    attendees: Optional[List[CalendarEventAttendee]] = None
    reminders: Optional[CalendarEventReminders] = None
    status: Optional[str] = None  # "confirmed", "tentative", "cancelled"
    htmlLink: Optional[str] = None
    created: Optional[str] = None  # ISO 8601
    updated: Optional[str] = None  # ISO 8601
    organizer: Optional[Dict[str, Any]] = (
        None  # Typically {"email": "...", "displayName": "...", "self": True/False}
    )
    creator: Optional[Dict[str, Any]] = None
    recurringEventId: Optional[str] = None


class CalendarListEventsResponse(BaseModel):
    """Response for listing Google Calendar events."""

    kind: str = "calendar#events"
    etag: Optional[str] = None
    summary: Optional[str] = None
    description: Optional[str] = None
    updated: Optional[str] = None  # ISO 8601
    timeZone: Optional[str] = None
    accessRole: Optional[str] = None
    defaultReminders: Optional[List[CalendarEventReminderOverride]] = None
    nextPageToken: Optional[str] = None
    nextSyncToken: Optional[str] = None
    items: Optional[List[CalendarEvent]] = None


# --- Google Drive API Models ---


class DriveUser(BaseModel):
    """Represents a Google Drive user."""

    displayName: Optional[str] = None
    kind: Optional[str] = "drive#user"
    me: Optional[bool] = None
    permissionId: Optional[str] = None
    emailAddress: Optional[str] = None
    photoLink: Optional[str] = None


class DriveFileCapabilities(BaseModel):
    """Represents capabilities of a Drive file for the current user."""

    canAddChildren: Optional[bool] = None
    canChangeCopyRequiresWriterPermission: Optional[bool] = None
    canChangeViewersCanCopyContent: Optional[bool] = None
    canComment: Optional[bool] = None
    canCopy: Optional[bool] = None
    canDelete: Optional[bool] = None
    canDownload: Optional[bool] = None
    canEdit: Optional[bool] = None
    canListChildren: Optional[bool] = None
    canModifyContent: Optional[bool] = None
    canMoveChildrenOutOfDrive: Optional[bool] = None
    canMoveChildrenWithinDrive: Optional[bool] = None
    canMoveItemOutOfDrive: Optional[bool] = None
    canMoveItemWithinDrive: Optional[bool] = None
    canReadRevisions: Optional[bool] = None
    canRemoveChildren: Optional[bool] = None
    canRename: Optional[bool] = None
    canShare: Optional[bool] = None
    canTrash: Optional[bool] = None
    canUntrash: Optional[bool] = None


class DriveFile(BaseModel):
    """Represents a Google Drive file or folder resource."""

    kind: str = "drive#file"
    id: Optional[str] = None
    name: Optional[str] = None
    mimeType: Optional[str] = None
    description: Optional[str] = None
    starred: Optional[bool] = None
    trashed: Optional[bool] = None
    parents: Optional[List[str]] = None
    properties: Optional[Dict[str, str]] = None
    appProperties: Optional[Dict[str, str]] = None
    spaces: Optional[List[str]] = None  # e.g., ["drive"]
    version: Optional[int] = None
    webContentLink: Optional[str] = (
        None  # A link for downloading the content of a publicly-shared file.
    )
    webViewLink: Optional[str] = (
        None  # A link for opening the file in a relevant Google editor or viewer in a browser.
    )
    iconLink: Optional[str] = None
    hasThumbnail: Optional[bool] = None
    thumbnailLink: Optional[str] = None
    createdTime: Optional[str] = None  # ISO 8601
    modifiedTime: Optional[str] = None  # ISO 8601
    modifiedByMeTime: Optional[str] = None  # ISO 8601
    viewedByMeTime: Optional[str] = None  # ISO 8601
    sharedWithMeTime: Optional[str] = None  # ISO 8601
    owners: Optional[List[DriveUser]] = None
    lastModifyingUser: Optional[DriveUser] = None
    shared: Optional[bool] = None
    ownedByMe: Optional[bool] = None
    capabilities: Optional[DriveFileCapabilities] = None
    viewersCanCopyContent: Optional[bool] = None
    copyRequiresWriterPermission: Optional[bool] = None
    writersCanShare: Optional[bool] = None
    permissions: Optional[List[Dict[str, Any]]] = None  # Simplified for now
    folderColorRgb: Optional[str] = None
    originalFilename: Optional[str] = None
    fullFileExtension: Optional[str] = None
    fileExtension: Optional[str] = None
    md5Checksum: Optional[str] = None
    size: Optional[str] = None  # String representation of file size in bytes
    quotaBytesUsed: Optional[str] = None
    headRevisionId: Optional[str] = None
    # Add other fields as needed


class DriveListFilesResponse(BaseModel):
    """Response for listing Google Drive files."""

    kind: str = "drive#files"
    nextPageToken: Optional[str] = None
    incompleteSearch: Optional[bool] = None
    files: Optional[List[DriveFile]] = None


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
        data: Optional[Any] = None,  # Can be Dict for JSON, or bytes for uploads
        is_upload: bool = False,
        upload_content_type: Optional[str] = None,
        is_download: bool = False,
    ) -> Any:  # Return type can be Dict or bytes
        """
        Makes an HTTP request to the Google API.

        Args:
            method: HTTP method (GET, POST, PATCH, etc.).
            url: The full URL for the API endpoint.
            params: Optional dictionary of query parameters.
            data: Optional data for the request body.
                  If is_upload is False, this should be a dict (will be JSON-encoded).
                  If is_upload is True, this should be bytes.
            is_upload: Boolean indicating if this is a file upload.
            upload_content_type: Content-Type header for uploads.
            is_download: Boolean indicating if the response should be raw bytes.

        Returns:
            The JSON response from the API as a dictionary, or bytes if is_download is True.

        Raises:
            GoogleAPIError: If the API request fails.
        """
        request_headers = self.headers.copy()
        request_kwargs: Dict[str, Any] = {"headers": request_headers, "params": params}

        if is_upload:
            if upload_content_type:
                request_headers["Content-Type"] = upload_content_type
            request_kwargs["data"] = data  # Pass bytes directly for data
        elif data is not None:  # For non-uploads, assume JSON
            request_kwargs["json"] = data

        try:
            response = requests.request(method, url, **request_kwargs)
            response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)

            if is_download:
                return response.content  # Return raw bytes for downloads

            # For Drive API, sometimes a 204 No Content is returned on successful PATCH/DELETE
            if response.status_code == 204:
                return {}  # Return an empty dict to signify success with no content

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
        # Initial call to get message IDs and page token
        list_response_data = self._request("GET", url, params=params)

        detailed_messages: List[GmailMessage] = []
        if list_response_data.get("messages"):
            for message_stub in list_response_data["messages"]:
                if "id" in message_stub:
                    try:
                        # Fetch more details for each message.
                        # Using 'metadata' to get snippet, headers, labels, etc.,
                        # without fetching the full payload for efficiency in a list.
                        detailed_message = self.get_gmail_message(
                            message_id=message_stub["id"], format="metadata"
                        )
                        detailed_messages.append(detailed_message)
                    except GoogleAPIError as e:
                        # Log error and potentially skip this message or add a placeholder
                        logger.error(f"Failed to get details for message {message_stub['id']}: {e}")
                else:
                    # Handle cases where a message stub might not have an ID (should be rare)
                    logger.warning(f"Message stub without ID found: {message_stub}")

        return GmailListMessagesResponse(
            messages=detailed_messages if detailed_messages else None,
            nextPageToken=list_response_data.get("nextPageToken"),
            resultSizeEstimate=list_response_data.get("resultSizeEstimate"),
        )

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

    def update_gmail_draft(
        self,
        draft_id: str,
        to: List[str],
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        sender: Optional[str] = None,
    ) -> GmailDraft:
        """
        Updates an existing draft email.
        The message contained in the draft is replaced with the new content.

        Args:
            draft_id: The ID of the draft to update.
            to: List of recipient email addresses for the updated message.
            subject: The subject of the updated email.
            body: The plain text body of the updated email.
            cc: Optional list of CC recipient email addresses.
            bcc: Optional list of BCC recipient email addresses.
            sender: Optional sender email address.

        Returns:
            A GmailDraft object representing the updated draft.
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
        mime_message_lines.append("")  # Blank line before body
        mime_message_lines.append(body)

        mime_message = "\r\n".join(mime_message_lines)
        raw_message = base64.urlsafe_b64encode(mime_message.encode("utf-8")).decode("utf-8")

        # Construct the message part of the draft payload
        # The API expects the draft ID in the URL, and the message in the body.
        # The GmailDraft model includes an 'id' field, but for an update,
        # the API expects the message content directly under a 'message' key in the payload.
        data = {"message": {"raw": raw_message}}

        url = f"{GMAIL_API_BASE_URL}drafts/{draft_id}"
        response_data = self._request("PUT", url, data=data)
        # The response includes the draft ID and the new message.
        return GmailDraft(**response_data)

    def delete_gmail_draft(self, draft_id: str) -> None:
        """
        Permanently deletes the specified draft.

        Args:
            draft_id: The ID of the draft to delete.

        Returns:
            None. Raises GoogleAPIError on failure.
        """
        url = f"{GMAIL_API_BASE_URL}drafts/{draft_id}"
        self._request("DELETE", url)
        # A successful DELETE request to Gmail API for drafts returns a 204 No Content.
        # The _request method handles this by returning an empty dict, which we ignore.
        return

    # --- Google Calendar API Methods ---

    def list_calendar_events(
        self,
        calendar_id: str = "primary",
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        query: Optional[str] = None,
        max_results: Optional[int] = None,
        page_token: Optional[str] = None,
        single_events: Optional[bool] = None,  # Expands recurring events
        order_by: Optional[str] = None,  # "startTime" or "updated"
    ) -> CalendarListEventsResponse:
        """
        Lists events on the specified calendar.

        Args:
            calendar_id: Calendar identifier. Use "primary" for the primary calendar.
            time_min: ISO 8601 datetime string for the start of the time range.
            time_max: ISO 8601 datetime string for the end of the time range.
            query: Free text search.
            max_results: Maximum number of events to return.
            page_token: Token for pagination.
            single_events: Whether to expand recurring events into instances.
            order_by: The order of the events returned in the result.
                      Allowed values are "startTime" or "updated".

        Returns:
            A CalendarListEventsResponse object.
        """
        params: Dict[str, Any] = {}
        if time_min:
            params["timeMin"] = time_min
        if time_max:
            params["timeMax"] = time_max
        if query:
            params["q"] = query
        if max_results:
            params["maxResults"] = max_results
        if page_token:
            params["pageToken"] = page_token
        if single_events is not None:
            params["singleEvents"] = single_events
        if order_by:
            params["orderBy"] = order_by

        url = f"{CALENDAR_API_BASE_URL}calendars/{calendar_id}/events"
        response_data = self._request("GET", url, params=params)
        return CalendarListEventsResponse(**response_data)

    def create_calendar_event(
        self, calendar_id: str, event_data: CalendarEvent | Dict[str, Any]
    ) -> CalendarEvent:
        """
        Creates an event on the specified calendar.

        Args:
            calendar_id: Calendar identifier. Use "primary" for the primary calendar.
            event_data: An CalendarEvent object or a dictionary representing the event.
                        Must include 'start', 'end', and 'summary'.

        Returns:
            A CalendarEvent object representing the created event.
        """
        if isinstance(event_data, CalendarEvent):
            payload = event_data.model_dump(exclude_none=True)
        else:
            payload = event_data

        if not payload.get("summary") or not payload.get("start") or not payload.get("end"):
            raise ValueError("Event data must include 'summary', 'start', and 'end' times.")

        url = f"{CALENDAR_API_BASE_URL}calendars/{calendar_id}/events"
        response_data = self._request("POST", url, data=payload)
        return CalendarEvent(**response_data)

    def update_calendar_event(
        self,
        calendar_id: str,
        event_id: str,
        event_data: CalendarEvent | Dict[str, Any],
        send_updates: Optional[str] = None,  # "all", "externalOnly", "none"
    ) -> CalendarEvent:
        """
        Updates an existing event on the specified calendar.

        Args:
            calendar_id: Calendar identifier.
            event_id: Event identifier.
            event_data: An CalendarEvent object or a dictionary with fields to update.
                        Must include 'start', 'end', and 'summary' if they are being changed.
            send_updates: Guests who should receive notifications about the update.
                          Acceptable values are "all", "externalOnly", "none".
                          Default is "none" if not specified by the API.

        Returns:
            A CalendarEvent object representing the updated event.
        """
        if isinstance(event_data, CalendarEvent):
            payload = event_data.model_dump(exclude_none=True, exclude_unset=True)
        else:
            payload = event_data

        # The API requires start and end times if they are part of the update.
        # However, a PATCH request can update only specific fields.
        # We'll let the API validate the presence of required fields for an update.

        url = f"{CALENDAR_API_BASE_URL}calendars/{calendar_id}/events/{event_id}"
        params: Dict[str, Any] = {}
        if send_updates:
            params["sendUpdates"] = send_updates

        response_data = self._request("PATCH", url, params=params, data=payload)
        return CalendarEvent(**response_data)

    def delete_calendar_event(
        self,
        calendar_id: str,
        event_id: str,
        send_updates: Optional[str] = None,  # "all", "externalOnly", "none"
    ) -> None:
        """
        Deletes an event from the specified calendar.

        Args:
            calendar_id: Calendar identifier.
            event_id: Event identifier.
            send_updates: Guests who should receive notifications about the deletion.
                          Acceptable values are "all", "externalOnly", "none".
                          Default is "none" if not specified by the API.

        Returns:
            None. Raises GoogleAPIError on failure.
        """
        url = f"{CALENDAR_API_BASE_URL}calendars/{calendar_id}/events/{event_id}"
        params: Dict[str, Any] = {}
        if send_updates:
            params["sendUpdates"] = send_updates

        self._request("DELETE", url, params=params)
        # A successful DELETE request to Google Calendar API returns a 204 No Content status.
        # The _request method handles this by returning an empty dict, which we ignore here.
        return

    # --- Google Drive API Methods ---

    def list_drive_files(
        self,
        query: Optional[str] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
        fields: Optional[str] = None,  # e.g., "nextPageToken, files(id, name, mimeType)"
        corpora: Optional[str] = None,  # "user", "drive", "allDrives"
        drive_id: Optional[str] = None,  # For searching within a specific shared drive
        order_by: Optional[str] = None,  # e.g. "createdTime desc, name"
    ) -> DriveListFilesResponse:
        """
        Lists or searches files in Google Drive.

        Args:
            query: Query string to filter files (e.g., "name contains 'report'",
            "mimeType='application/pdf'").
            page_size: Number of files to return per page.
            page_token: Token for pagination.
            fields: Selector specifying which fields to include in a partial response.
            corpora: The source of files to query. "user" for user's personal Drive,
                     "drive" for a specific shared drive (requires drive_id),
                     "allDrives" for all shared drives.
            drive_id: ID of the shared drive to search. Required if corpora is "drive".
            order_by: A comma-separated list of sort keys.

        Returns:
            A DriveListFilesResponse object.
        """
        params: Dict[str, Any] = {}
        if query:
            params["q"] = query
        if page_size:
            params["pageSize"] = page_size
        if page_token:
            params["pageToken"] = page_token
        if fields:
            params["fields"] = fields
        if corpora:
            params["corpora"] = corpora
            if corpora == "drive" and not drive_id:
                raise ValueError("drive_id is required when corpora is 'drive'.")
        if drive_id:
            params["driveId"] = drive_id
            params["includeItemsFromAllDrives"] = True  # Often needed with driveId
            params["supportsAllDrives"] = True  # Often needed with driveId
        if order_by:
            params["orderBy"] = order_by

        url = f"{DRIVE_API_BASE_URL}files"
        response_data = self._request("GET", url, params=params)
        return DriveListFilesResponse(**response_data)

    def download_drive_file(self, file_id: str) -> bytes:
        """
        Downloads a file's content from Google Drive.

        Args:
            file_id: The ID of the file to download.

        Returns:
            The file content as bytes.
        """
        url = f"{DRIVE_API_BASE_URL}files/{file_id}"
        params = {"alt": "media"}
        # The _request method will return bytes when is_download=True
        file_content: bytes = self._request("GET", url, params=params, is_download=True)
        return file_content

    def upload_drive_file_multipart(
        self,
        file_name: str,
        file_content: bytes,
        mime_type: str,
        parents: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> DriveFile:
        """
        Uploads a file to Google Drive using multipart upload, allowing metadata and content.

        Args:
            file_name: The name of the file in Drive.
            file_content: The content of the file as bytes.
            mime_type: The MIME type of the file.
            parents: Optional list of parent folder IDs.
            description: Optional description for the file.

        Returns:
            A DriveFile object representing the uploaded file.
        """
        metadata: Dict[str, Any] = {"name": file_name, "mimeType": mime_type}
        if parents:
            metadata["parents"] = parents
        if description:
            metadata["description"] = description

        upload_url = f"{DRIVE_API_UPLOAD_BASE_URL}files?uploadType=media"

        # We need to use requests.post directly here as _request is not set up for this.
        temp_headers = self.headers.copy()
        temp_headers["Content-Type"] = mime_type

        try:
            response_upload = requests.post(upload_url, headers=temp_headers, data=file_content)
            response_upload.raise_for_status()
            uploaded_file_data = response_upload.json()
            file_id = uploaded_file_data.get("id")

            if not file_id:
                raise GoogleAPIError("Failed to get file ID after simple media upload.")

            # Step 2: Update metadata for the uploaded file
            return self.update_drive_file_metadata(file_id, metadata_update=metadata)

        except requests.exceptions.HTTPError as e:
            raise GoogleAPIError(
                f"Drive multipart upload (simulated) failed: {e.response.text}",
                e.response.status_code,
            ) from e
        except requests.exceptions.RequestException as e:
            raise GoogleAPIError(f"Drive multipart upload (simulated) request failed: {e}") from e

    def update_drive_file_metadata(
        self, file_id: str, metadata_update: Dict[str, Any]
    ) -> DriveFile:
        """
        Updates a file's metadata.

        Args:
            file_id: The ID of the file to update.
            metadata_update: A dictionary containing the metadata fields to update
                             (e.g., {"name": "new_name.txt", "description": "new desc"}).

        Returns:
            A DriveFile object representing the updated file.
        """
        url = f"{DRIVE_API_BASE_URL}files/{file_id}"
        # Ensure common query params for updates are included if necessary
        params = {"supportsAllDrives": True}  # Good practice for shared drives

        response_data = self._request("PATCH", url, params=params, data=metadata_update)
        return DriveFile(**response_data)

    def update_drive_file_content(
        self, file_id: str, new_content: bytes, new_mime_type: Optional[str] = None
    ) -> DriveFile:
        """
        Updates a file's content.

        Args:
            file_id: The ID of the file to update.
            new_content: The new content for the file as bytes.
            new_mime_type: Optional. The new MIME type of the file. If not provided,
                           Drive attempts to auto-detect. It's best to provide it.

        Returns:
            A DriveFile object representing the updated file.
        """
        url = f"{DRIVE_API_UPLOAD_BASE_URL}files/{file_id}?uploadType=media"

        # Determine content type for the upload
        content_type = new_mime_type
        if not content_type:
            logger.warning(
                "new_mime_type not provided for content update, upload might be "
                "misinterpereted. Defaulting to application/octet-stream for safety."
            )
            content_type = "application/octet-stream"

        response_data = self._request(
            "PATCH", url, data=new_content, is_upload=True, upload_content_type=content_type
        )
        return DriveFile(**response_data)

    def get_drive_file_metadata(self, file_id: str, fields: Optional[str] = None) -> DriveFile:
        """
        Gets a file's metadata by ID.

        Args:
            file_id: The ID of the file.
            fields: Comma-separated list of fields to include in the response.

        Returns:
            A DriveFile object.
        """
        url = f"{DRIVE_API_BASE_URL}files/{file_id}"
        params: Dict[str, Any] = {"supportsAllDrives": True}
        if fields:
            params["fields"] = fields
        response_data = self._request("GET", url, params=params)
        return DriveFile(**response_data)


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
