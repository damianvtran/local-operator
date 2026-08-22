"""Credentials management for Local Operator.

This module handles API key storage and retrieval for various AI services.
It securely stores credentials in a local config file and provides methods
for accessing them when needed.
"""

import getpass
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

from pydantic import SecretStr

from local_operator.cli_style import CYAN, SUCCESS, can_encode, paint

# Name of the file used to store credentials in .env format
CREDENTIALS_FILE_NAME: str = "credentials.env"

#: Restrictive mode for the credentials file and its directory. Credentials are
#: the highest-sensitivity data the app writes, so both the file (0600) and the
#: directory that holds it (0700) exclude every other account on a shared host.
_CREDENTIALS_MODE = 0o600
_CONFIG_DIR_MODE = 0o700


def _reject_control_chars(key: str, value: str) -> None:
    """Refuse a credential value that would corrupt the flat ``key=value`` file.

    The store is one ``key=value`` line per credential with no quoting or
    escaping, so a newline in a value silently splits it into a second, bogus
    entry (and a NUL/other control byte produces a line the loader cannot read
    back). Rejecting at the boundary keeps the format an invariant rather than
    something every writer has to remember to sanitise; a legitimate API key or
    token never contains one of these bytes.
    """
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise ValueError(f"{key} contains control characters, which are not allowed.")


class CredentialManager:
    """Manages secure storage and retrieval of API credentials.

    This class handles storing API keys and other sensitive credentials in a local
    encrypted configuration file. It provides methods for safely reading and writing
    credentials while maintaining proper file permissions.

    Attributes:
        config_dir (Path): Directory where credential files are stored
        config_file (Path): Path to the credentials file
        credentials (Dict[str, SecretStr]): Dictionary of credentials
    """

    config_dir: Path
    config_file: Path
    credentials: Dict[str, SecretStr]

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        self.config_file = self.config_dir / CREDENTIALS_FILE_NAME
        self._ensure_config_exists()
        self.load_from_file()

    def load_from_file(self) -> Dict[str, SecretStr]:
        """Load credentials from the config file."""
        self.credentials = {}

        with open(self.config_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    self.credentials[key] = SecretStr(value)

        return self.credentials

    def write_to_file(self) -> None:
        """Write credentials to the config file atomically.

        Write-to-temp then ``os.replace`` rather than truncating the real file:
        a crash or full disk mid-write used to leave the file half-written,
        losing every key the process had — the store has no backup, so a
        partial write is unrecoverable data loss. ``os.replace`` is atomic on a
        single filesystem, so a reader either sees the whole old file or the
        whole new one, never a torn one.

        The temp file is opened with 0600 via :func:`os.open` so the secret is
        never briefly world-readable between create and chmod, and it is created
        in the SAME directory as the target so the final ``os.replace`` is a
        rename within one filesystem (a cross-device replace is not atomic and
        would raise).
        """
        body = "".join(
            f"{key}={value.get_secret_value()}\n" for key, value in self.credentials.items()
        )
        directory = self.config_file.parent
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(directory), prefix=".credentials-", suffix=".tmp")
        try:
            os.fchmod(fd, _CREDENTIALS_MODE)
            with os.fdopen(fd, "w") as f:
                f.write(body)
            os.replace(tmp_path, self.config_file)
        except BaseException:
            # Leave no half-written temp file behind on any failure path,
            # including KeyboardInterrupt — the real file is still intact
            # because the replace had not run yet.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _ensure_config_exists(self) -> None:
        """Ensure the credentials configuration file exists and has proper permissions.

        Creates the config directory and credentials file if they don't exist.
        Sets restrictive file permissions (600) to protect sensitive credential data.
        The file permissions ensure only the owner can read/write the credentials.

        The config file is created as an empty file that will be populated later
        when credentials are added via set_credential().
        """
        if not self.config_file.exists():
            # 0700 on the directory, matching the log dir (paths.ensure_log_dir):
            # transcripts and credentials are the same sensitivity class, and the
            # default 0755 exposed the directory listing to every other account
            # on a shared machine. Only set at CREATION — never chmod an existing
            # directory, which may hold other things a user deliberately shared.
            self.config_file.parent.mkdir(parents=True, exist_ok=True, mode=_CONFIG_DIR_MODE)
            self.config_file.touch()
            self.config_file.chmod(_CREDENTIALS_MODE)
        else:
            # Re-tighten a pre-existing credentials file that is looser than
            # 0600. A file created by an older build (or copied in by hand) may
            # be world-readable; this is the one moment we can quietly fix it
            # without touching a directory the user may share on purpose.
            try:
                if (self.config_file.stat().st_mode & 0o077) != 0:
                    self.config_file.chmod(_CREDENTIALS_MODE)
            except OSError:
                pass

    def get_credentials(self) -> Dict[str, SecretStr]:
        """Get all credentials from the config file."""
        return self.credentials

    def get_credential(self, key: str) -> SecretStr:
        """Retrieve the credential from config file.

        Args:
            key (str): The environment variable key to retrieve

        Returns:
            SecretStr: The credential value wrapped in SecretStr
        """
        if key not in self.credentials:
            # Check if the key is in the environment variables
            if key in os.environ:
                self.set_credential(key, os.environ[key], write=False)

        return self.credentials.get(key, SecretStr(""))

    def list_credential_keys(self, non_empty: bool = True) -> List[str]:
        """List all credential keys from the config file.

        Args:
            non_empty (bool): Whether to filter out empty credentials

        Returns:
            List[str]: List of credential keys
        """
        output = []

        for key, value in self.get_credentials().items():
            if not non_empty or value:
                output.append(key)

        return output

    def set_credential(self, key: str, value: str, write: bool = True) -> None:
        """Set the credential in the config file.
        If the key already exists, it will be updated.
        If the key does not exist, it will be added.

        Args:
            key (str): The environment variable key to set
            value (str): The credential value to set
            write (bool): Whether to write the credential to the config file

        Raises:
            ValueError: if the value contains control characters that would
                corrupt the flat ``key=value`` store.
        """
        _reject_control_chars(key, value)
        self.credentials[key] = SecretStr(value)

        if write:
            self.write_to_file()

    def prompt_for_credential(
        self, key: str, reason: str = "not found in configuration"
    ) -> SecretStr:
        """Prompt the user to enter a credential if not present in environment.

        Args:
            key (str): The environment variable key to check
            reason (str): The reason for prompting the user

        Returns:
            SecretStr: The credential value wrapped in SecretStr

        Raises:
            ValueError: If the user enters an empty credential.
            EOFError: If stdin closes before a value is read (piped empty input).
            KeyboardInterrupt: If the user cancels the prompt.
        """
        # Calculate border length based on key length
        line_length = max(50, len(key) + 12)
        # Box drawing is decorative; on a stdout whose encoding cannot represent
        # it (PYTHONIOENCODING=ascii, a legacy Windows code page) drawing it
        # crashed the prompt with UnicodeEncodeError before it could ask for
        # anything. Fall back to ASCII rules so the prompt still works there.
        heavy = can_encode("─╭╮├┤╰╯")
        h, tl, tr, ml, mr, bl, br = (
            ("─", "╭", "╮", "├", "┤", "╰", "╯")
            if heavy
            else (
                "-",
                "+",
                "+",
                "+",
                "+",
                "+",
                "+",
            )
        )
        border = h * line_length

        # Colour is gated on NO_COLOR/tty/TERM by ``paint`` \u2014 a raw escape here
        # painted literal ``[1;36m`` into a piped or dumb-terminal transcript.
        def cyan(text: str) -> str:
            return paint(text, CYAN)

        # Print the setup box
        print(cyan(f"{tl}{border}{tr}"))
        setup_padding = " " * (line_length - len(key) - 7)
        print(
            cyan(f"│ {key} Setup{setup_padding}│")
            if heavy
            else cyan(f"| {key} Setup{setup_padding}|")
        )
        print(cyan(f"{ml}{border}{mr}"))
        reason_padding = " " * (line_length - len(key) - len(reason) - 3)
        body = f"{key} {reason}."
        print(cyan(f"│ {body}{reason_padding}│") if heavy else cyan(f"| {body}{reason_padding}|"))
        print(cyan(f"{bl}{border}{br}"))

        prompt = paint(f"Please enter your {key}: ", "1;94")
        if sys.stdin.isatty():
            # Interactive terminal: getpass hides the key so it never lands in
            # scrollback of a session that may be screen-shared.
            credential = getpass.getpass(prompt).strip()
        else:
            # Non-interactive stdin (piped/scripted): getpass on a non-tty falls
            # back to echoing the input with a warning, which is both noisy and
            # useless for automation. Read one line from stdin instead \u2014 this is
            # the documented automation contract: `printf '%s\\n' "$KEY" |
            # local-operator credential update NAME`. An empty pipe raises
            # EOFError, which the command handler turns into one plain line.
            print(prompt, end="", flush=True)
            line = sys.stdin.readline()
            if line == "":
                raise EOFError(f"{key} is required for this step.")
            credential = line.strip()
        if not credential:
            raise ValueError(f"{key} is required for this step.")

        # Save the new API key to config file
        self.set_credential(key, credential, write=True)

        # ASCII fallback for the check glyph too: a stdout that cannot encode the
        # box drawing cannot encode ✓ either, and crashing on the SUCCESS line
        # after the key is already saved is the worst place to fail (item 14).
        tick = "✓" if can_encode("✓") else "[ok]"
        print(paint(f"\n{tick} Credential successfully saved!", SUCCESS))

        return SecretStr(credential)
