"""Per-category privacy filtering.

The privacy filter is applied to Gmail data BEFORE it is shown to the user in
the review UI. It is a *floor*, not a ceiling: it removes/redacts data the user
has decided should never reach Claude, but the user still has to approve every
single request on top of whatever the filter leaves in place.

Categories:
  - body            : message body (text + html)
  - metadata        : sender / recipients / date / subject
  - attachments     : attachment metadata (never content)
  - thread_history  : prior messages in a thread

Each category policy is one of: allow, redact, block.
  - allow  : pass through unchanged
  - redact : partially mask the value
  - block  : replace with the BLOCKED sentinel
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .gmail_client import Attachment, GmailMessage, GmailThread

logger = logging.getLogger(__name__)

BLOCKED = "[BLOCKED BY PRIVACY FILTER]"
REDACTED = "[REDACTED]"

VALID_POLICIES = {"allow", "redact", "block"}
CATEGORIES = ("body", "metadata", "attachments", "thread_history")


@dataclass
class FilteredMessage:
    """A message after privacy filtering. Mirrors GmailMessage fields but the
    values may be blocked/redacted. This is what gets shown to the user and,
    on approval, returned to Claude."""

    id: str
    thread_id: str
    subject: str
    sender: str
    recipients: list[str] = field(default_factory=list)
    date: str = ""
    body_text: str = ""
    body_html: str = ""
    attachments: list[dict[str, Any]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "subject": self.subject,
            "sender": self.sender,
            "recipients": self.recipients,
            "date": self.date,
            "body_text": self.body_text,
            "body_html": self.body_html,
            "attachments": self.attachments,
            "labels": self.labels,
        }


@dataclass
class FilteredThread:
    id: str
    subject: str
    messages: list[FilteredMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "messages": [m.to_dict() for m in self.messages],
        }


class PrivacyFilter:
    """Config-driven per-category filter.

    ``settings`` is the ``privacy`` section of the config:
        {
          "default_policy": "block",
          "categories": {"body": "allow", "metadata": "allow", ...}
        }
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        self._default_policy = self._normalize_policy(
            settings.get("default_policy", "block")
        )
        raw_categories = settings.get("categories", {}) or {}
        # Resolve every known category to a concrete policy now so lookups are
        # cheap and deterministic later.
        self._policies: dict[str, str] = {}
        for category in CATEGORIES:
            self._policies[category] = self._normalize_policy(
                raw_categories.get(category, self._default_policy)
            )
        logger.info(
            "PrivacyFilter initialized: default=%s categories=%s",
            self._default_policy,
            self._policies,
        )

    def _normalize_policy(self, policy: Any) -> str:
        value = str(policy).strip().lower()
        if value not in VALID_POLICIES:
            logger.warning(
                "Invalid privacy policy %r; falling back to 'block'", policy
            )
            return "block"
        return value

    def policy_for(self, category: str) -> str:
        return self._policies.get(category, self._default_policy)

    def set_policy(self, category: str, policy: str) -> None:
        """Update a single category policy at runtime (menu bar toggles)."""
        if category not in CATEGORIES:
            raise ValueError(f"Unknown privacy category: {category}")
        self._policies[category] = self._normalize_policy(policy)
        logger.info("Privacy policy for %s set to %s", category, self._policies[category])

    def policies(self) -> dict[str, str]:
        """Return a copy of current policies (for the UI)."""
        return dict(self._policies)

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #
    def filter_message(self, msg: GmailMessage) -> FilteredMessage:
        """Apply the per-category policies to a single message."""
        body_policy = self.policy_for("body")
        meta_policy = self.policy_for("metadata")
        attach_policy = self.policy_for("attachments")

        return FilteredMessage(
            id=msg.id,
            thread_id=msg.thread_id,
            subject=self._apply_text(msg.subject, meta_policy),
            sender=self._apply_address(msg.sender, meta_policy),
            recipients=[self._apply_address(r, meta_policy) for r in msg.recipients],
            date=self._apply_text(msg.date, meta_policy),
            body_text=self._apply_body(msg.body_text, body_policy),
            body_html=self._apply_body(msg.body_html, body_policy),
            attachments=self._apply_attachments(msg.attachments, attach_policy),
            labels=msg.labels,
        )

    def filter_thread(self, thread: GmailThread) -> FilteredThread:
        """Apply policies to a thread.

        ``thread_history`` controls whether prior messages survive: if blocked,
        only the most recent message is kept; the rest are dropped.
        """
        history_policy = self.policy_for("thread_history")
        messages = thread.messages

        if history_policy == "block" and len(messages) > 1:
            logger.debug(
                "thread_history=block: keeping only latest of %d messages",
                len(messages),
            )
            messages = messages[-1:]

        filtered = [self.filter_message(m) for m in messages]

        if history_policy == "redact" and len(thread.messages) > 1:
            # Keep the latest message intact, redact the bodies of older ones.
            for older in filtered[:-1]:
                older.body_text = REDACTED
                older.body_html = REDACTED

        subject = self._apply_text(thread.subject, self.policy_for("metadata"))
        return FilteredThread(id=thread.id, subject=subject, messages=filtered)

    # ------------------------------------------------------------------ #
    # Value-level helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_text(value: str, policy: str) -> str:
        if not value:
            return value
        if policy == "allow":
            return value
        if policy == "block":
            return BLOCKED
        # redact: show nothing useful but keep the field present
        return REDACTED

    @staticmethod
    def _apply_body(value: str, policy: str) -> str:
        if not value:
            return value
        if policy == "allow":
            return value
        if policy == "block":
            return BLOCKED
        # redact body: keep a short preview, mask the rest
        preview = value.strip().splitlines()
        first_line = preview[0][:80] if preview else ""
        return f"{first_line} ... {REDACTED}"

    @staticmethod
    def _apply_address(value: str, policy: str) -> str:
        if not value:
            return value
        if policy == "allow":
            return value
        if policy == "block":
            return BLOCKED
        # redact: keep the domain, mask the local part -> ***@example.com
        if "@" in value:
            # Handle "Name <user@domain>" form too.
            local_and_domain = value.split("<")[-1].rstrip(">")
            if "@" in local_and_domain:
                domain = local_and_domain.split("@", 1)[1]
                return f"***@{domain}"
        return REDACTED

    def _apply_attachments(
        self, attachments: list[Attachment], policy: str
    ) -> list[dict[str, Any]]:
        if policy == "block":
            if attachments:
                return [{"name": BLOCKED, "mime_type": BLOCKED, "size": 0}]
            return []
        result: list[dict[str, Any]] = []
        for att in attachments:
            if policy == "redact":
                result.append(
                    {"name": REDACTED, "mime_type": att.mime_type, "size": att.size}
                )
            else:  # allow
                result.append(
                    {"name": att.name, "mime_type": att.mime_type, "size": att.size}
                )
        return result


# ---------------------------------------------------------------------------- #
# Google Drive privacy filtering
# ---------------------------------------------------------------------------- #
# Drive needs its own categories. We keep a separate, self-contained filter
# class here (mirroring PrivacyFilter) rather than overloading the Gmail one, so
# the two services stay independent and the menu bar can drive either of them.
#
# Categories:
#   - file_content     : the actual document text / bytes (highest risk)
#   - file_metadata    : name / owners / created/modified times / sharing status
#   - file_list        : results of list_files operations (names + ids)
#   - folder_structure : results of folder listing operations

from .drive_client import DriveFile, DriveFileContent  # noqa: E402

DRIVE_CATEGORIES = ("file_content", "file_metadata", "file_list", "folder_structure")


@dataclass
class FilteredDriveFile:
    """A Drive file's metadata after privacy filtering."""

    id: str
    name: str
    mime_type: str
    size: int
    created_time: str = ""
    modified_time: str = ""
    owners: list[str] = field(default_factory=list)
    shared: Any = False
    web_view_link: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "mime_type": self.mime_type,
            "size": self.size,
            "created_time": self.created_time,
            "modified_time": self.modified_time,
            "owners": self.owners,
            "shared": self.shared,
            "web_view_link": self.web_view_link,
        }


@dataclass
class FilteredDriveFileContent:
    """A Drive file's content after privacy filtering."""

    file: FilteredDriveFile
    content_text: str = ""
    content_bytes_info: str = ""  # description of binary content (never raw bytes)
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file.to_dict(),
            "content_text": self.content_text,
            "content_bytes_info": self.content_bytes_info,
            "truncated": self.truncated,
        }


class DrivePrivacyFilter:
    """Config-driven per-category filter for Google Drive data.

    ``settings`` is the ``drive_privacy`` section of the config:
        {
          "default_policy": "block",
          "categories": {"file_content": "allow", "file_metadata": "allow", ...}
        }
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        self._default_policy = self._normalize_policy(
            settings.get("default_policy", "block")
        )
        raw_categories = settings.get("categories", {}) or {}
        self._policies: dict[str, str] = {}
        for category in DRIVE_CATEGORIES:
            self._policies[category] = self._normalize_policy(
                raw_categories.get(category, self._default_policy)
            )
        logger.info(
            "DrivePrivacyFilter initialized: default=%s categories=%s",
            self._default_policy,
            self._policies,
        )

    def _normalize_policy(self, policy: Any) -> str:
        value = str(policy).strip().lower()
        if value not in VALID_POLICIES:
            logger.warning(
                "Invalid privacy policy %r; falling back to 'block'", policy
            )
            return "block"
        return value

    def policy_for(self, category: str) -> str:
        return self._policies.get(category, self._default_policy)

    def set_policy(self, category: str, policy: str) -> None:
        """Update a single category policy at runtime (menu bar toggles)."""
        if category not in DRIVE_CATEGORIES:
            raise ValueError(f"Unknown drive privacy category: {category}")
        self._policies[category] = self._normalize_policy(policy)
        logger.info(
            "Drive privacy policy for %s set to %s",
            category,
            self._policies[category],
        )

    def policies(self) -> dict[str, str]:
        """Return a copy of current policies (for the UI)."""
        return dict(self._policies)

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #
    def filter_file_metadata(self, drive_file: DriveFile) -> FilteredDriveFile:
        """Apply the file_metadata policy to a single file's metadata.

        ``id`` and ``mime_type`` are structural identifiers and always pass
        through so the result remains actionable; everything else honors the
        metadata policy.
        """
        policy = self.policy_for("file_metadata")
        return self._filter_file(drive_file, policy)

    def filter_file_list(self, files: list[DriveFile]) -> list[dict[str, Any]]:
        """Apply the file_list policy to list_files results."""
        policy = self.policy_for("file_list")
        return [self._filter_file(f, policy).to_dict() for f in files]

    def filter_folder_listing(self, files: list[DriveFile]) -> list[dict[str, Any]]:
        """Apply the folder_structure policy to folder listing results."""
        policy = self.policy_for("folder_structure")
        return [self._filter_file(f, policy).to_dict() for f in files]

    def filter_file_content(
        self, content: DriveFileContent
    ) -> FilteredDriveFileContent:
        """Apply policies to a file's content.

        The body text/bytes honor the file_content policy; the embedded file
        metadata honors the file_metadata policy.
        """
        content_policy = self.policy_for("file_content")
        meta_policy = self.policy_for("file_metadata")

        filtered_file = self._filter_file(content.file, meta_policy)

        text = self._apply_content_text(content.content_text, content_policy)
        bytes_info = self._apply_content_bytes(
            content.content_bytes, content_policy
        )
        return FilteredDriveFileContent(
            file=filtered_file,
            content_text=text,
            content_bytes_info=bytes_info,
            truncated=content.truncated,
        )

    # ------------------------------------------------------------------ #
    # Value-level helpers
    # ------------------------------------------------------------------ #
    def _filter_file(self, drive_file: DriveFile, policy: str) -> FilteredDriveFile:
        return FilteredDriveFile(
            id=drive_file.id,
            name=self._apply_text(drive_file.name, policy),
            mime_type=drive_file.mime_type,
            size=drive_file.size,
            created_time=self._apply_text(drive_file.created_time, policy),
            modified_time=self._apply_text(drive_file.modified_time, policy),
            owners=[self._apply_address(o, policy) for o in drive_file.owners],
            shared=BLOCKED if policy == "block" else drive_file.shared,
            web_view_link=self._apply_text(drive_file.web_view_link, policy),
        )

    @staticmethod
    def _apply_text(value: str, policy: str) -> str:
        if not value:
            return value
        if policy == "allow":
            return value
        if policy == "block":
            return BLOCKED
        return REDACTED

    @staticmethod
    def _apply_address(value: str, policy: str) -> str:
        if not value:
            return value
        if policy == "allow":
            return value
        if policy == "block":
            return BLOCKED
        if "@" in value:
            domain = value.split("@", 1)[1]
            return f"***@{domain}"
        return REDACTED

    @staticmethod
    def _apply_content_text(value: str, policy: str) -> str:
        if not value:
            return value
        if policy == "allow":
            return value
        if policy == "block":
            return BLOCKED
        preview = value.strip().splitlines()
        first_line = preview[0][:80] if preview else ""
        return f"{first_line} ... {REDACTED}"

    @staticmethod
    def _apply_content_bytes(value: bytes, policy: str) -> str:
        """Binary content is never returned raw; describe it instead."""
        if not value:
            return ""
        if policy == "block":
            return BLOCKED
        if policy == "redact":
            return f"[binary content: {len(value)} bytes] {REDACTED}"
        # allow: still do not emit raw bytes over MCP - report the size.
        return f"[binary content: {len(value)} bytes]"


# ---------------------------------------------------------------------------- #
# Slack privacy filtering
# ---------------------------------------------------------------------------- #
# Slack needs its own categories. We keep a separate, self-contained filter
# class here (mirroring PrivacyFilter) rather than overloading the others, so
# the services stay independent and the menu bar can drive any of them.
#
# Categories:
#   - message_content : the text of messages
#   - user_identity   : user names, emails, real names
#   - channel_list    : channel names and metadata
#   - thread_content  : thread reply content
#
# Each category policy is one of: allow, redact, block (same semantics as above).

SLACK_CATEGORIES = (
    "message_content",
    "user_identity",
    "channel_list",
    "thread_content",
)


class SlackPrivacyFilter:
    """Config-driven per-category filter for Slack data.

    ``settings`` is the ``slack_privacy`` section of the config:
        {
          "default_policy": "block",
          "categories": {"message_content": "allow", "user_identity": "allow", ...}
        }

    Returns plain dicts (the shape returned to Claude on approval). ``id`` and
    structural identifiers (channel_id, user_id, thread_ts, timestamp) always
    pass through so results remain actionable; the rest honor their category
    policy.
    """

    def __init__(self, settings: dict[str, Any]) -> None:
        self._default_policy = self._normalize_policy(
            settings.get("default_policy", "block")
        )
        raw_categories = settings.get("categories", {}) or {}
        self._policies: dict[str, str] = {}
        for category in SLACK_CATEGORIES:
            self._policies[category] = self._normalize_policy(
                raw_categories.get(category, self._default_policy)
            )
        logger.info(
            "SlackPrivacyFilter initialized: default=%s categories=%s",
            self._default_policy,
            self._policies,
        )

    def _normalize_policy(self, policy: Any) -> str:
        value = str(policy).strip().lower()
        if value not in VALID_POLICIES:
            logger.warning(
                "Invalid Slack privacy policy %r; falling back to 'block'", policy
            )
            return "block"
        return value

    def policy_for(self, category: str) -> str:
        return self._policies.get(category, self._default_policy)

    def set_policy(self, category: str, policy: str) -> None:
        """Update a single category policy at runtime (menu bar toggles)."""
        if category not in SLACK_CATEGORIES:
            raise ValueError(f"Unknown Slack privacy category: {category}")
        self._policies[category] = self._normalize_policy(policy)
        logger.info(
            "Slack privacy policy for %s set to %s",
            category,
            self._policies[category],
        )

    def policies(self) -> dict[str, str]:
        """Return a copy of current policies (for the UI)."""
        return dict(self._policies)

    # ------------------------------------------------------------------ #
    # Filtering
    # ------------------------------------------------------------------ #
    def filter_message(self, msg: Any) -> dict[str, Any]:
        """Apply per-category policies to a single SlackMessage."""
        return self._filter_message(msg, self.policy_for("message_content"))

    def filter_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        return [self.filter_message(m) for m in messages]

    def filter_thread(self, messages: list[Any]) -> list[dict[str, Any]]:
        """Apply policies to a thread's replies.

        The first message (thread root) honors ``message_content`` so a blocked
        ``thread_content`` still surfaces the originating message; subsequent
        replies honor ``thread_content``.
        """
        thread_policy = self.policy_for("thread_content")
        root_policy = self.policy_for("message_content")
        result: list[dict[str, Any]] = []
        for index, msg in enumerate(messages):
            policy = root_policy if index == 0 else thread_policy
            result.append(self._filter_message(msg, policy))
        return result

    def filter_channels(self, channels: list[Any]) -> list[dict[str, Any]]:
        """Apply the ``channel_list`` policy to a list of SlackChannel."""
        policy = self.policy_for("channel_list")
        result: list[dict[str, Any]] = []
        for ch in channels:
            result.append(
                {
                    "id": ch.id,
                    "name": self._apply_text(ch.name, policy),
                    "is_private": ch.is_private,
                    "topic": self._apply_text(ch.topic, policy),
                    "purpose": self._apply_text(ch.purpose, policy),
                    "member_count": ch.member_count,
                }
            )
        return result

    # ------------------------------------------------------------------ #
    # Value-level helpers
    # ------------------------------------------------------------------ #
    def _filter_message(self, msg: Any, content_policy: str) -> dict[str, Any]:
        identity_policy = self.policy_for("user_identity")
        return {
            "id": msg.id,
            "channel_id": msg.channel_id,
            "channel_name": msg.channel_name,
            "user_id": msg.user_id,
            "user_name": self._apply_text(msg.user_name, identity_policy),
            "text": self._apply_body(msg.text, content_policy),
            "thread_ts": msg.thread_ts,
            "reply_count": msg.reply_count,
            "attachments": msg.attachments if content_policy == "allow" else [],
            "files": self._apply_files(msg.files, content_policy),
            "timestamp": msg.timestamp.isoformat() if msg.timestamp else "",
        }

    def _apply_files(self, files: list[Any], policy: str) -> list[dict[str, Any]]:
        if policy == "block":
            if files:
                return [{"name": BLOCKED, "mimetype": BLOCKED, "size": 0}]
            return []
        result: list[dict[str, Any]] = []
        for f in files:
            if policy == "redact":
                result.append(
                    {"name": REDACTED, "mimetype": f.mimetype, "size": f.size}
                )
            else:  # allow
                result.append(
                    {
                        "id": f.id,
                        "name": f.name,
                        "title": f.title,
                        "mimetype": f.mimetype,
                        "size": f.size,
                        "url_private": f.url_private,
                    }
                )
        return result

    @staticmethod
    def _apply_text(value: str, policy: str) -> str:
        if not value:
            return value
        if policy == "allow":
            return value
        if policy == "block":
            return BLOCKED
        return REDACTED

    @staticmethod
    def _apply_body(value: str, policy: str) -> str:
        if not value:
            return value
        if policy == "allow":
            return value
        if policy == "block":
            return BLOCKED
        preview = value.strip().splitlines()
        first_line = preview[0][:80] if preview else ""
        return f"{first_line} ... {REDACTED}"
