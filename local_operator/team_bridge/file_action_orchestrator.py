"""Unregistered human-file action orchestrator for a trusted desktop client.

The caller MUST be admitted desktop main, not renderer claims. A future adapter
must gate the actual selected backend as direct-local before making these calls,
and own a boot-unique window identity and monotonically increasing generation.
The desktop bearer is shared: this module does not attest Electron main, a human
gesture, or protect against a compromised admitted renderer. The hub remains the
membership authority. Cross-process invalidation has delivery latency and cannot
recall an HTTP request already started. No router or IPC binding is installed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.file_action_registry import (
    ActionUnavailable,
    InertActionBinding,
    InertFileActionRegistry,
    InertOperationView,
)
from local_operator.team_bridge.file_download import (
    FileActionFence,
    download_human_file,
    verify_file_action_current,
)
from local_operator.team_bridge.file_history_action import (
    HistoryScope,
    commission_human_history,
)


@dataclass(frozen=True, repr=False)
class TrustedMainView:
    """Read only from the admitted main lifecycle provider, never request JSON."""

    window_id: str
    generation: int
    tenant_id: str
    conversation_id: str
    message_id: str


class HumanFileActionOrchestrator:
    """Compose existing no-refresh seams; dependencies do not make this a route.

    `current_view` is a synchronous internal provider for main-owned state. It
    must fail closed for closed/replaced windows, tenant changes, or a backend
    no longer known direct-local. Renderer-supplied view/account/credential
    fields must never be passed through as this provider's answer.
    """

    def __init__(
        self,
        client: TeamBridgeClient,
        registry: InertFileActionRegistry,
        current_view: Callable[[str], TrustedMainView | None],
    ) -> None:
        self._client = client
        self._registry = registry
        self._current_view = current_view

    def _view(self, window_id: str) -> TrustedMainView:
        try:
            view = self._current_view(window_id)
        except Exception:
            raise ActionUnavailable() from None
        if not isinstance(view, TrustedMainView) or view.window_id != window_id:
            raise ActionUnavailable()
        return view

    def _native_identity(self) -> tuple[int, str, str]:
        # The client's existing exact-pin / exactly-one selection never refreshes.
        # Read the lifecycle tuple independently; helper rechecks it before HTTP.
        try:
            row = self._client._pinned_row()
            if row is None or type(row.id) is not int or row.id <= 0:
                raise ActionUnavailable()
            store = self._client._auth_store()
            rows = store.native_lifecycle_snapshot(store.db_path, "radient")
        except Exception:
            # Store failures may contain paths or credential details.
            raise ActionUnavailable() from None
        if not isinstance(rows, list) or any(not isinstance(r, tuple) or len(r) != 9 for r in rows):
            raise ActionUnavailable()
        matches = [r for r in rows if r[0] == row.id]
        if len(matches) != 1:
            raise ActionUnavailable()
        item = matches[0]
        if (
            item[1] != row.updated_at
            or item[3] is not None
            or item[4] != "oauth"
            or not isinstance(item[2], str)
            or not isinstance(item[5], str)
            or item[8] is not None
        ):
            raise ActionUnavailable()
        return row.id, item[5], item[2]

    @staticmethod
    def _same_view(binding: InertActionBinding, view: TrustedMainView) -> bool:
        return (
            binding.window_id,
            binding.view_generation,
            binding.tenant_id,
            binding.conversation_id,
            binding.message_id,
        ) == (
            view.window_id,
            view.generation,
            view.tenant_id,
            view.conversation_id,
            view.message_id,
        )

    async def commission(self, window_id: str) -> str:
        # Pin epoch BEFORE any history request; rotation invalidates delayed work.
        epoch = self._registry.current_epoch()
        view = self._view(window_id)
        credential_id, account_id, incarnation = self._native_identity()
        scope = HistoryScope(
            credential_id,
            account_id,
            incarnation,
            view.tenant_id,
            view.window_id,
            view.generation,
            view.conversation_id,
            view.message_id,
            epoch,
        )

        def current() -> HistoryScope:
            latest = self._view(window_id)
            return HistoryScope(
                credential_id,
                account_id,
                incarnation,
                latest.tenant_id,
                latest.window_id,
                latest.generation,
                latest.conversation_id,
                latest.message_id,
                epoch,
            )

        return await commission_human_history(self._client, self._registry, scope, current)

    def begin(self, window_id: str, context: str, file_id: str) -> str:
        binding = self._registry.inspect_context(context)
        if not self._same_view(binding, self._view(window_id)):
            raise ActionUnavailable()
        operation = self._registry.start(context, file_id)
        try:
            # Invalidation can race start; no operation is returned on an observed
            # transition. Registry claim rechecks liveness again before bytes.
            if (
                self._registry.inspect_context(context) != binding
                or not self._same_view(binding, self._view(window_id))
                or self._native_identity()
                != (binding.credential_id, binding.account_id, binding.login_incarnation)
            ):
                raise ActionUnavailable()
        except Exception:
            self._registry.cancel(operation)
            raise
        return operation

    def _fence(
        self, window_id: str, operation: str, expected: InertOperationView
    ) -> FileActionFence:
        current = self._registry.current(operation)
        if current != expected or not self._same_view(current.binding, self._view(window_id)):
            raise ActionUnavailable()
        b = current.binding
        return FileActionFence(
            b.account_id,
            b.credential_id,
            b.login_incarnation,
            b.tenant_id,
            b.view_generation,
            b.conversation_id,
            b.message_id,
            current.file_id,
        )

    async def bytes(self, window_id: str, operation: str) -> bytes:
        # Claim is one-shot and happens before any HTTP. All resource IDs come
        # from the same-response registry binding, never the operation request.
        pending = self._registry.inspect_operation(operation)
        if not self._same_view(pending.binding, self._view(window_id)):
            raise ActionUnavailable()
        selected = self._registry.claim(operation)
        if selected != pending:
            raise ActionUnavailable()
        fence = self._fence(window_id, operation, selected)
        return await download_human_file(
            self._client,
            fence.conversation_id,
            fence.message_id,
            fence.attachment_id,
            tenant_id=fence.tenant_id,
            expected_account_id=fence.account_id,
            action_fence=fence,
            current_action_fence=lambda: self._fence(window_id, operation, selected),
        )

    def verify(self, window_id: str, operation: str) -> None:
        selected = self._registry.current(operation)
        fence = self._fence(window_id, operation, selected)
        verify_file_action_current(
            self._client, fence, lambda: self._fence(window_id, operation, selected)
        )

    def cancel(self, window_id: str, operation: str) -> None:
        # Revocation remains possible after the view changed, but another
        # window cannot cancel an operation merely by presenting its handle.
        if self._registry.inspect_operation(operation).binding.window_id != window_id:
            raise ActionUnavailable()
        self._registry.cancel(operation)

    def revoke(self, window_id: str, context: str) -> None:
        if self._registry.inspect_context(context).window_id != window_id:
            raise ActionUnavailable()
        self._registry.revoke(context)

    def invalidate(self, window_id: str, generation: int) -> None:
        self._registry.invalidate_view(window_id, generation)
