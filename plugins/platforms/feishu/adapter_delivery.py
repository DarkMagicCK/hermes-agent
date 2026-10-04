"""Feishu topic delivery: recover an in-topic anchor before applying the configured policy.

Missing/withdrawn anchors, explicit routing rejections and inability to resolve an
anchor through history lookup use the configured parent-chat policy. Authentication,
permission, flood and ambiguous transport failures of an actual send stay at their
original destination.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any

from gateway.platforms.base import SendResult

if TYPE_CHECKING:
    from plugins.platforms.feishu.adapter import FeishuAdapter

logger = logging.getLogger(__name__)
_REPLY_MISSING_CODES = frozenset({230011, 231003})
_TOPIC_ROUTING_CODES = _REPLY_MISSING_CODES | {99992402}
_TOPIC_ANCHOR_LIMIT = 3
_TOPIC_STATE_KEY = "_feishu_topic_delivery"


def topic_delivery_state(metadata: dict | None) -> dict:
    """Shared by a turn's metadata copies, but never cached across conversations/turns."""
    if metadata is None:
        return {}
    state = metadata.setdefault(_TOPIC_STATE_KEY, {})
    return state


def _identifier(value: Any) -> str:
    return str(value or "").strip()


def _safe_parameter(value: Any) -> str:
    text = _identifier(value)
    return text if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", text) else "unavailable"


class FeishuTopicDeliveryMixin:
    def _topic_delivery_terminal(self: "FeishuAdapter", chat_id: str, reply_to: str | None,
                                 metadata: dict | None) -> SendResult | None:
        state = topic_delivery_state(metadata)
        scope = (getattr(self, "_app_id", None), chat_id, _identifier((metadata or {}).get("thread_id")))
        anchor = _identifier((metadata or {}).get("reply_to_message_id")) or _identifier(reply_to) or None
        if state.get("scope") == scope and state.get("requested_anchor") == anchor:
            return state.get("terminal")
        return None

    async def _list_topic_reply_anchors(self: "FeishuAdapter", thread_id: str, excluded: set[str]) -> list[str] | SendResult:
        from lark_oapi.api.im.v1 import ListMessageRequest

        client = self._client
        if client is None:
            return SendResult(success=False, error="Not connected")
        request = (ListMessageRequest.builder().container_id_type("thread").container_id(thread_id)
                   .sort_type("ByCreateTimeDesc").page_size(20).build())
        try:
            response = await self._run_blocking(client.im.v1.message.list, request)
        except Exception as exc:
            # No outbound message was attempted: inability to read history is distinct
            # from an ambiguous send and may use the configured parent-chat policy.
            return SendResult(success=False, error=f"Feishu topic lookup failed ({type(exc).__name__})")
        if not self._response_succeeded(response):
            return self._response_error_result(response, default_message="topic lookup failed")
        anchors = []
        for item in getattr(getattr(response, "data", None), "items", None) or []:
            anchor = _identifier(getattr(item, "message_id", None))
            if (not anchor or anchor in excluded or anchor in anchors
                    or getattr(item, "deleted", False) is True):
                continue
            # List is scoped to this thread; reject inconsistent rows rather than cross topics.
            item_thread = _identifier(getattr(item, "thread_id", None))
            if item_thread and item_thread != thread_id:
                continue
            anchors.append(anchor)
            if len(anchors) >= _TOPIC_ANCHOR_LIMIT:
                break
        return anchors

    async def _send_to_topic_parent(self: "FeishuAdapter", *, chat_id: str, msg_type: str,
                                    payload: str, state: dict) -> Any:
        response = await self._send_raw_with_retry(
            chat_id=chat_id, msg_type=msg_type, payload=payload, reply_to=None, metadata=None)
        if self._response_succeeded(response):
            message_id = self._extract_response_field(response, "message_id")
            if message_id:
                state.setdefault("parent_message_ids", set()).add(message_id)
        return response

    async def _apply_topic_delivery_fallback(
        self: "FeishuAdapter", *, chat_id: str, thread_id: str, anchor: str | None,
        msg_type: str, payload: str, state: dict, code: Any, stage: str,
    ) -> Any:
        policy = getattr(self, "_topic_delivery_fallback", "main_chat")
        correlation_id = state.setdefault("correlation_id", uuid.uuid4().hex[:16])
        # Never include message bodies, file paths, credentials, or raw API error text.
        diagnostic = (
            f"Feishu topic delivery failed; ref={correlation_id} "
            f"code={_safe_parameter(code)} stage={stage} "
            f"chat_id={_safe_parameter(chat_id)} thread_id={_safe_parameter(thread_id)} "
            f"reply_to={_safe_parameter(anchor)} app_id={_safe_parameter(getattr(self, '_app_id', None))} "
            f"message_type={_safe_parameter(msg_type)} policy={policy}"
        )
        logger.error("[Feishu] %s", diagnostic)
        if policy == "main_chat":
            state["destination"] = "main_chat"
            # Direct parent send: no topic metadata, no recursive policy evaluation.
            return await self._send_to_topic_parent(
                chat_id=chat_id, msg_type=msg_type, payload=payload, state=state)

        result = SendResult(success=False, error=diagnostic, retry_suppressed=True,
                            raw_response={"feishu_topic_delivery": {"policy": policy, "correlation_id": correlation_id}})
        # Store BEFORE await: concurrent progress/media/final sends see the same terminal outcome.
        state["terminal"] = result
        if policy == "error_notice":
            try:
                notice = await self._send_raw_message(
                    chat_id=chat_id, msg_type="text", payload=json.dumps({"text": diagnostic}),
                    reply_to=None, metadata=None,
                )
                if not self._response_succeeded(notice):
                    logger.error("[Feishu] Topic diagnostic failed ref=%s code=%s", correlation_id,
                                 _safe_parameter(getattr(notice, "code", None)))
            except Exception as exc:
                logger.error("[Feishu] Topic diagnostic failed ref=%s exception=%s", correlation_id,
                             type(exc).__name__)
        return result

    async def _feishu_send_with_retry(
        self: "FeishuAdapter", *, chat_id: str, msg_type: str, payload: str, reply_to: str | None, metadata: dict | None,
    ) -> Any:
        thread_id = _identifier((metadata or {}).get("thread_id"))
        anchor = _identifier(reply_to) or _identifier((metadata or {}).get("reply_to_message_id")) or None
        if not thread_id:
            response = await self._send_raw_with_retry(
                chat_id=chat_id, msg_type=msg_type, payload=payload, reply_to=anchor, metadata=metadata)
            if anchor and not self._response_succeeded(response) and getattr(response, "code", None) in _REPLY_MISSING_CODES:
                return await self._send_raw_with_retry(
                    chat_id=chat_id, msg_type=msg_type, payload=payload, reply_to=None, metadata=None)
            return response

        state = topic_delivery_state(metadata)
        # One turn can send progress, stream chunks, final text and media concurrently. Serialize
        # routing decisions (not global chats) so stale recovery and the diagnostic run only once.
        lock = state.setdefault("lock", asyncio.Lock())
        async with lock:
            scope = (getattr(self, "_app_id", None), chat_id, thread_id)
            requested_anchor = _identifier((metadata or {}).get("reply_to_message_id")) or anchor
            is_parent_continuation = anchor in state.get("parent_message_ids", ())
            requested_changed = ("requested_anchor" in state and requested_anchor != state["requested_anchor"]
                                 and not is_parent_continuation)
            if state.get("scope", scope) != scope or requested_changed:
                # A redirect or transport-owner change is a new destination. Never reuse
                # another topic/bot's terminal decision or recovered anchor.
                for key in ("terminal", "destination", "anchor", "correlation_id", "parent_message_ids"):
                    state.pop(key, None)
            state["scope"] = scope
            if not is_parent_continuation:
                state["requested_anchor"] = requested_anchor
            if state.get("terminal") is not None:
                return state["terminal"]
            if state.get("destination") == "main_chat":
                return await self._send_to_topic_parent(
                    chat_id=chat_id, msg_type=msg_type, payload=payload, state=state)
            anchor = state.get("anchor") or anchor
            excluded = set()
            code, stage = "missing_anchor", "resolve_anchor"
            if anchor:
                response = await self._send_raw_with_retry(
                    chat_id=chat_id, msg_type=msg_type, payload=payload, reply_to=anchor, metadata=metadata)
                if self._response_succeeded(response):
                    state["anchor"] = anchor
                    return response
                code = getattr(response, "code", None)
                if code not in _TOPIC_ROUTING_CODES:
                    return response
                excluded.add(anchor)
                stage = "reply_anchor"

            candidates = await self._list_topic_reply_anchors(thread_id, excluded)
            if isinstance(candidates, SendResult):
                return await self._apply_topic_delivery_fallback(
                    chat_id=chat_id, thread_id=thread_id, anchor=anchor, msg_type=msg_type, payload=payload,
                    state=state, code=getattr(candidates.raw_response, "code", None) or "lookup_failed",
                    stage="lookup_failed")
            for candidate in candidates:
                response = await self._send_raw_with_retry(
                    chat_id=chat_id, msg_type=msg_type, payload=payload, reply_to=candidate, metadata=metadata)
                if self._response_succeeded(response):
                    state["anchor"] = candidate
                    return response
                code = getattr(response, "code", None)
                if code not in _TOPIC_ROUTING_CODES:
                    return response
                anchor, stage = candidate, "reanchor"
            return await self._apply_topic_delivery_fallback(
                chat_id=chat_id, thread_id=thread_id, anchor=anchor, msg_type=msg_type, payload=payload,
                state=state, code=code, stage=stage)

    async def _send_raw_with_retry(self: "FeishuAdapter", **kwargs) -> Any:
        # Feishu deduplicates UUIDs. Reusing one for retries to the SAME destination avoids
        # duplicates when a successful request loses its ACK; reanchoring gets a new UUID.
        request_uuid = str(uuid.uuid4())
        for attempt in range(3):
            try:
                return await self._send_raw_message(**kwargs, uuid_value=request_uuid)
            except Exception as exc:
                if attempt == 2 or (kwargs["msg_type"] == "post" and re.search(
                        "content format of the post type is incorrect", str(exc), re.IGNORECASE)):
                    raise
                logger.warning("[Feishu] Send transport retry attempt=%d exception=%s", attempt + 1,
                               type(exc).__name__)
                await asyncio.sleep(2 ** attempt)

    async def _send_raw_message(
        self: "FeishuAdapter", *, chat_id: str, msg_type: str, payload: str, reply_to: str | None,
        metadata: dict | None, uuid_value: str | None = None,
    ) -> Any:
        thread_id = _identifier((metadata or {}).get("thread_id"))
        anchor = _identifier(reply_to) or _identifier((metadata or {}).get("reply_to_message_id")) or None
        uuid_value = uuid_value or str(uuid.uuid4())
        client = self._client
        if client is None:
            raise RuntimeError("Not connected")
        if anchor:
            body = self._build_reply_message_body(
                content=payload, msg_type=msg_type, reply_in_thread=bool(thread_id), uuid_value=uuid_value)
            return await self._run_blocking(
                client.im.v1.message.reply, self._build_reply_message_request(anchor, body))
        if thread_id:
            raise ValueError("Feishu topic send requires a reply anchor; use topic delivery policy")
        if chat_id.startswith("feishu_user_id:"):
            receive_id, receive_id_type = chat_id.split(":", 1)[1], "user_id"
        else:
            receive_id, receive_id_type = chat_id, "open_id" if chat_id.startswith("ou_") else "chat_id"
        body = self._build_create_message_body(
            receive_id=receive_id, msg_type=msg_type, content=payload, uuid_value=uuid_value)
        return await self._run_blocking(
            client.im.v1.message.create, self._build_create_message_request(receive_id_type, body))
