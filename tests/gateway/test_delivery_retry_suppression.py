"""Policy-consumed delivery failures remain failures across every gateway retry boundary.

The default result contract keeps ordinary retry/fallback behavior. Only an explicit
terminal outcome suppresses later sends and leaves a diagnostic, non-delivered ledger row.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, ProcessingOutcome, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.stream_consumer import GatewayStreamConsumer


class _Adapter(BasePlatformAdapter):
    def __init__(self, result):
        super().__init__(PlatformConfig(), Platform.SLACK)
        self.send = AsyncMock(return_value=result)
        self.edit_message = AsyncMock(return_value=result)
        self.send_document = AsyncMock(return_value=result)
        self.send_image = AsyncMock(return_value=result)
        self.send_image_file = AsyncMock(return_value=result)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("instance transport mock must be installed")

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def _event():
    return MessageEvent(text="question", message_id="inbound", source=SessionSource(
        platform=Platform.SLACK, chat_id="chat", thread_id="topic", chat_type="group"))


def _terminal():
    # Typed retry hints cannot override the explicit policy decision.
    return SendResult(success=False, error="topic unavailable", retry_suppressed=True,
                      retryable=True, retry_after=0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [
    "retry", "retry_after_transient", "retry_notice_terminal", "ordinary_retry", "ordinary_fallback",
    "stream_first", "stream_commentary", "stream_chunk", "stream_flood",
    "stream_fresh", "stream_edit", "media_file", "media_images", "poststream_media",
    "status", "approval", "media_caption", "queued_media", "ambiguous_timeout", "stream_scope",
])
async def test_only_explicit_terminal_outcomes_stop_later_delivery(path, tmp_path, monkeypatch):
    terminal = _terminal()
    adapter = _Adapter(terminal)
    event = _event()
    sleep = AsyncMock()
    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", sleep)
    if path == "ambiguous_timeout":
        adapter.send.return_value = SendResult(False, error="TimeoutError: ")
        result = await adapter._send_with_retry("chat", "original")
        assert not result.success and not result.retry_suppressed
        adapter.send.assert_awaited_once()
        sleep.assert_not_awaited()
        return
    if path == "stream_scope":
        adapter.platform = Platform.FEISHU
        metadata = {"thread_id": "topic"}
        first = GatewayStreamConsumer(adapter, "chat", metadata=metadata)
        second = GatewayStreamConsumer(adapter, "chat", metadata=metadata)
        one = first._metadata_for_send()
        one["_feishu_topic_delivery"]["terminal"] = terminal
        assert first._metadata_for_send()["_feishu_topic_delivery"]["terminal"] is terminal
        assert second._metadata_for_send()["_feishu_topic_delivery"] == {}
        assert "_feishu_topic_delivery" not in metadata
        return
    if path.startswith("retry") or path.startswith("ordinary"):
        if path in {"retry_after_transient", "retry_notice_terminal"}:
            adapter.send.side_effect = [SendResult(False, error="connection reset", retryable=True), terminal]
        elif path == "ordinary_retry":
            adapter.send.side_effect = [SendResult(False, error="connection reset", retryable=True), SendResult(True)]
        elif path == "ordinary_fallback":
            adapter.send.side_effect = [SendResult(False, error="bad markup"), SendResult(True)]
        result = await adapter._send_with_retry(
            "chat", "original", base_delay=0, max_retries=0 if path == "retry_notice_terminal" else 2)
        if path.startswith("ordinary"):
            assert result.success and not result.retry_suppressed
            assert adapter.send.await_count == 2
            assert sleep.await_count == (path == "ordinary_retry")
        else:
            assert result is terminal and not result.success
            assert adapter.send.await_count == (2 if path in {"retry_after_transient", "retry_notice_terminal"} else 1)
            assert sleep.await_count == (path == "retry_after_transient")
        return
    if path.startswith("stream"):
        consumer = GatewayStreamConsumer(adapter, "chat")
        if path == "stream_first":
            assert not await consumer._send_or_edit("original")
        elif path == "stream_commentary":
            assert not await consumer._send_commentary("original")
        elif path == "stream_chunk":
            assert await consumer._send_new_chunk("original", None) is None
        elif path == "stream_flood":
            assert await consumer._send_with_flood_retry(content="original", retry_log="retry %s") is terminal
        elif path == "stream_fresh":
            consumer._message_id = "preview"
            assert not await consumer._try_fresh_final("original")
        else:
            consumer._message_id = "preview"
            assert not await consumer._send_or_edit("original", finalize=True)
        assert consumer.retry_suppressed_result is terminal
        sends, edits = adapter.send.await_count, adapter.edit_message.await_count
        consumer._reset_message_state()  # a tool boundary must not revive the terminal turn
        consumer._accumulated = "original plus more"
        await consumer._send_or_edit("original plus more", finalize=True)
        await consumer._send_commentary("later commentary")
        await consumer._send_new_chunk("later chunk", None)
        await consumer._send_fallback_final("original plus more")
        await consumer._flush_segment_tail_on_edit_failure()
        assert adapter.send.await_count == sends
        assert adapter.edit_message.await_count == edits
        assert not consumer.final_response_sent and not consumer.final_content_delivered
        sleep.assert_not_awaited()
        return
    if path == "queued_media":
        from gateway.run import GatewayRunner
        attachment = tmp_path / "report.pdf"
        attachment.write_bytes(b"attachment")
        adapter.send.return_value = SendResult(True, message_id="body")
        runner = object.__new__(GatewayRunner)
        result = await runner._deliver_queued_first_response(
            f"answer\nMEDIA: {attachment}", event.source, adapter)
        assert result.success is False and result.retry_suppressed
        assert result._text_already_delivered is True
        assert not hasattr(terminal, "_text_already_delivered")
        adapter.send.assert_awaited_once()
        adapter.send_document.assert_awaited_once()
        return
    if path == "media_caption":
        adapter.warning_notifications_enabled = Mock(return_value=False)
        result = await adapter.emit_media_warning("chat", "media unavailable", caption="caption")
        assert result is terminal
        adapter.send.assert_awaited_once()
        return
    if path == "status":
        from gateway.run import _send_or_update_status_coro
        result = await _send_or_update_status_coro(
            adapter, "chat", "status", "original", {"_feishu_topic_delivery": {"terminal": terminal}})
        assert result is terminal
    elif path == "approval":
        from gateway.run import _approval_send_outcome
        assert _approval_send_outcome(SimpleNamespace(result=lambda **kwargs: terminal), 1) == "suppressed"
    elif path == "media_images":
        result = await adapter.send_multiple_images("chat", [("https://example.com/a.png", ""),
                                                             ("https://example.com/b.png", "")])
        assert result is terminal
        adapter.send_image.assert_awaited_once()
    elif path == "media_file":
        notices = adapter._notify_media_delivery_failure = AsyncMock()
        outcomes = []
        await adapter._deliver_media_attachments(
            event, [("one.pdf", False), ("two.pdf", False)], [],
            force_document_attachments=False, human_delay=0, metadata={}, record_delivery=outcomes.append)
        assert outcomes == [terminal]
        adapter.send_document.assert_awaited_once()
        notices.assert_not_awaited()
    else:
        from gateway.run import GatewayRunner
        files = [tmp_path / "one.pdf", tmp_path / "two.pdf"]
        for file in files:
            file.write_bytes(b"attachment")
        runner = object.__new__(GatewayRunner)
        result = await runner._deliver_media_from_response(
            "\n".join(f"MEDIA: {file}" for file in files), event, adapter, thread_metadata={})
        assert result is terminal
        adapter.send_document.assert_awaited_once()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["normal", "prior", "tts", "queued", "queued_edit", "queued_stream",
                                  "queued_state", "recovery", "atomic_prior"])
async def test_terminal_original_is_retained_but_never_recovered(path, tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (123, 456))
    monkeypatch.setattr(dl, "ledger_enabled", lambda *args: True)
    terminal = _terminal()
    adapter = _Adapter(terminal)
    event = _event()
    session_key, text = "agent:main:slack:group:chat:topic", "original answer"
    oid = dl.compute_obligation_id(session_key, event.message_id, text)
    timer = Mock()
    adapter.gateway_runner = SimpleNamespace(_schedule_flood_redelivery=timer)
    runner = object.__new__(GatewayRunner)
    runner._deliver_media_from_response = AsyncMock()
    if path in {"normal", "prior", "tts"}:
        if path == "prior":
            event._delivery_retry_suppressed_result = terminal
        attachment = tmp_path / "report.pdf"
        attachment.write_bytes(b"attachment")
        adapter._message_handler = AsyncMock(return_value=f"{text}\nMEDIA: {attachment}")
        adapter._start_typing_refresh = Mock(return_value=None)
        adapter._stop_typing_refresh = AsyncMock()
        adapter._wants_auto_tts = Mock(return_value=path == "tts")
        if path == "tts":
            audio = tmp_path / "generated.ogg"
            audio.write_bytes(b"generated audio")
            adapter._synthesize_auto_tts = AsyncMock(return_value=([str(audio)], None))
            adapter.play_tts = AsyncMock(return_value=terminal)
        adapter.on_processing_complete = AsyncMock()
        await adapter._process_message_background(event, session_key)
        adapter.on_processing_complete.assert_awaited_once_with(event, ProcessingOutcome.FAILURE)
        adapter.send_document.assert_not_awaited()
        assert adapter.send.await_count == (path == "normal")
    elif path.startswith("queued"):
        consumer = None
        metadata = {}
        if path == "queued_edit":
            consumer = SimpleNamespace(message_id="preview", _turn_split_delivery=False)
        elif path == "queued_stream":
            consumer = SimpleNamespace(retry_suppressed_result=terminal)
        elif path == "queued_state":
            metadata = {"_feishu_topic_delivery": {"terminal": terminal}}
        result = await runner._deliver_queued_first_response(
            text, event.source, adapter, metadata=metadata, stream_consumer=consumer,
            session_key=session_key, inbound_message_id=event.message_id)
        assert result is terminal and result.success is False
        runner._deliver_media_from_response.assert_not_awaited()
        assert adapter.send.await_count == (path == "queued")
    elif path == "atomic_prior":
        # Observe the row at the crash boundary before finalization: it must already
        # be terminal, never a pending/attempting obligation that a restart can revive.
        async def finalize(obligation_id, *_args):
            with dl._connect() as conn:
                assert conn.execute("SELECT state FROM delivery_obligations WHERE obligation_id=?",
                                    (obligation_id,)).fetchone()[0] == "abandoned"
        adapter._finalize_delivery_obligation = finalize
        result, _ = await adapter.send_final_ledgered(
            event, session_key, text, {}, reply_to=None, suppressed_result=terminal)
        assert result is terminal
        adapter.send.assert_not_awaited()
    else:
        dl.record_obligation(obligation_id=oid, session_key=session_key, platform="slack", chat_id="chat",
                             thread_id="topic", content=text)
        monkeypatch.setattr(dl, "_owner_alive", lambda *_args: False)
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        runner._obligation_adapter = AsyncMock(return_value=adapter)
        runner._arm_flood_timers_for_waiting_rows = AsyncMock()
        assert await runner._redeliver_claimed_obligations(claimed) == 0
        adapter.send.assert_awaited_once()
    with dl._connect() as conn:
        state, content, error = conn.execute(
            "SELECT state, content, last_error FROM delivery_obligations WHERE obligation_id=?", (oid,)
        ).fetchone()
    assert (state, content) == ("abandoned", text)
    assert "retry_suppressed" in error and terminal.error in error
    timer.assert_not_called()
    assert dl.pending_retries() == []
    assert dl.sweep_failed_for_runtime("slack", now=10**10) == []
    monkeypatch.setattr(dl, "_owner_alive", lambda *_args: False)
    assert dl.sweep_recoverable() == []
