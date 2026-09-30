"""MEDIA replies keep the event's Feishu topic anchor through real delivery."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import _thread_metadata_for_source
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from plugins.platforms.feishu.adapter import FeishuAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("post_stream", [False, True], ids=["non_streaming", "post_stream"])
@pytest.mark.parametrize("parent_id", [None, "om_parent"], ids=["topic_seed", "topic_reply"])
@pytest.mark.parametrize(
    "filename,directive,message_type",
    [
        ("report.pdf", "", "file"),
        ("chart.png", "", "image"),
        ("clip.mp4", "", "media"),
        ("chart.png", "[[as_document]]\n", "file"),
    ],
)
async def test_media_delivery_replies_in_originating_topic(
    tmp_path, monkeypatch, post_stream, parent_id, filename, directive, message_type,
):
    monkeypatch.setenv("FEISHU_REACTIONS", "false")
    media = tmp_path / filename
    media.write_bytes(b"attachment")
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
    source = SessionSource(
        platform=Platform.FEISHU, chat_id="oc_chat", chat_type="group",
        thread_id="omt_topic", message_id="om_root",
    )
    event = MessageEvent(
        text="Send the attachment", message_type=MessageType.TEXT, source=source,
        message_id="om_current", reply_to_message_id=parent_id,
    )
    upload = Mock(return_value=SimpleNamespace(
        success=lambda: True, data=SimpleNamespace(file_key="file_key", image_key="image_key"),
    ))
    sent = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_sent"))
    messages = SimpleNamespace(reply=Mock(return_value=sent), create=Mock(return_value=sent), list=Mock())
    adapter = FeishuAdapter(PlatformConfig(enabled=True, typing_indicator=False))
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(
        file=SimpleNamespace(create=upload), image=SimpleNamespace(create=upload), message=messages,
    )))
    response = f"{directive}MEDIA:{media}"
    try:
        if post_stream:
            runner = object.__new__(GatewayRunner)
            await runner._deliver_media_from_response(response, event, adapter)
        else:
            adapter.set_message_handler(AsyncMock(return_value=response))
            await adapter._process_message_background(event, build_session_key(source))
    finally:
        adapter._shutdown_sdk_executor()

    upload.assert_called_once()
    messages.create.assert_not_called()
    messages.list.assert_not_called()
    messages.reply.assert_called_once()
    request = messages.reply.call_args.args[0]
    assert request.message_id == (event.reply_to_message_id or event.message_id)
    assert request.request_body.reply_in_thread is True
    assert request.request_body.msg_type == message_type


@pytest.mark.parametrize(
    "platform,thread_id,source_anchor,explicit_anchor",
    [
        (Platform.FEISHU, "omt_topic", "om_root", None),
        (Platform.FEISHU, "omt_topic", "om_root", "om_parent"),
        (Platform.FEISHU, "omt_topic", None, "om_parent"),
        (Platform.FEISHU, "omt_topic", None, None),
        (Platform.FEISHU, None, "om_root", "om_parent"),
        (Platform.FEISHU, "", "om_root", "om_parent"),
        (Platform.TELEGRAM, "42", "om_root", "om_parent"),
        (Platform.SLACK, "thread_ts", "om_root", "om_parent"),
        (Platform.DISCORD, "thread_channel", "om_root", "om_parent"),
    ],
)
def test_metadata_builders_preserve_only_available_feishu_topic_anchors(
    platform, thread_id, source_anchor, explicit_anchor,
):
    source = SessionSource(
        platform=platform, chat_id="chat", chat_type="group",
        thread_id=thread_id, message_id=source_anchor,
    )
    runner = object.__new__(GatewayRunner)
    for builder in (_thread_metadata_for_source, runner._thread_metadata_for_source):
        metadata = builder(source, explicit_anchor) or {}
        anchor = explicit_anchor or source_anchor
        if platform == Platform.FEISHU and thread_id and anchor:
            assert metadata["reply_to_message_id"] == anchor
            assert metadata["thread_id"] == source.thread_id
        else:
            assert "reply_to_message_id" not in metadata
