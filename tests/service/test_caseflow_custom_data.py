from langchain_core.messages import AIMessage

from service.utils import langchain_to_chat_message


def test_ai_message_custom_data_is_preserved_for_caseflow_workbench():
    message = AIMessage(
        content="CaseFlow 处理结果",
        additional_kwargs={
            "custom_data": {
                "intent": "咨询类",
                "priority": "low",
                "needs_human_approval": False,
            }
        },
    )

    chat_message = langchain_to_chat_message(message)

    assert chat_message.type == "ai"
    assert chat_message.content == "CaseFlow 处理结果"
    assert chat_message.custom_data["intent"] == "咨询类"
    assert chat_message.custom_data["priority"] == "low"
    assert chat_message.custom_data["needs_human_approval"] is False
