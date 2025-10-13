import gradio as gr
from gradio import ChatMessage
from agents import Agent, Runner, function_tool

from src.tools.deck_shuffler import DeckShuffler, Card
from typing import List

deck = DeckShuffler()
deck.shuffle()

@function_tool
def get_cards_from_deck(count: int = 1) -> List[Card]:
    return deck.get_card(count)

@function_tool
def shuffle_deck() -> None:
    return deck.shuffle()

dealer = Agent(
  name = "Dealer",
  instructions="""
  You are a dealer, running a game of Texas Hold 'em Poker.
  You have access to a deck shuffler machine, exposed as a tool. The machine can shuffle the deck for you (shuffle method). You can use the get_card method to retrieve cards from the top of the deck.
""",
  tools = [shuffle_deck, get_cards_from_deck]
)

async def poker_game(user_msg: str, history: list):
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
    messages.append({"role": "user", "content": user_msg})
    responses = []
    reply_created = False
    active_agent = None

    result = Runner.run_streamed(dealer, messages)
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if event.data.type == "response.output_text.delta":
                if not reply_created:
                    responses.append(ChatMessage(role="assistant", content=""))
                    reply_created = True
                responses[-1].content += event.data.delta
        elif event.type == "agent_updated_stream_event":
            active_agent = event.new_agent.name
            responses.append(
                ChatMessage(
                    content=event.new_agent.name,
                    metadata={"title": "Agent Now Running", "id": active_agent},
                )
            )
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                if event.item.raw_item.type == "file_search_call":
                    responses.append(
                        ChatMessage(
                            content=f"Query used: {event.item.raw_item.queries}",
                            metadata={
                                "title": "File Search Completed",
                                "parent_id": active_agent,
                            },
                        )
                    )
                else:
                    tool_name = getattr(event.item.raw_item, "name", "unknown_tool")
                    tool_args = getattr(event.item.raw_item, "arguments", {})
                    responses.append(
                        ChatMessage(
                            content=f"Calling tool {tool_name} with arguments {tool_args}",
                            metadata={"title": "Tool Call", "parent_id": active_agent},
                        )
                    )
            if event.item.type == "tool_call_output_item":
                responses.append(
                    ChatMessage(
                        content=f"Tool output: '{event.item.raw_item['output']}'",
                        metadata={"title": "Tool Output", "parent_id": active_agent},
                    )
                )
            if event.item.type == "handoff_call_item":
                responses.append(
                    ChatMessage(
                        content=f"Name: {event.item.raw_item.name}",
                        metadata={
                            "title": "Handing Off Request",
                            "parent_id": active_agent,
                        },
                    )
                )
        yield responses

demo = gr.ChatInterface(
    poker_game,
    title="Texas Hold'em",
    theme=gr.themes.Soft(
        primary_hue="green", secondary_hue="slate", font=[gr.themes.GoogleFont("Inter")]
    ),
    examples=[
        "Can you draw two cards from the deck?",
        "Can you shuffle the deck?"
    ],
    submit_btn=True,
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    type="messages",
    save_history=False,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)