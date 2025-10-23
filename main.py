import gradio as gr
from gradio import ChatMessage
from agents import Agent, Runner, function_tool

from pydantic import BaseModel

from src.tools.deck_shuffler import DeckShuffler, Card
from typing import List

class Player(BaseModel):
    name: str
    balance: int
    current_bet: int
    dealer_button: bool = False

class Table(BaseModel):
    players: List[Player]
    pot: int

table = Table(
    players=[
        Player(name = "Player 1", balance = 50, current_bet = 0, dealer_button=True),
        Player(name = "Player 2", balance = 50, current_bet = 0),
        Player(name = "Player 3", balance = 50, current_bet = 0),
        Player(name = "Player 4", balance = 50, current_bet = 0),
    ],
    pot = 0
)

deck = DeckShuffler()
deck.shuffle()

# Dealer Tools

@function_tool
def get_cards_from_deck(count: int = 1) -> List[Card]:
    return deck.get_card(count)


@function_tool
def shuffle_deck() -> None:
    print("Shuffling the deck")
    return deck.shuffle()

# Player Tools

@function_tool
def bet(name: str, amount: int) -> str:
    """Place a bet for a player, deducting the amount from their balance"""
    print(f"Bet function called: {name} with {amount}")
    player = next((player for player in table.players if player.name == name), None)
    if player is None:
        return f"Player {name} not found"
    if player.balance < amount:
        return f"Insufficient balance. {name} has ${player.balance} but tried to bet ${amount}"
    player.balance -= amount
    return f"{name} bet ${amount}. Remaining balance: ${player.balance}"

@function_tool
def check(name: str) -> None:
    pass

@function_tool
def fold(name: str) -> None:
    pass

# Player and Dealer Tools

@function_tool
def get_table() -> Table:
    """Returns the state of the table, including the number of players, each of the player's chips, the size of the pot, and the cards visible on the table"""
    return table


# Agents

PLAYER_INSTRUCTIONS = """
  You are a player in a game of Texas Hold'em. You should bet intelligently based on the cards you are dealt.
  The dealer will ask you how you would like to play - e.g., to fold, call, or something different. Use the tools (bet, fold) to play the game.
  You have access to a table tool, which returns the state of the table - i.e., number of players, their chips, and the size of the pot.
  If you would like to check, use the check tool.
  If you would like to bet (call or raise during the round) use the bet tool. Pass your player name and the amount you would like to bet.
  If you would like to fold, use the fold tool.
  Do not ask the user for any instructions. Just play the game to the best of your ability.
  When you have completed your turn, hand back control to the dealer.
"""

player1_agent = Agent(
    name="Player 1",
    handoff_description="Player 1",
    instructions=PLAYER_INSTRUCTIONS,
    tools=[get_table, check, bet, fold],
)

player2_agent = player1_agent.clone(
    name="Player 2"
)

player3_agent = player1_agent.clone(
    name="Player 3"
)

player4_agent = player1_agent.clone(
    name="Player 4"
)

dealer = Agent(
    name="Dealer",
    handoff_description="The Dealer in the game",
    instructions="""
  You are a dealer, running a game of Texas Hold'em Poker.
  You have access to a deck shuffler machine, exposed as a tool. The machine can shuffle the deck for you (shuffle method). You can use the get_card method to retrieve cards from the top of the deck.
  You have access to a table tool, which returns the state of the table - i.e., number of players, their chips, and the size of the pot.
  Call the players (agents) via tools when it's their turn to play. Wait for each player to take their turn before moving on to the next player.
  Don't use any blinds for the game.
""",
    tools=[
        shuffle_deck,
        get_cards_from_deck,
        get_table,
        player1_agent.as_tool(tool_name="Player1", tool_description="Player 1 in your game"),
        player2_agent.as_tool(tool_name="Player2", tool_description="Player 2 in your game"),
        player3_agent.as_tool(tool_name="Player3", tool_description="Player 3 in your game"),
        player4_agent.as_tool(tool_name="Player4", tool_description="Player 4 in your game"),
    ],
    # handoffs=[player1_agent, player2_agent, player3_agent, player4_agent]
)

# player1_agent.handoffs = [dealer]
# player2_agent.handoffs = [dealer]
# player3_agent.handoffs = [dealer]
# player4_agent.handoffs = [dealer]


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
                            content=f"{active_agent} calls tool {tool_name} with arguments {tool_args}",
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
    examples=["Can you draw two cards from the deck?", "Can you shuffle the deck?", "Play a round of poker"],
    submit_btn=True,
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    type="messages",
    save_history=False,
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
