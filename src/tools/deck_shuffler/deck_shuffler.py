"""
Deck Shuffler Module for Poker Application

This module provides a DeckShuffler class that simulates a standard 52-card deck
and provides methods for shuffling, drawing cards, and returning cards to the deck.
"""

import random
from typing import List


class Card:
    """Represents a single playing card."""

    SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

    def __init__(self, suit: str, rank: str):
        """
        Initialize a card with a suit and rank.

        Args:
            suit: The suit of the card (Hearts, Diamonds, Clubs, Spades)
            rank: The rank of the card (2-10, J, Q, K, A)
        """
        self.suit = suit
        self.rank = rank

    def __repr__(self) -> str:
        """Return a string representation of the card."""
        return f"{self.rank} of {self.suit}"

    def __eq__(self, other) -> bool:
        """Check if two cards are equal."""
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank


class DeckShuffler:
    """A deck shuffler that manages a standard 52-card deck."""

    def __init__(self):
        """Initialize a new deck with all 52 cards."""
        self.deck: List[Card] = []
        self.discarded: List[Card] = []
        self._initialize_deck()

    def _initialize_deck(self) -> None:
        """Create a standard 52-card deck (no jokers)."""
        self.deck = [
            Card(suit, rank)
            for suit in Card.SUITS
            for rank in Card.RANKS
        ]
        self.discarded = []

    def shuffle(self) -> None:
        """
        Randomize the order of the cards in the deck.

        This uses Python's random.shuffle() which implements the Fisher-Yates algorithm.
        """
        random.shuffle(self.deck)

    def get_card(self, count: int = 1) -> List[Card]:
        """
        Draw cards from the top of the deck.

        Args:
            count: Number of cards to draw (default: 1)

        Returns:
            A list of Card objects drawn from the deck

        Raises:
            ValueError: If count is less than 1 or if there aren't enough cards in the deck
        """
        if count < 1:
            raise ValueError("Count must be at least 1")

        if count > len(self.deck):
            raise ValueError(
                f"Cannot draw {count} cards. Only {len(self.deck)} cards remaining in deck."
            )

        # Draw cards from the top (end of the list)
        drawn_cards = []
        for _ in range(count):
            card = self.deck.pop()
            drawn_cards.append(card)
            self.discarded.append(card)

        return drawn_cards

    def return_all_cards(self) -> None:
        """
        Return all issued (discarded) cards to the bottom of the deck.

        The discarded cards are added to the bottom of the deck in the order
        they were discarded.
        """
        # Add discarded cards to the bottom (beginning) of the deck
        self.deck = self.discarded + self.deck
        self.discarded = []

    def cards_remaining(self) -> int:
        """
        Get the number of cards remaining in the deck.

        Returns:
            Number of cards still in the deck
        """
        return len(self.deck)

    def cards_discarded(self) -> int:
        """
        Get the number of cards that have been drawn from the deck.

        Returns:
            Number of cards currently discarded
        """
        return len(self.discarded)

    def reset(self) -> None:
        """
        Reset the deck to its initial state with all 52 cards.

        This returns all cards to the deck and reinitializes it.
        """
        self._initialize_deck()
