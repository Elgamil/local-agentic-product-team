"""Read human text at a gate.  Deterministic on purpose.

An approval must mean the person approved, so no model interprets it.  Only a short
list of plain affirmations counts.  "Yes, but make it blue" is not an approval: it
carries a request, so it falls through to being treated as one.

Product names follow the same principle from the other direction.  Free text is not
a name just because it failed to match something: see :func:`read_name_reply`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    QUESTION = "question"
    PAUSE = "pause"
    EXIT = "exit"
    STATUS = "status"
    HELP = "help"
    TEAM = "team"
    PRODUCTS = "products"
    RETRY = "retry"
    ACCEPT = "accept"
    DONE = "done"
    REMOVE_STEP = "remove_step"
    MOVE_STEP = "move_step"
    EMPTY = "empty"
    AUTONOMY = "autonomy"
    VERIFY = "verify"
    REPORT = "report"
    STOP = "stop"


@dataclass(frozen=True)
class ParsedInput:
    intent: Intent
    text: str = ""
    number: int | None = None
    position: int | None = None


_AFFIRMATIONS = frozenset(
    {
        "y", "yes", "yep", "yeah", "yes please", "ok", "okay", "sure",
        "approve", "approved", "i approve", "i approve it", "approve it",
        "looks good", "looks good to me", "lgtm", "sounds good", "go ahead",
        "proceed", "continue", "confirm", "confirmed", "accept the name",
        "keep it", "keep this name", "keep the name", "go for it", "lets go",
    }
)
_NEGATIONS = frozenset({"n", "no", "nope", "reject", "rejected", "i reject", "reject it"})
_PAUSE_WORDS = frozenset({"pause", "pause this", "pause the product", "not now", "later"})
_EXIT_WORDS = frozenset({"exit", "quit", "bye", "goodbye", "stop"})

MAX_NAME_WORDS = 6
# Words that say something about the suggested name, without being a name.
_NAME_AGREEMENT = frozenset(
    {
        "y", "yes", "yep", "yup", "yeah", "ok", "okay", "sure", "fine", "good", "great",
        "perfect", "nice", "cool", "awesome", "excellent", "happy", "keep", "use", "accept",
        "accepted", "agree", "agreed", "approve", "approved", "confirm", "confirmed",
        "like", "love", "prefer", "looks", "sounds", "lgtm", "works", "go", "ahead",
        "proceed", "continue", "stick",
    }
)
_NAME_NEGATION = frozenset({"n", "no", "nope", "nah", "not", "dont", "never", "wont", "cant"})
_WANTS_ANOTHER_NAME = frozenset(
    {"different", "another", "other", "else", "change", "rename", "new", "alternative",
     "something", "anything"}
)
# Words that carry no meaning of their own in a reply about the name.
_NAME_FILLER = frozenset(
    {
        "a", "an", "and", "am", "as", "be", "by", "can", "for", "i", "id", "im", "is", "it",
        "its", "just", "lets", "me", "my", "name", "of", "on", "one", "please", "proposed",
        "really", "so", "suggested", "suggestion", "thanks", "thank", "that", "thats", "the",
        "then", "this", "to", "very", "we", "will", "with", "would", "want", "you", "your",
    }
)
_FIRST_PERSON = frozenset({"i", "id", "im", "ive", "me", "my", "we"})
_UNSURE = re.compile(
    r"\b(?:not sure|unsure|maybe|perhaps|dunno|idk|dont know|no idea|whatever|hmm+|i guess)\b"
)
_NAMING_PHRASE = re.compile(
    r"\b(?:call\s+(?:it|this|the\s+product)|name\s+(?:it|this|the\s+product)|"
    r"rename\s+(?:it\s+|this\s+)?to|change\s+(?:it|this|the\s+name)\s+to|"
    r"the\s+name\s+(?:should\s+be|will\s+be|is)|use\s+the\s+name|"
    r"i\s+prefer|i'?d\s+prefer|i\s+would\s+prefer|how\s+about|what\s+about|"
    r"let'?s\s+go\s+with|go\s+with)\s+(?P<name>.+)$",
    re.IGNORECASE,
)
_USE_NAME = re.compile(r"^(?:please\s+)?use\s+(?P<name>.+)$", re.IGNORECASE)
_QUOTED_NAME = re.compile(r"(?<!\w)[\"'“‘]([^\"“”‘’]+?)[\"'”’](?!\w)")
_NAME_ENDING = re.compile(r"\s+(?:please|instead|thanks|thank\s+you)$", re.IGNORECASE)
_QUESTION_OPENERS = (
    "what", "why", "how", "when", "where", "who", "which",
    "can", "could", "would", "will", "should", "is", "are", "do", "does",
)
_COMMANDS: dict[str, Intent] = {
    "approve": Intent.APPROVE,
    "reject": Intent.REJECT,
    "change": Intent.MODIFY,
    "modify": Intent.MODIFY,
    "ask": Intent.QUESTION,
    "pause": Intent.PAUSE,
    "exit": Intent.EXIT,
    "quit": Intent.EXIT,
    "status": Intent.STATUS,
    "help": Intent.HELP,
    "team": Intent.TEAM,
    "products": Intent.PRODUCTS,
    "retry": Intent.RETRY,
    "accept": Intent.ACCEPT,
    "done": Intent.DONE,
    "autonomy": Intent.AUTONOMY,
    "verify": Intent.VERIFY,
    "report": Intent.REPORT,
    "stop": Intent.STOP,
}
_REMOVE = re.compile(r"^(?:please\s+)?(?:remove|delete|drop)\s+step\s+(\d+)\.?$", re.IGNORECASE)
_MOVE = re.compile(
    r"^(?:please\s+)?move\s+step\s+(\d+)\s+to\s+(?:step\s+|position\s+)?(\d+)\.?$", re.IGNORECASE
)


def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def is_affirmation(text: str) -> bool:
    """True only for a plain yes.  "ok?" asks something, so it never approves anything."""
    return normalize(text) in _AFFIRMATIONS and not text.strip().endswith("?")


def looks_like_question(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    first = normalize(stripped).split(" ", 1)[0] if stripped else ""
    return first in _QUESTION_OPENERS


def parse_input(text: str) -> ParsedInput:
    """Classify one line of terminal input, without any model."""
    raw = text.strip()
    if not raw:
        return ParsedInput(Intent.EMPTY)

    if raw.startswith("/"):
        word, _, rest = raw[1:].partition(" ")
        intent = _COMMANDS.get(word.lower())
        if word.lower() in ("remove", "move"):
            return parse_input(f"{word} step {rest.strip()}")
        if intent is None:
            return ParsedInput(Intent.HELP)
        return ParsedInput(intent, rest.strip())

    normalized = normalize(raw)
    if is_affirmation(raw):
        return ParsedInput(Intent.APPROVE, raw)
    if normalized in _NEGATIONS and not raw.endswith("?"):
        return ParsedInput(Intent.REJECT, raw)
    if normalized in _PAUSE_WORDS:
        return ParsedInput(Intent.PAUSE, raw)
    if normalized in _EXIT_WORDS:
        return ParsedInput(Intent.EXIT, raw)

    removal = _REMOVE.match(raw)
    if removal:
        return ParsedInput(Intent.REMOVE_STEP, raw, number=int(removal.group(1)))
    move = _MOVE.match(raw)
    if move:
        return ParsedInput(
            Intent.MOVE_STEP, raw, number=int(move.group(1)), position=int(move.group(2))
        )

    if looks_like_question(raw):
        return ParsedInput(Intent.QUESTION, raw)
    return ParsedInput(Intent.MODIFY, raw)


def extract_name(text: str) -> str:
    """Take the name out of phrases like "call it 'Milk Market'"."""
    cleaned = re.sub(
        r"^(?:please\s+)?(?:let'?s\s+)?(?:call\s+it|name\s+it|use|i\s+prefer|i'?d\s+prefer|"
        r"how\s+about|the\s+name\s+is|rename\s+it\s+to|change\s+it\s+to)\s+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip().strip("\"'“”‘’.?!,").strip()


class NameAnswer(str, Enum):
    ACCEPT = "accept"
    DECLINE = "decline"
    NEW_NAME = "new_name"
    POSSIBLE_NAME = "possible_name"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class NameReply:
    answer: NameAnswer
    name: str = ""


def _compact(text: str) -> str:
    return normalize(text).replace(" ", "")


def _stated_name(text: str) -> str:
    """Return a name the person explicitly stated, or an empty string."""
    quoted = _QUOTED_NAME.search(text)
    if quoted:
        return quoted.group(1).strip()
    match = _NAMING_PHRASE.search(text) or _USE_NAME.match(text)
    if match is None:
        return ""
    return _NAME_ENDING.sub("", match.group("name").strip().rstrip(".?!,")).strip("\"'“”‘’ ")


def _names_nothing(candidate: str) -> bool:
    """True for words after "use" or "go with" that are not a name, such as "this one"."""
    words = set(normalize(candidate).split())
    meaningless = _NAME_FILLER | _NAME_AGREEMENT | _NAME_NEGATION | _WANTS_ANOTHER_NAME
    return words <= meaningless or bool(
        words & {"something", "anything", "different", "another", "else"}
    )


def _name_command(raw: str) -> NameReply:
    parsed = parse_input(raw)
    if parsed.intent == Intent.APPROVE:
        return NameReply(NameAnswer.ACCEPT)
    if parsed.intent == Intent.REJECT:
        return NameReply(NameAnswer.DECLINE)
    if parsed.intent == Intent.MODIFY:
        if parsed.text:
            return NameReply(NameAnswer.NEW_NAME, extract_name(parsed.text))
        return NameReply(NameAnswer.DECLINE)
    return NameReply(NameAnswer.UNCLEAR)


def read_name_reply(text: str, proposed: str) -> NameReply:
    """Interpret an answer to "do you want this product name?".

    The rule is about what the reply *contains*, not a list of accepted sentences.
    Remove the words that carry no meaning.  If only agreement remains, the name is
    accepted, which covers "ok for the name" and every phrase like it.

    A reply only changes the name when it clearly states one: "call it Milk Market",
    or a quoted name.  Text that merely might be a name comes back as POSSIBLE_NAME
    for the session to confirm, and anything else as UNCLEAR.  Neither renames the
    product, so a phrase this function does not recognise can never become a name.
    """
    raw = text.strip()
    if raw.startswith("/"):
        return _name_command(raw)

    normalized = normalize(raw)
    is_question = raw.endswith("?")
    if not is_question:
        if is_affirmation(raw) or _compact(raw) == _compact(proposed):
            return NameReply(NameAnswer.ACCEPT)
        if normalized in _NEGATIONS:
            return NameReply(NameAnswer.DECLINE)

    stated = _stated_name(raw)
    if stated and _compact(stated) == _compact(proposed):
        return NameReply(NameAnswer.ACCEPT)
    if stated and not _names_nothing(stated):
        return NameReply(NameAnswer.NEW_NAME, stated)

    if _UNSURE.search(normalized) or looks_like_question(raw):
        return NameReply(NameAnswer.UNCLEAR)

    words = set(normalized.split())
    proposed_words = set(normalize(proposed).split()) | {_compact(proposed)}
    meaningful = words - _NAME_FILLER - proposed_words
    opinion = _NAME_AGREEMENT | _NAME_NEGATION

    if meaningful & _WANTS_ANOTHER_NAME and meaningful <= opinion | _WANTS_ANOTHER_NAME:
        return NameReply(NameAnswer.DECLINE)
    if meaningful <= opinion:
        if words & _NAME_NEGATION:
            return NameReply(NameAnswer.DECLINE)
        if meaningful & _NAME_AGREEMENT or words & proposed_words:
            return NameReply(NameAnswer.ACCEPT)
        return NameReply(NameAnswer.UNCLEAR)

    is_sentence = bool(words & (opinion | _FIRST_PERSON)) or len(raw.split()) > MAX_NAME_WORDS
    if is_sentence:
        return NameReply(NameAnswer.UNCLEAR)
    return NameReply(NameAnswer.POSSIBLE_NAME, extract_name(raw))
