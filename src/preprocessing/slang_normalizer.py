"""Phase 0: expand chat slang and separate emoji before spelling correction.

Only two kinds of change are made:

* known slang / short forms are replaced with their full form, and
* emoji are removed from the text (and returned separately for emotion analysis).

Every other character, including punctuation, case and spacing between words, is
left untouched, so text without slang or emoji reaches Phases 1-3 unchanged.
"""

import re
from typing import Any

import emoji

from src.correction.pipeline import LEXICON, VARIANTS


# Chat shorthand (mostly English) and Romanized Indic short forms.
# Keys are lowercase. Values may contain several words.
CHAT_SLANG = {
    "u": "you", "ur": "your", "r": "are", "n": "and", "y": "why",
    "pls": "please", "plz": "please", "plzz": "please",
    "thx": "thanks", "thnx": "thanks", "tnx": "thanks", "ty": "thank you",
    "gr8": "great", "gud": "good", "gd": "good", "b4": "before", "2day": "today",
    "2moro": "tomorrow", "tmrw": "tomorrow", "tmr": "tomorrow", "2nite": "tonight",
    "bcoz": "because", "bcz": "because", "coz": "because", "cuz": "because",
    "abt": "about", "wid": "with", "wit": "with", "w/o": "without", "nd": "and",
    "msg": "message", "pic": "picture", "ppl": "people", "frnd": "friend",
    "frnds": "friends", "bday": "birthday", "luv": "love", "wanna": "want to",
    "gonna": "going to", "gotta": "got to", "dunno": "do not know", "kinda": "kind of",
    "sorta": "sort of", "ya": "yes", "yep": "yes", "yup": "yes", "nope": "no",
    "k": "ok", "kk": "ok", "okie": "ok", "okk": "ok", "hru": "how are you",
    "wbu": "what about you", "idk": "i do not know", "idc": "i do not care",
    "imo": "in my opinion", "tbh": "to be honest", "btw": "by the way",
    "brb": "be right back", "omw": "on my way", "asap": "as soon as possible",
    "lol": "laughing out loud", "lmao": "laughing a lot", "rofl": "rolling on the floor laughing",
    "omg": "oh my god", "wtf": "what the heck", "smh": "shaking my head",
    "fml": "my life is a mess", "ily": "i love you", "ilu": "i love you", "jk": "just kidding",
    "nvm": "never mind", "ikr": "i know right", "ofc": "of course", "srsly": "seriously",
    "rly": "really", "v": "very", "vry": "very", "sry": "sorry", "soz": "sorry",
    "bro": "brother", "sis": "sister", "fam": "family", "gn": "good night",
    "gm": "good morning", "hbd": "happy birthday", "np": "no problem",
    "w8": "wait", "l8r": "later", "ttyl": "talk to you later", "xoxo": "hugs and kisses",
}

INDIC_SLANG = {
    "bht": "bahut", "bhot": "bahut", "bhut": "bahut", "bohot": "bahut", "boht": "bahut",
    "nhi": "nahi", "nai": "nahi", "nahin": "nahi", "nhin": "nahi",
    "h": "hai", "hn": "haan",
    "kya": "kya", "kyu": "kyun", "kyon": "kyun", "q": "kyun",
    "mje": "mujhe", "mujhko": "mujhe", "mjhe": "mujhe", "tje": "tujhe",
    "acha": "accha", "achha": "accha", "acchha": "accha",
    "thik": "theek", "thk": "theek",
    "pta": "pata", "kb": "kab", "kahn": "kahan",
    "yr": "yaar", "yar": "yaar", "bhai": "bhai", "bhaiya": "bhaiya",
    "shukriya": "shukriya", "dhanyavad": "dhanyavaad", "mst": "mast",
    "sb": "sab", "kch": "kuch", "kuchh": "kuch", "abi": "abhi", "abhe": "abhi",
    "tm": "tum", "hm": "hum",
    "gya": "gaya", "gyi": "gayi", "rha": "raha", "rhi": "rahi", "rhe": "rahe",
}

SLANG_MAP: dict[str, tuple[str, str]] = {
    **{key: (value, "chat") for key, value in CHAT_SLANG.items()},
    **{key: (value, "indic") for key, value in INDIC_SLANG.items()},
}
# Identity entries carry no information and must never rewrite a correction-lexicon word.
_PROTECTED = {word.casefold() for word in LEXICON} | set(VARIANTS)
SLANG_MAP = {
    key: value
    for key, value in SLANG_MAP.items()
    if key != value[0] and key not in _PROTECTED
}

_EDGE_PUNCTUATION = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)
_LONG_REPEAT = re.compile(r"(.)\1{2,}")


def _match_case(template: str, replacement: str) -> str:
    if len(template) > 1 and template.isupper():
        return replacement.upper()
    if template[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _lookup(core: str) -> tuple[str, str] | None:
    """Find ``core`` in the slang map, also trying it with letter stretching removed."""
    key = core.casefold()
    if key in _PROTECTED:
        return None
    if key in SLANG_MAP:
        return SLANG_MAP[key]
    # "plzzz" -> "plz" / "pls", "bhooot" -> "bhot"
    for squashed in (_LONG_REPEAT.sub(r"\1\1", key), _LONG_REPEAT.sub(r"\1", key)):
        # Only squash towards slang of 2+ letters: "nnn" is not a stretched "n" (= "and").
        if len(squashed) >= 2 and squashed != key and squashed not in _PROTECTED and squashed in SLANG_MAP:
            return SLANG_MAP[squashed]
    return None


class SlangNormalizer:
    """Expand slang and pull emoji out of text before it is corrected."""

    def normalize(self, text: str) -> dict[str, Any]:
        replacements: list[dict[str, Any]] = []
        found_emoji: list[dict[str, Any]] = []
        output_tokens: list[str] = []

        for position, token in enumerate(text.split()):
            if emoji.emoji_count(token):
                for item in emoji.emoji_list(token):
                    found_emoji.append(
                        {
                            "emoji": item["emoji"],
                            "name": emoji.demojize(item["emoji"], delimiters=("", "")).replace("_", " "),
                            "position": position,
                        }
                    )
                token = emoji.replace_emoji(token, replace="")
                # Variation selectors / zero-width joiners left behind by emoji sequences.
                token = re.sub("[‍︎️]", "", token)
                if not token:
                    continue

            prefix, core, suffix = _EDGE_PUNCTUATION.match(token).groups()
            match = _lookup(core) if core else None
            if match is None:
                output_tokens.append(token)
                continue

            replacement, kind = match
            new_core = _match_case(core, replacement)
            output_tokens.append(f"{prefix}{new_core}{suffix}")
            replacements.append(
                {
                    "original": core,
                    "replacement": new_core,
                    "kind": kind,
                    "position": position,
                }
            )

        normalized = " ".join(output_tokens)
        # Keep the input exactly as typed when nothing was changed.
        if not replacements and not found_emoji:
            normalized = text

        return {
            "input": text,
            "normalized": normalized,
            "replacements": replacements,
            "emoji": found_emoji,
            "changed": bool(replacements or found_emoji),
        }
