"""Hand-built dictionaries used by the rule-based emotion scorer.

Weights are on a rough 0.5-3.0 scale: 1.0 is an ordinary emotion cue, 2.0+ is a
strong, unambiguous cue (e.g. 😡 for anger).
"""

EMOTIONS = ("joy", "sadness", "anger", "fear", "surprise", "love", "neutral")


def _group(emotion: str, weight: float, items: str) -> dict[str, tuple[str, float]]:
    return {item: (emotion, weight) for item in items.split()}


# --- Emoji ------------------------------------------------------------------
EMOJI_EMOTIONS: dict[str, tuple[str, float]] = {
    **_group("joy", 2.0, "😂 🤣 😆 😁 😄 😃 😀 😹 🥳 🎉 🎊 🤩"),
    **_group("joy", 1.5, "😊 🙂 😺 😸 😌 😎 🤗 👍 👏 🙌 ✌️ 💃 🕺 ✨ 🌞 😜 😝 😛 🤪 😋 🔥 💯 🙏"),
    **_group("sadness", 2.0, "😭 😢 😿 💔 🥺 😞 😔"),
    **_group("sadness", 1.5, "😟 🙁 ☹️ 😥 😪 😓 😩 😫 🥀 🌧️ 😶 🫠 🥲"),
    **_group("anger", 2.5, "😡 🤬 😠 👿 💢"),
    **_group("anger", 1.5, "😤 👊 🖕 😾 🙄 😒 🤨 😑"),
    **_group("fear", 2.0, "😨 😰 😱 🙀 😧 😖"),
    **_group("fear", 1.5, "😬 😳 🫣 👻 💀 ☠️ 🥶 😦"),
    **_group("surprise", 2.0, "😲 😮 😯 🤯 😵 😵‍💫 ‼️ ⁉️"),
    **_group("surprise", 1.5, "👀 😶‍🌫️ 🫢 ❗ ❓"),
    **_group("love", 2.5, "😍 🥰 😘 ❤️ 💕 💖 💗 💓 💞 💘 💝 😻"),
    **_group("love", 2.0, "♥️ 💜 💙 💚 🧡 💛 🤍 🖤 🤎 💑 💏 👩‍❤️‍👨 😚 😙 💋 🌹 🫶"),
    **_group("neutral", 1.0, "😐 🤔 🙃 🤷 🤷‍♂️ 🤷‍♀️ 📌 📝 ✅"),
}


# --- Chat slang (matched on the raw token, before normalisation) -------------
SLANG_EMOTIONS: dict[str, tuple[str, float]] = {
    **_group("joy", 1.5, "lol lool lolz lmao lmfao rofl haha hahaha hehe hihi xd yay woohoo yayy lit dope"),
    **_group("joy", 1.5, "mast jhakaas zabardast dhamaal sahi badhiya bindaas kadak bombat sakkath semma gethu"),
    **_group("sadness", 1.5, "fml sigh welp meh ugh_sad bleh rip"),
    **_group("anger", 1.5, "ugh wtf smh bakwas bekar bakwaas faltu chee chi bc bkl"),
    **_group("anger", 1.0, "pathetic cringe"),
    **_group("fear", 1.0, "yikes eek"),
    **_group("surprise", 1.5, "omg omfg wow woah whoa damn arey arrey arre aiyo ayyo aiyyo ayyayyo"),
    **_group("love", 1.5, "ily ilu xoxo luv babe jaan jaanu shona"),
    **_group("neutral", 0.5, "ok okay k kk hmm hmmm idk nvm"),
}


# --- Words: English + Romanized Hindi / Kannada / Tamil / Bengali -------------
WORD_EMOTIONS: dict[str, tuple[str, float]] = {
    # joy
    **_group("joy", 1.5, "happy happiness glad joy joyful excited exciting awesome amazing great fantastic "
                         "wonderful delighted fun enjoy enjoyed enjoying cheerful thrilled yay celebrate "
                         "celebrating proud blessed grateful best superb excellent nice good laughing laugh"),
    **_group("joy", 1.5, "khush khushi khushi anand maza mazaa mazza masti badiya badhiya accha achha "
                         "shandaar kamaal"),                                        # Hindi
    **_group("joy", 1.5, "santosha santosh khushi chennagide sakkath majaa"),     # Kannada
    **_group("joy", 1.5, "santhosham magizhchi jollyaa nalla super"),             # Tamil
    **_group("joy", 1.5, "anondo khushi darun bhalo"),                            # Bengali
    # sadness
    **_group("sadness", 1.5, "sad sadness unhappy depressed depressing cry crying cried tears lonely "
                             "alone miss missing hurt hurting heartbroken broken upset sorry gloomy "
                             "miserable lost tired disappointed disappointing regret"),
    **_group("sadness", 1.5, "dukhi dukh udaas udas rona ro rula rulaya akela takleef afsos dard"),  # Hindi
    **_group("sadness", 1.5, "dukha besara bejaar"),                              # Kannada
    **_group("sadness", 1.5, "sogam kashtam varutham azhugai"),                   # Tamil
    **_group("sadness", 1.5, "dukkho kanna mon_kharap kharap"),                   # Bengali
    # anger
    **_group("anger", 1.5, "angry anger mad furious annoyed annoying irritated irritating hate hated "
                           "rage worst stupid idiot nonsense disgusting frustrated frustrating terrible"),
    **_group("anger", 2.0, "gussa ghussa naraz naraaz chidh pagal bewakoof ullu"),  # Hindi
    **_group("anger", 2.0, "kopa sitti"),                                         # Kannada
    **_group("anger", 2.0, "kobam kovam erichal"),                                # Tamil
    **_group("anger", 2.0, "raag rag bhirokto"),                                  # Bengali
    # fear
    **_group("fear", 1.5, "scared afraid fear fearful terrified nervous anxious anxiety worried worry "
                          "panic panicking frightened horror horrible creepy tension tense stressed"),
    **_group("fear", 1.5, "darr dar darta darti darr_lag ghabrahat ghabra tension"),  # Hindi
    **_group("fear", 1.5, "bhaya hedarike"),                                      # Kannada
    **_group("fear", 1.5, "bayam bhayam"),                                        # Tamil
    **_group("fear", 1.5, "bhoy bhoi"),                                           # Bengali
    # surprise
    **_group("surprise", 1.5, "surprised surprise shocked shocking unexpected unbelievable wow amazed "
                              "astonished suddenly"),
    **_group("surprise", 1.5, "sach sachchi achanak hairaan hairan"),             # Hindi
    **_group("surprise", 1.5, "nijavaagi ascharya"),                              # Kannada
    **_group("surprise", 1.5, "aacharyam nijamaa"),                               # Tamil
    **_group("surprise", 1.5, "obak shotti"),                                     # Bengali
    # love
    **_group("love", 2.0, "love loving loved adore lovely sweetheart darling crush romantic"),
    **_group("love", 2.0, "pyaar pyar mohabbat ishq dil"),                       # Hindi
    **_group("love", 2.0, "preethi ishta"),                                       # Kannada
    **_group("love", 2.0, "kadhal anbu"),                                         # Tamil
    **_group("love", 2.0, "bhalobasha bhalobashi"),                               # Bengali
}


# Words that flip the emotion of a nearby cue ("not happy", "khush nahi").
NEGATORS = frozenset(
    "not no never dont don't didnt didn't isnt isn't wasnt wasn't cant can't wont won't "
    "nahi nahin nhi mat na naa illa illai alla allai noy na".split()
)

# Words that make a nearby emotion stronger.
INTENSIFIERS: dict[str, float] = {
    **{word: 1.5 for word in "very so really too extremely super totally".split()},
    **{word: 1.5 for word in "bahut bohot bht bhot ekdum bilkul kaafi".split()},  # Hindi
    **{word: 1.5 for word in "tumba thumba".split()},                            # Kannada
    **{word: 1.5 for word in "romba rombha semma".split()},                      # Tamil
    **{word: 1.5 for word in "khub onek".split()},                               # Bengali
}

EMOTION_EMOJI = {
    "joy": "😄", "sadness": "😢", "anger": "😡", "fear": "😨",
    "surprise": "😲", "love": "❤️", "neutral": "😐",
}
