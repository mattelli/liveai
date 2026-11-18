import re
import string

RULES = [
    ("igh", "aɪ"), ("tch", "tʃ"), ("ch", "tʃ"),
    ("sh", "ʃ"), ("ph", "f"), ("ck", "k"),
    ("ng", "ŋ"), ("th", "θ"), ("oo", "uː"),
    ("ee", "iː"), ("ai", "eɪ"), ("ay", "eɪ"),
    ("ow", "aʊ")
]

SINGLE = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ", "f": "f", "g": "g",
    "h": "h", "i": "ɪ", "j": "dʒ", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "ɒ", "p": "p", "q": "k", "r": "ɹ", "s": "s", "t": "t", "u": "ʌ",
    "v": "v", "w": "w", "x": "ks", "y": "j", "z": "z"
}

EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F]"  # Emoticons (faces)
)

def tokenize_phonemes(word:str):
    w = word.lower()
    phonemes: list[str] = []
    i = 0
    while i < len(w):
        matched = False
        for g, p in RULES:
            if w.startswith(g, i):
                phonemes.append(p)
                i += len(g)
                matched = True
                break
        if matched:
            continue
        c = w[i]
        if c in SINGLE:
            phonemes.append(SINGLE[c])
        i += 1
    return phonemes


def tokenize_sentence(sentence):

    def split_words_emojis(cleaned):
        tokens = []
        i = 0
        while i < len(cleaned):
            ch = cleaned[i]
            if EMOJI_RE.match(ch):
                tokens.append(ch)
                i += 1
                continue
            if ch.isalnum():
                start = i
                while i < len(cleaned) and cleaned[i].isalnum():
                    i += 1
                tokens.append(cleaned[start:i])
                continue
            i += 1  # skip spaces or unknown chars
        return tokens
 
    def convert_words_to_phonemes_keep_emojis_as_is(tokens):
        result = []
        for tok in tokens:
            if EMOJI_RE.match(tok):
                result.append(tok)
            else:
                result.extend(tokenize_phonemes(tok))
        return result

    cleaned = "".join(ch for ch in sentence if ch not in string.punctuation)
    tokens = split_words_emojis(cleaned)
    return convert_words_to_phonemes_keep_emojis_as_is(tokens)


# -----------------------------
#  EXAMPLE
# -----------------------------
sentence = "Hello world 😄! This is amazing 😂!"
print(tokenize_sentence(sentence))
