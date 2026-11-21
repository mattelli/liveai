# PLAN — Natural pauses in speech synthesis

## 1. Goals

Make the Piper TTS output sound natural on long text by inserting real pauses
at punctuation and block boundaries, instead of reading a flattened run-on
line.

Scope: `src/audio_generator.py`, `src/tools/speech.py`, `config.toml`,
`config-code.toml`, `README.md`, tests. **No change to `src/live_ai.py`**
(see §7 — it already forwards the whole response text to `speak_if_enabled`).

Decisions taken with the user:

| # | Decision |
|---|----------|
| 1 | Layer B: explicit silence insertion (not just text normalization) |
| 2 | Base pauses: comma 150 ms, sentence 350 ms, paragraph 600 ms |
| 3 | Pauses are scaled by `speech_speed` (pause = base / speech_speed) |
| 4 | `to_wav()` uses the same pause logic as live `speak()` |
| 5 | Pauses are **always enabled** — no `pauses_enabled` toggle |
| 6 | Bullets / headings become separate sentences |
| 7 | Non-prose content (code blocks, tables) is **configurable** via `non_prose_content` in the config files: `false` (default) = not read aloud (replaced by the configurable `omitted_placeholder`), `true` = read aloud too — see §4.2 |

## 2. Current behavior

`SpeechGenerator.clean_for_speech()` (audio_generator.py:166-178) strips
markdown/emoji and then does `re.sub(r"\s+", " ", text).strip()`: **every
newline, blank line, heading and list break is collapsed into a single
space.** The cleaned string is handed to `voice.synthesize()` in one call
(audio_generator.py:180-218) and only the raw `audio_int16_bytes` chunks are
written to the stream — **no silence is ever emitted**.

Consequences: (a) espeak's only pause cue is punctuation, and commas get a
very short one; (b) block structure is gone, so paragraphs and list items run
together; (c) a long answer is read as one breathless stream. Code blocks and
tables are read out character by character, which is worse still.

`to_wav()` (audio_generator.py:141-164) has the same problem: one
`synthesize_wav()` call on the flattened text.

`[speech_gen]` has no pause keys.

## 3. Expected behavior

Pauses are applied **unconditionally** (no toggle):

1. The text is split into **segments** at clause / sentence / paragraph
   boundaries. The punctuation terminator **stays attached** to its segment
   (`"hello,"`, `"world."`), so espeak still applies its natural intonation;
   the explicit silence is *added after* it.
2. Each segment is synthesized separately and a silence of
   `base_ms / speech_speed` is written between segments:
   * after `,` `;` `:` `—` `–` → `pause_comma_ms` (150)
   * after `.` `!` `?` `…` → `pause_sentence_ms` (350)
   * at a blank line / block boundary → `pause_paragraph_ms` (600)
   * after the final segment → no pause
3. Block structure is preserved: a single newline, a heading line, or a list
   marker (`- `, `* `, `+ `, `1. `, `1) `) starts a new sentence; a blank line
   starts a new paragraph.
4. Non-prose blocks (fenced code, tables, indented code) are governed by
   `non_prose_content` (§4.2):
   * `false` (**default**) — not read aloud: each block is replaced by the
     configured `omitted_placeholder` sentence (default `"content omitted"`);
     an empty `omitted_placeholder` makes the voice stay silent instead.
   * `true` — the block is **spoken as well** (code and table text are read
     out, each line becoming its own sentence).
5. Silences are written through the existing `_write_chunk()` slice loop, so
   Ctrl+C / `request_stop()` still interrupts a pause within ~50 ms.
6. `to_wav()` writes the same silences as WAV frames, so an exported file
   matches live playback.

Example — `"First point.\n\n- alpha\n- beta"` becomes 4 segments:
`First point.` (paragraph pause) `alpha` (sentence) `beta` (sentence) —
instead of `First point. alpha beta` read in one breath.

## 4. Design

### 4.1 `SpeechGenerator.__init__` — new parameters

```python
pause_comma_ms: int = 150,
pause_sentence_ms: int = 350,
pause_paragraph_ms: int = 600,
non_prose_content: bool = False,
omitted_placeholder: str = "content omitted",
```

Stored as `self._pause_ms = {"comma": …, "sentence": …, "paragraph": …}`,
`self.non_prose_content` and `self.omitted_placeholder`. Validation: each
duration must be a non-negative number, else `ValueError` (same style as the
existing `speech_speed` / `volume` checks). An empty `omitted_placeholder`
means "stay silent"; `omitted_placeholder` is only used when
`non_prose_content` is `False`.

### 4.2 New pure helper — `segment_for_speech(text)`

```python
@staticmethod
def segment_for_speech(
    text: str,
    non_prose_content: bool = False,
    omitted_placeholder: str = "content omitted",
) -> list[tuple[str, str]]:
    """Split text into (segment_text, pause_after) pairs.

    pause_after is one of "none", "comma", "sentence", "paragraph".
    """
```

Steps:
1. **Handle non-prose blocks first**, before any whitespace handling, driven by
   `non_prose_content`:
   * `non_prose_content=False` (**default**) — each fenced code block
     (```` ``` … ``` ```` / `~~~`), markdown table (a line starting with `|`
     plus its `|---|` separator row) and indented code block (4-space /
     tab-indented run) is **replaced by `omitted_placeholder`** (inserted as
     its own sentence, so it gets a sentence pause around it). An empty
     `omitted_placeholder` drops the block with no replacement.
   * `non_prose_content=True` — the same blocks are **kept and spoken**: the
     fence/table markers are stripped and each code/table line becomes its own
     sentence.
   * In both cases: inline code `` `x` `` → backticks removed, the text is
     kept and read; URLs `http(s)://…` → replaced by a short spoken word
     (`"link"`) instead of being spelled out.

   Because both knobs are constructor parameters, `segment_for_speech` takes
   them as arguments rather than reading config directly — keeping the helper
   pure.
2. Reuse the markdown/emoji stripping of `clean_for_speech` (factor the two
   regexes into a small `_strip_formatting()` helper so both share it).
3. Normalize block structure: blank line → paragraph break; single newline,
   heading line, and list-marker line → sentence break.
4. Split each block into sentences on `[.!?…]+` (terminator kept), then each
   sentence into clauses on `[,;:]` and dashes (terminator kept).
5. Assign `pause_after`: clause → `"comma"`, last clause of a sentence →
   `"sentence"`, last sentence of a paragraph → `"paragraph"`, final segment
   of the whole text → `"none"`.
6. Drop empty segments; collapse whitespace **inside** a segment only.

Pure function, no audio, no I/O → directly unit-testable.

### 4.3 Silence bytes

```python
def _silence_bytes(self, ms: float) -> bytes:
    frames = int(self.samplerate * (ms / 1000.0))
    return b"\x00" * (frames * self.channels * self._SAMPLE_SIZES[self.datatype])
```

Reuses `_SAMPLE_SIZES` (line 73). `ms` is already speed-scaled by the caller:
`base_ms / self.speech_speed`.

### 4.4 `speak()` integration

```python
segments = self.segment_for_speech(text)
...
for seg_text, pause_after in segments:
    if self._stop_event.is_set():
        break
    for chunk in voice.synthesize(seg_text, syn_config=self.syn_config):
        if self._stop_event.is_set():
            break
        written += self._write_chunk(chunk.audio_int16_bytes)
    else:
        written += self._write_chunk(self._silence_bytes(ms_for(pause_after)))
```

The existing `stream.start()` / stop-event / `_abort_stream()` / `stream.stop()`
`finally` block is unchanged.

### 4.5 `to_wav()` integration

Set the WAV params explicitly (`setframerate(samplerate)`,
`setsampwidth(2)`, `setnchannels(channels)`) and call
`synthesize_wav(..., set_wav_format=False)` per segment, writing
`_silence_bytes()` frames between segments.

### 4.6 Config plumbing

`config.toml` and `config-code.toml`, under `[speech_gen]`:

```toml
pause_comma_ms = 150     # pause after , ; : — (scaled by speech_speed)
pause_sentence_ms = 350  # pause after . ! ? …
pause_paragraph_ms = 600 # pause at paragraph / blank-line boundaries
non_prose_content = false  # true = also read code blocks / tables aloud
omitted_placeholder = "content omitted"  # spoken instead of code blocks / tables when non_prose_content = false ("" = silent)
```

`src/tools/speech.py::_speech_kwargs()` forwards them with
`getattr(cfg, "<key>", <default>)`, matching the existing optional-key pattern
(a missing `[speech_gen]` section or missing keys keep the defaults). No
`pauses_enabled` key.

## 5. Verification — unit tests

New section `G10 pauses` in `tests/test_audio_generator.py`, using the existing
`FakePiperVoice` / `FakeOutputStream` fakes (no audio hardware, no model):

| Test | Asserts |
|------|---------|
| `test_segment_splits_on_sentence_and_paragraph` | `"A. B\n\nC"` → 3 segments, pauses `sentence`, `paragraph`, `none` |
| `test_segment_keeps_terminator_attached` | `"hello, world."` → `"hello,"` / `"world."` |
| `test_segment_bullets_and_headings_are_separate_sentences` | `"- a\n- b"` and `"# H\nx"` split per item/heading |
| `test_segment_drops_fenced_code_block` | `non_prose_content=False`: code absent; the placeholder sentence is present instead |
| `test_segment_drops_table` | `non_prose_content=False`: table rows absent; surrounding prose kept; placeholder present |
| `test_segment_drops_indented_code_block` | `non_prose_content=False`: indented block absent; placeholder present |
| `test_segment_empty_placeholder_is_silent` | `omitted_placeholder=""` → block dropped, no placeholder text |
| `test_segment_custom_placeholder_is_used` | `omitted_placeholder="see the code"` → that text appears |
| `test_segment_non_prose_content_true_speaks_code` | `non_prose_content=True`: code text present in the segments, no placeholder |
| `test_segment_non_prose_content_true_speaks_table` | `non_prose_content=True`: table cell text present, no placeholder |
| `test_segment_keeps_inline_code_text` | `` `x` `` → `x` (read aloud) |
| `test_segment_replaces_url_with_word` | `http://…` → `link` |
| `test_segment_collapses_whitespace_inside_segment` | inner runs of spaces/tabs → one space |
| `test_pauses_insert_silence_between_segments` | `stream.written` = audio, silence, audio…; silence length = `samplerate*0.35` frames × 2 bytes |
| `test_pause_durations_scale_with_speech_speed` | `speech_speed=2.0` halves the silence byte count |
| `test_no_trailing_pause_after_last_segment` | last written buffer is audio, not silence |
| `test_stop_request_during_pause_aborts_playback` | stop set while writing a pause → loop breaks, `stream.aborted == 1` |
| `test_to_wav_inserts_silence_frames` | WAV frame count = audio frames + silence frames |
| `test_negative_pause_duration_raises` | `ValueError` |

New tests in `tests/test_tools.py` (T16 speech section):

| Test | Asserts |
|------|---------|
| `test_speech_kwargs_forwards_pause_config` | the 5 keys (3 durations + toggle + placeholder) reach the factory |
| `test_speech_kwargs_pause_defaults_when_absent` | missing keys → `150, 350, 600, False, "content omitted"` |

Regression: the full suite (`python -m pytest tests/ -q`) must stay green.
Note: `clean_for_speech` keeps its current signature and output (still used by
the `G1 clean` tests and by `strip_formatting`), so those tests are unaffected.

## 6. Tasks — one commit per task

Proceed **one task at a time**; each task ends with its tests green and a
commit whose message details what was done. **Ask for confirmation before
starting the next task.**

| Task | Content | Commit message (draft) |
|------|---------|------------------------|
| **T1** | `SpeechGenerator` pause parameters + validation + `non_prose_content` + `omitted_placeholder`; `[speech_gen]` keys in both TOMLs; `_speech_kwargs` forwarding. Behavior unchanged for now. Tests: validation + kwargs forwarding/defaults. | `feat(speech): add pause duration configuration` |
| **T2** | `_strip_formatting()` factor-out + `segment_for_speech()` pure helper, including non-prose handling (code blocks, tables, URLs) driven by `non_prose_content` / `omitted_placeholder`. Tests: the 13 segmentation tests. | `feat(speech): split text into clause/sentence/paragraph segments` |
| **T3** | `_silence_bytes()` + `speak()` per-segment synthesis with speed-scaled silences. Tests: insertion, scaling, no trailing pause, stop-during-pause. | `feat(speech): insert speed-scaled silences between segments when speaking` |
| **T4** | `to_wav()` per-segment synthesis + silence frames. Tests: frame count. | `feat(speech): apply the same pauses to to_wav exports` |
| **T5** | README: document the `[speech_gen]` pause keys, `non_prose_content` and the placeholder. | `docs: document the speech pause options` |

## 7. Out of scope / notes

* `src/live_ai.py` needs **no change**: `_handle_round_end()`
  (live_ai.py:2549-2551) already joins the round fragments and passes the full
  text to `speech_tools.speak_if_enabled()`, which forwards it to
  `SpeechGenerator.speak()`. All pause logic belongs in the generator, so the
  fix stays in one place and also covers `to_wav()`.
* No change to the Ctrl+C semantics or the SIGINT guard.
* No SSML/prosody tuning of the Piper model itself (`length_scale`,
  `noise_scale` untouched).
