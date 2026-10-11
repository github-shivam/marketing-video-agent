"""What the screenwriter is told about each video model and each kind of video.

Distilled from Google's Veo 3.1 prompting guide and ByteDance's Seedance 2.5
announcement and prompt guidance. Two things are deliberately NOT here:

* Exact prompt syntax (timestamps, ``<sfx>`` brackets, ``Shot 1 [0-3s]``).
  That is applied in code by :mod:`agents.render`, so the model never has to
  remember it. The writer only supplies the content, shot by shot.
* Anything the sources did not document. Where a rule is an inference rather
  than documented behaviour it says so.
"""

from __future__ import annotations

from state import VideoIntent

#: The rules every video model shares.
COMMON_RULES = """\
CRAFT RULES (apply to every shot):
- Be concrete and filmable. Describe what a camera would actually see and a
  microphone would actually hear. "A cinematic, emotional scene" gives the
  model nothing; "a woman in a green raincoat laughs as rain runs off her
  hood" does.
- Use full descriptive sentences with cause and effect, not lists of tags.
- Describe every recurring person and object fully, ONCE, in the CONTINUITY
  field (age range, hair, clothing, build). In each shot, refer to them by the
  same short noun phrase every time ("the woman in the blue linen shirt"),
  never by a fresh description and never as a full sentence. The shot's
  subject is a noun phrase and its action is a verb phrase; together they must
  read as one grammatical sentence. These models have no memory beyond the
  prompt, which is why the wording must not drift.
- Never name a real person, celebrity, or copyrighted character. Describe them
  instead ("a man in his thirties with glasses and a short beard"). A real
  name inside spoken dialogue is fine.
- Do not rely on readable text inside the frame. Generated lettering is
  frequently misspelled. Show products and places without legible signage.
- Phrase exclusions as what IS there: "an empty street" rather than "no cars".
- Never invent facts. Product names, prices and claims must come from the
  source material; if it is silent, show the benefit without stating a number.
- ONE camera move per shot. Stacking push, pan and tilt in one shot makes the
  image unstable. Name the shot size (wide, medium, close-up, extreme
  close-up) and the angle (eye-level, low, high, overhead).
- Each shot has one main action. Slow, small, continuous motion renders far
  better than fast or complex motion.
- Sound is part of the prompt. Fill sfx and ambient with specific sounds
  (footsteps on gravel, a kettle ticking as it cools), never mood words.
- Timing: shots in a segment must be gapless and together fill the whole
  segment. Too little in a span makes the model improvise; too much makes it
  add cuts or drop beats."""

VEO_GUIDE = """\
TARGET MODEL: Google Veo 3.1.
- Each segment is ONE generation of a continuous clip. Shots inside it are
  cuts or camera changes within that clip, directed with timestamps.
- Segment 0 is the base clip. Every later segment EXTENDS the same scene by 7
  seconds, so it must continue the same location, people, light and style. Do
  not cut to an unrelated scene. The segment lengths are given under STRUCTURE.
- Veo composes each shot as: cinematography, then subject, then action, then
  context, then style and ambiance. Fill those fields accordingly.
- Camera vocabulary that works: dolly, tracking, crane, aerial, slow pan,
  POV; wide, close-up, extreme close-up, low angle, two-shot; shallow depth
  of field, macro lens, wide-angle lens, soft focus.
- Speech: one speaker per shot, in short natural sentences. Put the exact
  words in dialogue, who is speaking in speaker, and the manner in delivery.
  A line with no speaker produces mouth movement with invented words.
- Audio is always generated. Whatever you leave out, the model invents, so
  describe sfx and ambient explicitly."""

SEEDANCE_GUIDE = """\
TARGET MODEL: ByteDance Seedance 2.5.
- The whole video is ONE segment generated in a single pass (up to 30s).
  There is no extension chain, so plan the full arc inside this segment.
- Seedance reads a prompt in this order: subject, action, scene, visual style,
  camera and cuts, sound. Fill the fields accordingly.
- It follows shot labels and integer-second timings closely. Give each shot
  one main action and one camera move.
- Keep each shot's content balanced: a very short shot with a lot in it gets
  dropped beats, a long shot with little in it gets improvised content.
- Use plain film terms: close-up, push in, tracking shot, low angle, FPV,
  dolly zoom, speed ramp. If you use a niche technique, also say what it looks
  like on screen.
- Speech: one language per line. Attach delivery notes to the speaker rather
  than to individual words. Do not repeat words within a line.
- Describe a recurring object or person identically in every shot to avoid
  drift between cuts.
- Seedance may add stray subtitles or music; the renderer adds the
  exclusions, so do not mention them."""

#: Structural direction per intent. The classifier picks one of these.
INTENT_GUIDES: dict[VideoIntent, str] = {
    "product_promo": (
        "PRODUCT PROMO. Open on the product in an irresistible hero moment in "
        "the first second. Show it solving a real problem for the audience, in "
        "use, with sensory detail (texture, steam, sound). Close on the "
        "product and the single call to action. Claims come only from the "
        "source material."
    ),
    "explainer": (
        "EXPLAINER. Open on a question or a surprising claim the audience "
        "recognises. Make ONE idea click through ONE analogy from the "
        "audience's daily life, shown visually rather than lectured. Close on "
        "a one-line takeaway. Never try to cover two ideas."
    ),
    "educational_howto": (
        "HOW-TO. Show the finished result first, then the steps, each visibly "
        "distinct, in order, with the action of the step on screen. Return to "
        "the result at the end. No step is described only in words."
    ),
    "brand_story": (
        "BRAND STORY. Start with a human moment, not a logo. Show the "
        "before and the after, then the reason the brand exists. End on the "
        "value the brand stands for, not on a sales line."
    ),
    "announcement": (
        "ANNOUNCEMENT. Lead with the news in the first second. Then the one "
        "detail that matters (what, when, where or how much), then the action "
        "to take. Energy should match the news."
    ),
    "entertainment": (
        "ENTERTAINMENT. A premise with a setup, an escalation and a payoff. "
        "Commit to a single mood. The brand or message, if any, is a light "
        "touch, never a pitch."
    ),
}


def model_guide(video_model: str) -> str:
    """Return the guidance block for the target video model."""
    return SEEDANCE_GUIDE if video_model == "seedance" else VEO_GUIDE
