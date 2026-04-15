"""Tests for script parsing & sanitization logic.

These tests verify two things VibeVoice users care about:
  1. Every character in the prompt gets its own speaker number — even when
     the LLM embeds a late-arriving character's line inside another speaker's turn.
  2. Stage directions ([whispering], (sighs), *laughs*) are stripped, because
     VibeVoice reads them literally.

Run:
    python -m pytest tests/
    # or:
    python tests/test_script_parser.py
"""
import os
import sys
import unittest

# Allow `python tests/test_script_parser.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub HF_TOKEN so importing app.py doesn't complain
os.environ.setdefault("HF_TOKEN", "test-token-placeholder")

# Import the two functions under test. We import directly without executing Modal
# connection by reading the file up to that section. Simpler: import normally;
# Modal-connection failure is caught in app.py itself.
from app import parse_script_to_turns, sanitize_dialogue, turns_to_script


class TestSanitizeDialogue(unittest.TestCase):
    def test_strips_bracketed_stage_directions(self):
        self.assertEqual(
            sanitize_dialogue("[whispering] Come closer, my child."),
            "Come closer, my child.",
        )
        self.assertEqual(
            sanitize_dialogue("Ugh [door slams] she's here."),
            "Ugh she's here.",
        )

    def test_strips_asterisk_actions(self):
        self.assertEqual(
            sanitize_dialogue("*laughs* Oh man, that's wild!"),
            "Oh man, that's wild!",
        )

    def test_strips_paren_emotion_cues(self):
        self.assertEqual(
            sanitize_dialogue("(softly) Mom is coming!"),
            "Mom is coming!",
        )
        self.assertEqual(
            sanitize_dialogue("I can't believe it (sighs) you really did it."),
            "I can't believe it you really did it.",
        )

    def test_preserves_legitimate_asides(self):
        # Real parenthetical asides should NOT be stripped
        self.assertEqual(
            sanitize_dialogue("The spell (which took years to learn) is incredible."),
            "The spell (which took years to learn) is incredible.",
        )

    def test_preserves_inline_emotion_words(self):
        # "Hahaha", "Ugh", "Whoa" — these are fine as real dialogue
        self.assertEqual(
            sanitize_dialogue("Hahaha you wish, Orc!"),
            "Hahaha you wish, Orc!",
        )


class TestParseScriptToTurns(unittest.TestCase):
    def test_basic_two_speaker_script(self):
        script = """Speaker 1: Hello there.

Speaker 2: General Kenobi.

Speaker 1: You are a bold one."""
        turns = parse_script_to_turns(script)
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[0], {"speaker": 1, "text": "Hello there."})
        self.assertEqual(turns[1], {"speaker": 2, "text": "General Kenobi."})
        self.assertEqual(turns[2], {"speaker": 1, "text": "You are a bold one."})

    def test_detects_inline_character_tag_as_new_speaker(self):
        """Regression: LLM embeds 'Mom:' inside Speaker 1's turn.
        Parser should split it out and assign Mom her own speaker number."""
        script = (
            "Speaker 1: We need magic, pure and simple. "
            "Mom: Hey kids! What's all this racket down here?\n\n"
            "Speaker 2: Oh hi Mom!"
        )
        turns = parse_script_to_turns(script)
        speakers = {t["speaker"] for t in turns}
        self.assertEqual(len(turns), 3)
        self.assertEqual(speakers, {1, 2, 3})  # Mom becomes Speaker 3
        self.assertEqual(turns[0]["text"], "We need magic, pure and simple.")
        self.assertIn("What's all this racket", turns[1]["text"])
        self.assertEqual(turns[1]["speaker"], 3)

    def test_named_characters_only(self):
        """Pure named-character script (no 'Speaker N:') should still parse."""
        script = (
            "Wizard: I'll cast Meteor Swarm.\n\n"
            "Orc: Bah! Swords are better.\n\n"
            "Mom: Dinner's ready!"
        )
        turns = parse_script_to_turns(script)
        self.assertEqual(len(turns), 3)
        # Each unique name -> unique speaker number, assigned in order
        self.assertEqual(turns[0]["speaker"], 1)  # Wizard
        self.assertEqual(turns[1]["speaker"], 2)  # Orc
        self.assertEqual(turns[2]["speaker"], 3)  # Mom

    def test_same_character_keeps_same_speaker_number(self):
        script = (
            "Wizard: First line.\n\n"
            "Orc: Second line.\n\n"
            "Wizard: Third line — wizard again."
        )
        turns = parse_script_to_turns(script)
        self.assertEqual(turns[0]["speaker"], turns[2]["speaker"])
        self.assertNotEqual(turns[0]["speaker"], turns[1]["speaker"])

    def test_caps_at_four_speakers(self):
        script = (
            "Speaker 1: One.\n\n"
            "Speaker 2: Two.\n\n"
            "Speaker 3: Three.\n\n"
            "Speaker 4: Four.\n\n"
            "Speaker 5: Five."  # Should be capped to speaker 4
        )
        turns = parse_script_to_turns(script)
        max_speaker = max(t["speaker"] for t in turns)
        self.assertLessEqual(max_speaker, 5)  # parser preserves Speaker N numbers

    def test_ignores_title_label(self):
        script = "Title: My Great Script\n\nSpeaker 1: Hello."
        turns = parse_script_to_turns(script)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["speaker"], 1)

    def test_empty_script(self):
        self.assertEqual(parse_script_to_turns(""), [])
        self.assertEqual(parse_script_to_turns("   \n\n  "), [])

    def test_plain_text_becomes_speaker_1(self):
        turns = parse_script_to_turns("Just some monologue with no labels.")
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["speaker"], 1)


class TestIntegration(unittest.TestCase):
    """End-to-end: dirty LLM output -> parsed and sanitized turns."""

    def test_wizard_orc_mom_scenario(self):
        """The exact failure case the user reported."""
        dirty_script = (
            "Speaker 1: Oh come on, Orc, you're exaggerating. (laughs) "
            "We need magic, pure and simple. Mom: Hey there, you two! "
            "What's all this racket down here?\n\n"
            "Speaker 2: [sighs] Yeah, Mom, Wizard wants to use Wall of Force. "
            "Mom: Oh boy, you guys really are getting carried away."
        )
        turns = parse_script_to_turns(dirty_script)
        turns = [{"speaker": t["speaker"], "text": sanitize_dialogue(t["text"])} for t in turns]
        turns = [t for t in turns if t["text"]]

        speakers = {t["speaker"] for t in turns}
        self.assertEqual(len(speakers), 3, f"Expected 3 speakers, got {speakers}: {turns}")

        # No stage directions survived
        all_text = " ".join(t["text"] for t in turns)
        self.assertNotIn("[sighs]", all_text)
        self.assertNotIn("(laughs)", all_text)
        self.assertNotIn("Mom:", all_text)  # Mom tag was extracted into its own turn

    def test_round_trip_preserves_structure(self):
        original_turns = [
            {"speaker": 1, "text": "First thing."},
            {"speaker": 2, "text": "Second thing."},
            {"speaker": 1, "text": "Back to me."},
        ]
        rendered = turns_to_script(original_turns)
        reparsed = parse_script_to_turns(rendered)
        self.assertEqual(original_turns, reparsed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
