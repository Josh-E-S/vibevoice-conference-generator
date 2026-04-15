# Example Prompts

A grab-bag of prompts covering different tones, speaker counts, and edge cases. Paste any of these into the app's prompt box to try it.

Each example also serves as a manual regression test — after generating, verify:
- [ ] Correct number of unique speaker tags (matches "Expected speakers" below)
- [ ] No `[brackets]`, `(parentheticals)`, or `*asterisks*` leaked into the audio
- [ ] Each speaker change is on its own turn (no "Mom: ..." buried inside another speaker's paragraph)

---

## 1. LARP Interruption (comedy, 3 speakers — includes a late arrival)

> A Wizard and Orc arguing about which spell is most powerful against dragons. Suddenly, their Mom comes downstairs into the basement to interrupt their LARPing session. Funny, humorous.

**Expected speakers:** 3 (Wizard, Orc, Mom)
**Why it's a good test:** Mom arrives mid-scene — earlier parser versions missed this and merged her lines into the previous speaker's turn.

---

## 2. Product Strategy Meeting (business, 4 speakers)

> A 4-person product meeting at a SaaS startup debating whether to raise prices. The CEO wants to raise, the CFO is cautious, the Head of Product worries about churn, and a customer success lead shares real user feedback.

**Expected speakers:** 4
**Why it's a good test:** Clear multi-role professional dialogue — validates that 4 distinct speakers are produced without drift.

---

## 3. Solo TED Talk (monologue, 1 speaker)

> A passionate 5-minute TED-style talk by a neuroscientist explaining why boredom is secretly the most creative mental state, with specific examples and a call to action.

**Expected speakers:** 1
**Why it's a good test:** Verifies that long-form single-speaker content works and the parser doesn't hallucinate extra speakers.

---

## 4. Detective Interrogation (dramatic, 3 speakers)

> A hard-boiled detective interrogates a nervous suspect in a small-town murder case, while the suspect's lawyer repeatedly objects and tries to end the interview. Tense, back-and-forth.

**Expected speakers:** 3 (Detective, Suspect, Lawyer)
**Why it's a good test:** Tests dramatic tone and ensures the LLM doesn't resort to forbidden stage directions like `[slams table]` or `(nervously)`.

---

## 5. Podcast Interview with Dog Expert (casual, 2 speakers)

> A casual 10-minute podcast interview where the host asks a dog behaviorist why dogs tilt their heads when you talk to them, with lots of tangents, personal stories, and funny dog examples.

**Expected speakers:** 2 (Host, Expert)
**Why it's a good test:** Natural conversational flow, tests that the LLM uses in-dialogue emotion ("haha", "oh wow") instead of stage directions.

---

## Running the parser tests

```bash
cd /path/to/Conference-Generator-VibeVoice
python tests/test_script_parser.py
# or:
python -m unittest tests.test_script_parser -v
```
