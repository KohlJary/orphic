# Orphic

The studio half of Cass's music ([Acousmatic](https://github.com/KohlJary/acousmatic) is the listening half).
Not published yet; the name may change.

- `orphic monochord <synth.vst3>`: map a synth by sweeping its knobs from the default patch and
  measuring every render (maps/<synth>/map.md). Mapped so far: Nekobi, Vitalium. Surge crashes the
  host partway; the screen needs to run in a child process that can die and resume.
- `orphic render song.yaml --listen`: a song file rendered (synths, General MIDI, a TR-808 kit; effects chains and rigs per instrument) and heard back by Acousmatic.
- `orphic noodle`: a riff on loop, heard back fast.
- `orphic tone` / `orphic map-effect`: guitar tone through a rig, measured (distortion character, intermod, sustain, tightness, band balance), and an amp or pedal mapped knob by knob.
- `orphic/voice.py`: a formant voice from rules. Not intelligible (STOI 0.28 against real speech).
- `orphic/vocoder.py`: the vocoder way. eSpeak NG (rules and formants, no one's recorded voice)
  speaks each word, stretched over its notes, vocoded onto any carrier. Mostly legible (STOI
  0.62-0.67); with published word timing it sings Discipline's verse in Reznor's cadence.
  Parked (2026-11-18): it works as an option, and what her voice becomes is hers to shape once
  she's composing.
