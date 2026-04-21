import argparse
import subprocess
import sys
import os
import tempfile
import numpy as np
from scipy.io import wavfile
import mido
from pathlib import Path

def apply_reverb(audio, sample_rate, reverb_amount=0):
    """
    Apply stereo algorithmic reverb to audio signal with large room sound.
    
    Args:
        audio: Input audio signal (float32, mono)
        sample_rate: Sample rate in Hz
        reverb_amount: Reverb mix percentage (0-100), 0=dry, 100=fully wet
    
    Returns: Stereo audio with reverb applied (2D array: samples x 2 channels)
    """
    if reverb_amount <= 0:
        # Return stereo with no reverb
        stereo = np.column_stack([audio, audio])
        return stereo
    
    # Clamp reverb amount to 0-100
    reverb_amount = np.clip(reverb_amount, 0, 100)
    mix = reverb_amount / 100.0
    
    # Create stereo reverb with different delays for left and right channels
    left_reverb = np.zeros_like(audio)
    right_reverb = np.zeros_like(audio)
    
    # Left channel delays (prime numbers for natural sound)
    left_delays_ms = [
        # Early reflections (0-100ms)
        23, 37, 53, 71, 89,
        # Mid reflections (100-500ms)
        113, 157, 211, 271, 337, 419, 487,
        # Late reflections / decay tail (500ms-2000ms)
        571, 677, 809, 977, 1123, 1289, 1451, 1613, 1787, 1949
    ]
    
    # Right channel delays (offset from left for stereo width)
    right_delays_ms = [
        29, 43, 61, 79, 97,
        127, 173, 227, 293, 359, 433, 503,
        607, 719, 857, 1009, 1163, 1319, 1483, 1657, 1823, 1987
    ]
    
    # Apply left channel reverb with smooth 2-second decay
    for i, delay_ms in enumerate(left_delays_ms):
        delay_samples = int(delay_ms * sample_rate / 1000.0)
        if delay_samples < len(audio):
            # Exponential decay over 2 seconds
            decay_position = delay_ms / 2000.0  # Position in 2-second decay
            decay = np.exp(-3.5 * decay_position)  # Exponential decay
            
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * decay
            left_reverb += delayed
    
    # Apply right channel reverb with smooth 2-second decay
    for i, delay_ms in enumerate(right_delays_ms):
        delay_samples = int(delay_ms * sample_rate / 1000.0)
        if delay_samples < len(audio):
            # Exponential decay over 2 seconds
            decay_position = delay_ms / 2000.0
            decay = np.exp(-3.5 * decay_position)
            
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = audio[:-delay_samples] * decay
            right_reverb += delayed
    
    # Normalize reverb channels to match dry signal's peak level
    # This ensures the mix percentage reflects the actual perceived amount
    dry_peak = np.max(np.abs(audio))
    if dry_peak > 0:
        left_peak = np.max(np.abs(left_reverb))
        if left_peak > 0:
            left_reverb = left_reverb * (dry_peak / left_peak)
        
        right_peak = np.max(np.abs(right_reverb))
        if right_peak > 0:
            right_reverb = right_reverb * (dry_peak / right_peak)
    
    # Mix dry and wet for each channel
    dry_level = 1.0 - mix
    wet_level = mix
    
    left_channel = audio * dry_level + left_reverb * wet_level
    right_channel = audio * dry_level + right_reverb * wet_level
    
    # Stack into stereo array
    stereo = np.column_stack([left_channel, right_channel])
    
    return stereo

# =============================================================================
# FluidSynth renderer
# =============================================================================
#
# The previous implementation synthesised raw oscillators (sine/square/saw/
# triangle) in numpy + a hand-rolled ADSR. The sound was harsh and the chord
# mix was muddy. We now delegate synthesis to **FluidSynth** + a General-MIDI
# SoundFont (FluidR3_GM.sf2) which:
#   - honours the MPE pitch-bend per channel already present in our MIDIs
#     (RPN ±2 semitones), so every 53-EDO note lands on its exact frequency,
#   - provides real sampled/synthetic instruments (piano, Rhodes, pad, sawtooth
#     lead, …) — not bare oscillators,
#   - has a built-in algorithmic reverb we can dial in via the ``reverb`` arg.
#
# The legacy ``apply_reverb`` function is kept as a post-processing stereo
# reverb (used when the user prefers the prior reverb character), but by
# default we route ``reverb`` into FluidSynth's own reverb send.

# Mapping from the historical ``waveform`` argument to a General-MIDI program
# number. The keys match what the notebooks and existing scripts already pass.
# We pick instruments that *resemble* the requested raw waveform but sound
# musical — e.g. "square" → GM Lead 1 (Square), "sawtooth" → GM Lead 2 (Saw).
_WAVEFORM_TO_GM_PROGRAM = {
    'sine':     73,   # Flute                 (mostly-sinusoidal tone)
    'triangle': 4,    # Electric Piano 1      (soft, triangle-ish partials)
    'square':   80,   # Lead 1 (Square)
    'sawtooth': 81,   # Lead 2 (Sawtooth)
    'clarinet': 71,   # Clarinet
    'piano':    0,    # Acoustic Grand Piano
    'rhodes':   4,    # Electric Piano 1
    'pad':      88,   # Pad 1 (new age)
    'organ':    17,   # Percussive Organ
    'strings':  48,   # String Ensemble 1
}

# Waveforms synthesised directly in numpy (pure oscillators — no soundfont).
# These guarantee an *exact* 53-EDO frequency because we compute
# f = 440 * 2**((midi + bend_fraction*2 - 69)/12) from the MIDI + pitch-bend
# ourselves, bypassing any sampled-instrument tuning drift.
_PURE_WAVEFORMS = {'sine', 'square', 'sawtooth', 'saw', 'triangle'}

# Canonical SoundFont search order.  The first existing file wins.  All three
# shipped by the ``fluid-soundfont-gm`` / ``fluid-soundfont-gs`` Ubuntu
# packages live under ``/usr/share/sounds/sf2``.  Users can override via the
# ``FLUIDSYNTH_SF2`` environment variable.
_DEFAULT_SOUNDFONTS = [
    os.environ.get('FLUIDSYNTH_SF2', ''),
    '/usr/share/sounds/sf2/FluidR3_GM.sf2',
    '/usr/share/sounds/sf2/FluidR3_GS.sf2',
    '/usr/share/sounds/sf2/default-GM.sf2',
    '/usr/share/soundfonts/FluidR3_GM.sf2',
    '/usr/share/soundfonts/default.sf2',
]


def _find_soundfont():
    for p in _DEFAULT_SOUNDFONTS:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "No SoundFont (.sf2) found. Install one with\n"
        "  sudo apt-get install fluid-soundfont-gm\n"
        "or set FLUIDSYNTH_SF2 to an explicit path."
    )


# =============================================================================
# Pan + ADSR helpers (per-note overlays on top of FluidSynth output)
# =============================================================================

# MIDI note range used to map pitch → stereo pan position. Notes at or below
# PAN_LO_MIDI pan fully left (-pan_range_deg); notes at or above PAN_HI_MIDI
# pan fully right (+pan_range_deg); notes in between interpolate linearly.
_PAN_LO_MIDI = 36.0   # ~C2
_PAN_HI_MIDI = 96.0   # ~C7


def _pan_gains(midi_note, pan_range_deg):
    """Equal-power pan gains (L, R) for a given MIDI note.

    Low notes pan left, high notes pan right. ``pan_range_deg`` is the max
    deviation from centre (e.g. 30 → notes spread across ±30°).
    """
    x = (float(midi_note) - _PAN_LO_MIDI) / max(1e-6, _PAN_HI_MIDI - _PAN_LO_MIDI)
    x = float(np.clip(x, 0.0, 1.0))
    pan_deg = (x * 2.0 - 1.0) * float(pan_range_deg)   # [-range .. +range]
    # Equal-power pan law: gain = cos(θ), sin(θ) with θ ∈ [0, π/2].
    theta = (np.clip(pan_deg / 90.0, -1.0, 1.0) + 1.0) * np.pi / 4.0
    return float(np.cos(theta)), float(np.sin(theta))


def _build_adsr_envelope(dur_sec, sample_rate, attack, decay, sustain, release):
    """Multiplicative ADSR envelope, length = (dur + release) samples.

    Shapes: sin² attack ramp, linear decay, flat sustain, exponential release.
    This rides on top of the soundfont's own envelope — it does NOT replace
    it; it just smooths the attack edge and lengthens the release tail for
    an organic, musical decay between chords.
    """
    n_on  = max(1, int(dur_sec * sample_rate))
    n_rel = max(0, int(release  * sample_rate))
    n_att = max(1, int(attack   * sample_rate))
    n_dec = max(0, int(decay    * sample_rate))
    # Constrain attack+decay to not exceed the note-on length.
    n_att = min(n_att, max(1, n_on // 2))
    n_dec = min(n_dec, max(0, n_on - n_att))
    env = np.empty(n_on + n_rel, dtype=np.float32)
    # Attack: sin² ramp 0 → 1 (smooth, click-free)
    env[:n_att] = np.sin(np.linspace(0.0, np.pi / 2.0, n_att, dtype=np.float32)) ** 2
    # Decay: linear 1 → sustain
    if n_dec > 0:
        env[n_att:n_att + n_dec] = np.linspace(1.0, sustain, n_dec, dtype=np.float32)
    # Sustain level for the remainder of note-on
    if n_on > n_att + n_dec:
        env[n_att + n_dec:n_on] = sustain
    # Release: exponential decay sustain → 0 over ``release`` seconds, then
    # forced to exact silence by a linear fade multiplier so the tail ends
    # at 0.0 (not exp(-k) ≈ 0.006 which is still audible).
    if n_rel > 0:
        k = np.linspace(0.0, 1.0, n_rel, dtype=np.float32)
        fade = 1.0 - k                           # linear 1 → 0
        env[n_on:n_on + n_rel] = sustain * np.exp(-5.0 * k) * fade
    return env


def _pure_osc(freq_hz, n_samples, sample_rate, waveform):
    """Generate ``n_samples`` of a pure oscillator at the exact ``freq_hz``.
    Sawtooth and square are band-limited via polyBLEP to avoid aliasing."""
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32) / float(sample_rate)
    phase = (freq_hz * t) % 1.0                  # [0, 1)
    if waveform == 'sine':
        return np.sin(2.0 * np.pi * phase).astype(np.float32)
    if waveform == 'triangle':
        # Triangle from phase: 1 - |4*phase - 2 - 1|, centred at 0
        return (2.0 * np.abs(2.0 * phase - 1.0) - 1.0).astype(np.float32) * -1.0
    # Band-limited saw / square via polyBLEP residual.
    dt = freq_hz / float(sample_rate)

    def _poly_blep(t_, dt_):
        out = np.zeros_like(t_)
        m1 = t_ < dt_
        tt = t_[m1] / dt_
        out[m1] = tt + tt - tt * tt - 1.0
        m2 = t_ > (1.0 - dt_)
        tt = (t_[m2] - 1.0) / dt_
        out[m2] = tt * tt + tt + tt + 1.0
        return out

    if waveform in ('sawtooth', 'saw'):
        saw = 2.0 * phase - 1.0
        saw -= _poly_blep(phase, dt)
        return saw.astype(np.float32)
    if waveform == 'square':
        sq = np.where(phase < 0.5, 1.0, -1.0).astype(np.float32)
        sq += _poly_blep(phase, dt)
        sq -= _poly_blep((phase + 0.5) % 1.0, dt)
        return sq.astype(np.float32)
    # Fallback
    return np.sin(2.0 * np.pi * phase).astype(np.float32)


def render_mpe_to_audio_data(
    midi_path,
    sample_rate=44100,
    speed=1.2,
    waveform='sine',
    reverb=0,
    save_path=None,
    *,
    instrument=None,
    soundfont=None,
    adsr=None,
    pan_range_deg=30.0,
    use_fluidsynth_reverb=False,   # legacy kwarg, ignored (we render per-note)
):
    """Render an MPE MIDI file to stereo int16 audio.

    Each note is rendered individually with FluidSynth (exact 53-EDO
    frequency via MPE pitch bend — see MIDI precision notes), then a
    multiplicative ADSR overlay smooths the attack and lengthens the release
    for an organic tail, then the note is panned left↔right according to its
    pitch and mixed into the global stereo output buffer.

    Args:
        midi_path:     Path to the MPE MIDI file.
        sample_rate:   Output sample rate. Default 44100.
        speed:         Playback speed multiplier. Default 1.2.
        waveform:      Historic arg mapping to a GM program via
                       :data:`_WAVEFORM_TO_GM_PROGRAM`. Ignored if
                       ``instrument`` is given.
        reverb:        Post-mix reverb amount 0–100 (uses :func:`apply_reverb`).
        save_path:     Optional ``.wav`` path to write.
        instrument:    Explicit GM program number (0–127). Overrides
                       ``waveform``.
        soundfont:     Explicit ``.sf2`` path.
        adsr:          Optional dict overriding the defaults
                       ``{'attack':0.012,'decay':0.04,'sustain':0.90,'release':0.55}``.
                       All values are in seconds except ``sustain`` (0..1).
        pan_range_deg: Max pan deviation from centre (default 30 → low notes
                       at -30°, high notes at +30°, equal-power law).

    Returns: ``(audio_int16, sample_rate)`` with ``audio_int16`` shape
    ``(2, n_samples)``.
    """
    import fluidsynth

    midi_path = Path(midi_path)
    if not midi_path.exists():
        print(f"Error: File not found: {midi_path}")
        return None, None

    # Pure-oscillator mode is selected by waveform name (no soundfont needed).
    pure_mode = (instrument is None) and (waveform in _PURE_WAVEFORMS)
    sf2 = None
    if not pure_mode:
        sf2 = soundfont or _find_soundfont()
    program = int(instrument) if instrument is not None else _WAVEFORM_TO_GM_PROGRAM.get(waveform, 0)
    program = max(0, min(127, program))

    # ADSR overlay defaults — gentle shaping on top of the soundfont envelope.
    adsr_params = {'attack': 0.012, 'decay': 0.04, 'sustain': 0.90, 'release': 0.55}
    if adsr:
        adsr_params.update(adsr)

    if pure_mode:
        print(f"play_mpe(PureOsc+ADSR) | waveform={waveform!r} | reverb={reverb}% "
              f"| pan=±{pan_range_deg:.0f}° | speed={speed}x")
    else:
        print(f"play_mpe(FluidSynth+ADSR) | sf2={Path(sf2).name} | program={program} "
              f"({waveform!r}) | reverb={reverb}% | pan=±{pan_range_deg:.0f}° | speed={speed}x")

    # ── Parse MIDI → list of notes with captured pitch-bend per channel ──
    # MPE files emit a bend on a channel just before each note_on; we must
    # snapshot the current bend at note_on time so every voice renders at its
    # exact 53-EDO frequency even when many notes share a channel over time.
    mid = mido.MidiFile(str(midi_path))
    channel_bends = {i: 0 for i in range(16)}   # signed [-8192, 8191]
    active = {}                                 # (ch, note) -> (start, bend, vel)
    notes = []
    t = 0.0
    for msg in mid:
        t += msg.time / max(speed, 1e-6)
        if msg.type == 'pitchwheel':
            channel_bends[msg.channel] = int(msg.pitch)
        elif msg.type == 'note_on' and msg.velocity > 0:
            active[(msg.channel, msg.note)] = (t, channel_bends[msg.channel], int(msg.velocity))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active:
                st, bend, vel = active.pop(key)
                notes.append({'start': st, 'dur': max(0.02, t - st),
                              'midi': int(msg.note), 'bend': bend, 'vel': vel,
                              'ch': msg.channel})
    # Close any hanging notes at EOF.
    for (ch, note), (st, bend, vel) in active.items():
        notes.append({'start': st, 'dur': max(0.02, t - st),
                      'midi': int(note), 'bend': bend, 'vel': vel, 'ch': ch})

    if not notes:
        print("⚠️ No playable MIDI notes found!")
        return None, None

    release_sec = float(adsr_params['release'])
    release_samples = int(release_sec * sample_rate)

    total_sec = max(n['start'] + n['dur'] for n in notes) + release_sec + 1.5
    total_samples = int(total_sec * sample_rate)
    out = np.zeros((2, total_samples), dtype=np.float32)

    # ── Shared FluidSynth instance (only if we need it) ──────────────────
    fs = None
    if not pure_mode:
        fs = fluidsynth.Synth(samplerate=float(sample_rate))
        try:
            fs.setting('synth.gain', 0.7)
            fs.setting('synth.reverb.active', 0)   # handled post-mix
            fs.setting('synth.chorus.active', 0)
        except Exception:
            pass
        sfid = fs.sfload(str(sf2))
        fs.program_select(0, sfid, 0, program)

    PB_MIN, PB_MAX = -8192, 8191
    PB_RANGE_SEMITONES = 2.0   # MPE RPN range set by tokenizer (±2 semitones)

    for nn in notes:
        n_on = int(nn['dur'] * sample_rate)

        if pure_mode:
            # Exact 53-EDO frequency from MIDI + bend. No soundfont, no
            # tuning drift — the oscillator runs at the precise Hz required.
            bend_frac = float(np.clip(nn['bend'], PB_MIN, PB_MAX)) / 8192.0
            midi_f = float(nn['midi']) + bend_frac * PB_RANGE_SEMITONES
            freq_hz = 440.0 * 2.0 ** ((midi_f - 69.0) / 12.0)
            # Velocity-based amplitude (vel 1..127 → 0.08..1.0, soft curve)
            amp = 0.08 + 0.92 * (float(nn['vel']) / 127.0) ** 1.6
            total = n_on + release_samples
            osc = _pure_osc(freq_hz, total, sample_rate, waveform) * amp
            # Low-amplitude tweaks per waveform so the mix level is similar
            if waveform == 'sine':
                osc *= 0.85
            elif waveform in ('sawtooth', 'saw'):
                osc *= 0.55
            elif waveform == 'square':
                osc *= 0.45
            elif waveform == 'triangle':
                osc *= 0.75
            mono = osc
        else:
            # Soundfont path: set bend, noteon, pull on-samples, noteoff, pull
            # release-samples.
            fs.pitch_bend(0, int(np.clip(nn['bend'], PB_MIN, PB_MAX)))
            fs.noteon(0, nn['midi'], nn['vel'])
            buf_on = fs.get_samples(n_on) if n_on > 0 else np.zeros(0, dtype=np.int16)
            fs.noteoff(0, nn['midi'])
            buf_rel = fs.get_samples(release_samples) if release_samples > 0 else np.zeros(0, dtype=np.int16)
            raw = np.concatenate([buf_on, buf_rel])
            if raw.size == 0:
                continue
            if raw.size % 2 == 1:
                raw = raw[:-1]
            mono = raw.reshape(-1, 2).mean(axis=1).astype(np.float32) / 32768.0

        if mono.size == 0:
            continue

        # ADSR overlay (organic attack/release; release forced to exact zero)
        env = _build_adsr_envelope(
            nn['dur'], sample_rate,
            attack=adsr_params['attack'], decay=adsr_params['decay'],
            sustain=adsr_params['sustain'], release=release_sec,
        )
        m = min(env.size, mono.size)
        voiced = mono[:m] * env[:m]

        # Stereo pan by pitch (low → left, high → right)
        L_gain, R_gain = _pan_gains(nn['midi'], pan_range_deg)

        start_s = int(nn['start'] * sample_rate)
        end_s = start_s + voiced.size
        if end_s > out.shape[1]:
            pad = end_s - out.shape[1]
            out = np.concatenate([out, np.zeros((2, pad), dtype=np.float32)], axis=1)
        out[0, start_s:end_s] += voiced * L_gain
        out[1, start_s:end_s] += voiced * R_gain

    if fs is not None:
        fs.delete()

    # ── Optional post-mix reverb (legacy stereo reverb, adds depth/width) ──
    if reverb > 0:
        dry_mono = out.mean(axis=0)
        wet = apply_reverb(dry_mono, sample_rate, reverb)   # (N, 2)
        if wet.shape[0] < out.shape[1]:
            wet = np.vstack([wet, np.zeros((out.shape[1] - wet.shape[0], 2), dtype=wet.dtype)])
        elif wet.shape[0] > out.shape[1]:
            wet = wet[:out.shape[1]]
        mix = float(np.clip(reverb, 0, 100)) / 100.0
        # Preserve the dry panned image; add reverb wet underneath.
        out[0] = out[0] * (1.0 - 0.35 * mix) + wet[:, 0].astype(np.float32) * 0.5 * mix
        out[1] = out[1] * (1.0 - 0.35 * mix) + wet[:, 1].astype(np.float32) * 0.5 * mix

    # ── Normalise and convert to int16 (2, N) for IPython.display.Audio ──
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out = out / peak * 0.95
    audio_int16 = (out * 32767.0).astype(np.int16)          # (2, N)

    print(f"Rendered {len(notes)} notes → {out.shape[1]} samples "
          f"({out.shape[1] / sample_rate:.2f}s) peak={peak:.3f}")

    if save_path is not None:
        wavfile.write(str(save_path), sample_rate, audio_int16.T)   # scipy wants (N, 2)
        print(f'Saved: {Path(save_path).name}')

    return audio_int16, sample_rate


# =============================================================================
# Legacy oscillator renderer (kept for reference / fallback)
# =============================================================================


def _render_mpe_to_audio_data_oscillator(midi_path, sample_rate=44100, speed=1.2, waveform='sine', reverb=0, save_path=None):
    """Pure-numpy oscillator + ADSR renderer. Original implementation kept as
    a fallback when FluidSynth is unavailable. Not called by default."""
    mid = mido.MidiFile(midi_path)

    # Storage for note events: (start_time, duration, frequency, velocity)
    note_events = []

    # State tracking
    channel_bends = {i: 0.0 for i in range(16)}
    active_notes = {}

    current_time = 0.0

    # Parse MIDI messages
    for msg in mid:
        current_time += msg.time / speed

        if msg.type == "pitchwheel":
            # Pitch Bend Range: +/- 2 semitones (+/- 200 cents)
            cents = (msg.pitch / 8192.0) * 200.0
            channel_bends[msg.channel] = cents

        elif msg.type == "note_on" and msg.velocity > 0:
            key = (msg.channel, msg.note)
            # Implicit note-off: close previous note on same (ch, note)
            if key in active_notes:
                start_time, freq_old, vel_old = active_notes.pop(key)
                duration = current_time - start_time
                if duration > 0.005:
                    note_events.append((start_time, duration, freq_old, vel_old))

            bend_cents = channel_bends.get(msg.channel, 0.0)
            base_freq = 440.0 * (2 ** ((msg.note - 69) / 12.0))
            freq = base_freq * (2 ** (bend_cents / 1200.0))

            active_notes[key] = (
                current_time,
                freq,
                msg.velocity / 127.0,
            )

        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active_notes:
                start_time, freq, vel = active_notes.pop(key)
                duration = current_time - start_time
                if duration > 0.005:
                    note_events.append((start_time, duration, freq, vel))

    # Flush any notes still active at end of file (missing note_off)
    for key, (start_time, freq, vel) in active_notes.items():
        duration = current_time - start_time
        if duration > 0.005:
            note_events.append((start_time, duration, freq, vel))
    active_notes.clear()

    if not note_events:
        print("⚠️ No notes found to render!")
        return None, None

    # Cap note durations — no single note should ring longer than max_note_dur.
    # Training data can have conversion artifacts (e.g. 106s notes); generated
    # data uses DUR tokens ≤ 16 beats.  A generous 10s cap covers all musical
    # durations at any reasonable tempo while eliminating artifacts.
    max_note_dur = 10.0  # seconds (after speed adjustment)
    capped = 0
    cleaned = []
    for start, dur, freq, vel in note_events:
        if dur > max_note_dur:
            dur = max_note_dur
            capped += 1
        cleaned.append((start, dur, freq, vel))
    note_events = cleaned
    if capped:
        print(f"  Capped {capped} note(s) to {max_note_dur}s max duration")

    # Hard-clip each note to the gap before the next chord onset.
    # Missing or late note_off messages cause notes to ring across chord
    # boundaries producing cluster-like noise.  Since all chords share the
    # same onset grid we can safely clip every note's duration to
    # (next_distinct_onset - this_onset) so notes stop exactly when the
    # next chord begins.
    onsets = sorted({t for t, _, _, _ in note_events})
    onset_to_next = {t: onsets[i + 1] for i, t in enumerate(onsets[:-1])}
    clipped_overlap = 0
    hard_clipped = []
    for start, dur, freq, vel in note_events:
        if start in onset_to_next:
            max_dur = onset_to_next[start] - start
            if dur > max_dur:
                dur = max_dur
                clipped_overlap += 1
        hard_clipped.append((start, dur, freq, vel))
    note_events = hard_clipped
    if clipped_overlap:
        print(f"  Hard-clipped {clipped_overlap} note(s) to chord boundary")

    # --- ADSR Configuration (Dynamic — computed per note from its duration) ---
    # Fixed release tail budget: at most 25% of note duration, capped at 0.4s, min 0.05s.
    # Attack: 5% of duration, capped at 0.08s.
    # Decay: 30% of duration, bringing level down to sustain.
    # Sustain: remainder of gate at a level proportional to note length.
    # This ensures every chord fills its own time slot without bleeding into the next.

    max_release = max(t + d for t, d, _, _ in note_events) + 0.4 + 0.5
    total_duration = max_release
    print(f"Rendering {len(note_events)} notes. Total duration: {total_duration:.2f}s (Speed: {speed}x)")

    # Synthesis
    num_samples = int(total_duration * sample_rate)
    audio = np.zeros(num_samples, dtype=np.float32)

    for start, dur, freq, vel in note_events:
        start_idx = int(start * sample_rate)
        gate_len = int(dur * sample_rate)

        # ── Dynamic ADSR times derived from note duration ──
        # Gate is capped at 80% of the note slot — the remaining 20% is natural
        # silence that separates chords and makes the progression feel articulated.
        effective_dur = dur * 0.80
        gate_len      = int(effective_dur * sample_rate)

        attack_time   = min(0.06, effective_dur * 0.05)
        decay_time    = effective_dur * 0.30
        # Sustain level: kept lower so the sustain phase starts already well
        # below peak — combined with the exponential fade below this produces a
        # natural piano/pluck-like decay instead of a flat organ-like hold.
        sustain_level = np.clip(0.30 + 0.20 * np.log1p(effective_dur) / np.log1p(4.0), 0.18, 0.50)
        # Release: 15% of effective duration — short tail, ends well before next chord
        release_time  = np.clip(effective_dur * 0.15, 0.03, 0.18)
        # Exponential decay rate applied during the sustain phase (per-second).
        # ~1.2 nepers/s ≈ -10 dB/s: audible, natural, but not abrupt.
        sustain_decay_rate = 1.2

        att_len = int(attack_time  * sample_rate)
        dec_len = int(decay_time   * sample_rate)
        rel_len = int(release_time * sample_rate)

        # Buffer for this note (Gate + Release)
        total_note_len = gate_len + rel_len
        env = np.zeros(total_note_len, dtype=np.float32)

        # We use a cursor to fill the buffer sequentially
        cursor = 0

        # 1. Attack Phase
        actual_att = min(att_len, gate_len)
        if actual_att > 0:
            env[0:actual_att] = np.linspace(0.0, 1.0, actual_att, endpoint=False)
            cursor += actual_att

        current_val = 1.0
        if gate_len < att_len:
            current_val = float(actual_att) / att_len

        # 2. Decay Phase
        remaining_gate = gate_len - cursor
        if remaining_gate > 0:
            actual_dec = min(dec_len, remaining_gate)
            decay_curve = np.linspace(current_val, sustain_level, dec_len, endpoint=False)
            env[cursor : cursor + actual_dec] = decay_curve[:actual_dec]
            cursor += actual_dec

            if actual_dec == dec_len:
                current_val = sustain_level
            else:
                current_val = decay_curve[actual_dec - 1]

        # 3. Sustain Phase — exponential decay from current_val toward 0 so the
        # note fades naturally while the key is still "held".
        remaining_gate = gate_len - cursor
        if remaining_gate > 0:
            t_sus = np.arange(remaining_gate) / sample_rate
            env[cursor : cursor + remaining_gate] = current_val * np.exp(-sustain_decay_rate * t_sus)
            current_val = float(env[cursor + remaining_gate - 1])
            cursor += remaining_gate

        # 4. Release Phase — exponential tail from wherever sustain ended.
        if rel_len > 0:
            t_rel = np.arange(rel_len) / sample_rate
            # Fall to ~0.3% of current_val over rel_len (-50 dB) for a clean tail.
            rel_rate = 6.0 / max(release_time, 1e-3)
            env[gate_len : gate_len + rel_len] = current_val * np.exp(-rel_rate * t_rel)

        # Make sure we don't go out of bounds of the main audio buffer
        end_idx = start_idx + len(env)
        if end_idx > num_samples:
              env = env[:num_samples - start_idx]
              end_idx = num_samples

        # Generate waveform oscillator
        t = np.arange(len(env)) / sample_rate
        p = 2 * np.pi * freq * t
        
        # Generate waveform based on selection
        if waveform == 'sine':
            # Pure sine wave
            osc = np.sin(p)
        elif waveform == 'triangle':
            # Bright triangle — odd harmonics with 1/n^1.4 rolloff.
            # Standard triangle uses 1/n^2 which is very dark/opaque;
            # 1/n^1.4 sits between triangle (2) and square (1), adding
            # presence without the harshness of a full square wave.
            max_n = max(1, int((sample_rate * 0.45) / max(freq, 1.0)))
            osc = np.zeros_like(p)
            for n in range(1, max_n + 1, 2):  # 1, 3, 5, ...
                osc += ((-1) ** ((n - 1) // 2)) * np.sin(n * p) / (n ** 1.4)
            osc *= 8 / (np.pi ** 2)
        elif waveform == 'square':
            # Band-limited square via Fourier series.
            # Odd harmonics only, amplitude 1/n.  Include as many as fit below
            # Nyquist — with only ~7 harmonics the tone is too close to a sine
            # because the upper partials carry the characteristic buzz.
            max_n = max(1, int((sample_rate * 0.45) / max(freq, 1.0)))
            osc = np.zeros_like(p)
            for n in range(1, max_n + 1, 2):  # 1, 3, 5, ...
                osc += np.sin(n * p) / n
            osc *= 4 / np.pi
        elif waveform == 'sawtooth':
            # Band-limited sawtooth via Fourier series.
            # All harmonics (even + odd), amplitude 1/n, alternating sign.
            # Cap at 0.20 × Nyquist (~4.4 kHz at 44.1 kHz SR) to tame the
            # harsh high-frequency energy that makes chords sound shrill.
            max_n = max(1, int((sample_rate * 0.20) / max(freq, 1.0)))
            osc = np.zeros_like(p)
            for n in range(1, max_n + 1):
                osc += ((-1) ** (n + 1)) * np.sin(n * p) / n
            osc *= 2 / np.pi
        elif waveform == 'clarinet':
            # Clarinet-like (odd harmonics with specific weights)
            osc = (1.0 * np.sin(p)) - (0.11 * np.sin(3 * p)) + (0.04 * np.sin(5 * p))
        else:
            # Default to sine
            osc = np.sin(p)

        # Equal-loudness compensation (simplified Fletcher–Munson tilt).
        # The ear is far less sensitive to low frequencies at moderate SPL, so
        # bass notes in a chord sound buried against mid/high voices.  We apply
        # a log-frequency gain relative to 1 kHz: low notes get boosted, highs
        # are gently attenuated.  Exponent 0.35 gives a ~+10 dB lift at 100 Hz
        # and ~-3 dB at 4 kHz — enough to mix chords evenly without muddying.
        ref_freq = 1000.0
        freq_gain = (ref_freq / max(freq, 20.0)) ** 0.35
        freq_gain = float(np.clip(freq_gain, 0.65, 3.0))

        # Add to main buffer
        audio[start_idx:end_idx] += osc * env * vel * 0.15 * freq_gain

    # Apply reverb if requested (returns stereo)
    if reverb > 0:
        audio = apply_reverb(audio, sample_rate, reverb)
        # Normalize stereo
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        audio_int16 = (audio * 32767).astype(np.int16)
        # Transpose to (channels, samples) format for audio players
        audio_int16 = audio_int16.T
    else:
        # No reverb - convert mono to stereo
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        audio_mono = (audio * 32767).astype(np.int16)
        # Create stereo in (channels, samples) format
        audio_int16 = np.vstack([audio_mono, audio_mono])
    
    if save_path is not None:
        wav_out = audio_int16.T if audio_int16.ndim == 2 else audio_int16
        wavfile.write(str(save_path), sample_rate, wav_out)
        print(f'Saved: {Path(save_path).name}')

    return audio_int16, sample_rate

def play_audio_data(audio_data, sample_rate):
    """
    Plays audio data using a temporary file and platform-specific command.
    """
    if audio_data is None:
        return

    # Create a temporary file
    # We use delete=False to close it before playing, then delete manually
    # Or rely on the tempfile context manager if the player blocks
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
            temp_filename = tf.name
            # wavfile.write expects (samples, channels); audio_data is (channels, samples)
            wav_data = audio_data.T if audio_data.ndim == 2 else audio_data
            wavfile.write(tf, sample_rate, wav_data)
        
        print(f"Playing...")
        
        # macOS
        if sys.platform == 'darwin':
            subprocess.run(['afplay', temp_filename], check=True)
        # Linux
        elif sys.platform.startswith('linux'):
             subprocess.run(['aplay', temp_filename], check=True)
        # Windows
        elif sys.platform == 'win32':
             # Powershell method or start
             subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{temp_filename}").PlaySync()'], check=True)
        else:
            print("Unsupported platform for playback.")

    except KeyboardInterrupt:
        print("\nPlayback interrupted.")
    finally:
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            # print("Temporary file cleaned up.")

def main():
    parser = argparse.ArgumentParser(description="Play MPE MIDI file with correct microtonal rendering.")
    parser.add_argument("midi_file", help="Path to the MIDI file")
    parser.add_argument("--speed", type=float, default=1.2, help="Playback speed factor (default: 1.2)")
    
    args = parser.parse_args()
    
    audio_data, sr = render_mpe_to_audio_data(args.midi_file, speed=args.speed)
    play_audio_data(audio_data, sr)

if __name__ == "__main__":
    main()
