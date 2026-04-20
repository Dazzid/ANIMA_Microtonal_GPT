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

def render_mpe_to_audio_data(midi_path, sample_rate=44100, speed=1.2, waveform='sine', reverb=0, save_path=None):
    """
    Renders MPE MIDI to audio data (numpy array) with correct timing, pitch bends, and ADSR envelope.
    
    Args:
        midi_path: Path to MIDI file
        sample_rate: Audio sample rate (default 44100)
        speed: Playback speed multiplier (default 1.2)
        waveform: Waveform type - 'sine', 'triangle', 'square', 'sawtooth', or 'clarinet' (default 'sine')
        reverb: Reverb amount as percentage (0-100), 0=no reverb, 100=maximum reverb
        save_path: Optional path to save .wav file to disk
    
    Returns: (audio_data_int16, sample_rate)
    """
    print(f"play_mpe | waveform: {waveform}, reverb: {reverb}%")
    midi_path = Path(midi_path)
    if not midi_path.exists():
        print(f"Error: File not found: {midi_path}")
        return None, None

    print(f"Loading {midi_path.name}...")
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
            # Band-limited triangle via Fourier series.
            # Odd harmonics only, amplitude 1/n^2, alternating sign.
            max_n = max(1, int((sample_rate * 0.45) / max(freq, 1.0)))
            osc = np.zeros_like(p)
            for n in range(1, max_n + 1, 2):  # 1, 3, 5, ...
                osc += ((-1) ** ((n - 1) // 2)) * np.sin(n * p) / (n ** 2)
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
            # Include as many harmonics as fit below Nyquist to get the
            # characteristic bright/buzzy timbre without aliasing.
            max_n = max(1, int((sample_rate * 0.45) / max(freq, 1.0)))
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
