"""MIDI Visualization for 53-TET MPE files using Plotly"""
import plotly.graph_objects as go
import mido
from pathlib import Path


def visualize_midi(midi_path, speed=1.5, max_duration=30):
    """
    Create a piano roll visualization of MIDI file with MPE pitch bends.
    
    Args:
        midi_path: Path to MIDI file
        speed: Speed multiplier for time axis (default 1.5)
        max_duration: Maximum duration to show in seconds (default 30, None for full)
    
    Returns:
        Plotly Figure object or None if no notes found
    """
    midi_path = Path(midi_path)
    if not midi_path.exists():
        print(f"Error: File not found: {midi_path}")
        return None
    
    mid = mido.MidiFile(midi_path)
    note_events = []
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
                start_time, pitch_old, vel_old = active_notes.pop(key)
                duration = current_time - start_time
                if duration > 0.005:
                    note_events.append({
                        'start': start_time,
                        'end': current_time,
                        'duration': duration,
                        'pitch': pitch_old,
                        'velocity': vel_old,
                        'base_note': key[1]
                    })

            bend_cents = channel_bends.get(msg.channel, 0.0)
            base_note = msg.note
            actual_pitch = base_note + (bend_cents / 100.0)  # Convert cents to semitones
            
            active_notes[key] = (
                current_time,
                actual_pitch,
                msg.velocity / 127.0,
            )
        
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active_notes:
                start_time, pitch, vel = active_notes.pop(key)
                duration = current_time - start_time
                if duration > 0.005:
                    note_events.append({
                        'start': start_time,
                        'end': current_time,
                        'duration': duration,
                        'pitch': pitch,
                        'velocity': vel,
                        'base_note': key[1]
                    })
    
    if not note_events:
        print("⚠️ No notes found!")
        return None
    
    # Apply duration filter if specified
    if max_duration:
        note_events = [n for n in note_events if n['start'] < max_duration]
        if not note_events:
            print(f"⚠️ No notes found in first {max_duration} seconds!")
            return None
    
    print(f"Visualizing {len(note_events)} notes")
    
    # Create figure
    fig = go.Figure()
    
    # Add notes as rectangular bars (piano roll style)
    note_height = 0.8  # Height of each note bar
    bend_height = 0.15  # Height of pitch bend indicator (thinner)
    
    for note in note_events:
        color_intensity = note['velocity']
        deviation = note['pitch'] - note['base_note']
        
        # Check if note has pitch bend (MPE) - threshold of 0.01 semitones (~1 cent)
        is_mpe = abs(deviation) > 0.01
        
        # Always draw the blue MIDI note base
        blue_alpha = 0.5 + 0.3*color_intensity
        y_bottom_base = note['base_note'] - note_height/2
        y_top_base = note['base_note'] + note_height/2
        
        fig.add_trace(go.Scatter(
            x=[note['start'], note['end'], note['end'], note['start'], note['start']],
            y=[y_bottom_base, y_bottom_base, y_top_base, y_top_base, y_bottom_base],
            fill='toself',
            fillcolor=f'rgba({int(100 + 100*color_intensity)}, {int(150 + 80*color_intensity)}, 255, {blue_alpha})',
            line=dict(color='rgba(70,130,220,0.4)', width=1),
            hovertemplate=(
                f"<b>MIDI Note:</b> {note['base_note']}<br>"
                f"<b>Actual Pitch:</b> {note['pitch']:.2f}<br>"
                f"<b>Pitch Bend:</b> {deviation*100:.1f} cents<br>"
                f"<b>Start:</b> {note['start']:.2f}s<br>"
                f"<b>Duration:</b> {note['duration']:.2f}s<br>"
                f"<b>Velocity:</b> {note['velocity']:.2f}<extra></extra>"
            ),
            showlegend=False,
            mode='lines'
        ))
        
        # If there's pitch bend, draw connector line and orange indicator at the actual pitch
        if is_mpe:
            # Draw thin gray connector line from MIDI note center to pitch bend center
            midi_center = note['base_note']
            bend_center = note['pitch']
            
            # Add connector at start of note
            fig.add_trace(go.Scatter(
                x=[note['start'], note['start']],
                y=[midi_center, bend_center],
                mode='lines',
                line=dict(color='rgba(120, 120, 120, 0.3)', width=1),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Add connector at end of note
            fig.add_trace(go.Scatter(
                x=[note['end'], note['end']],
                y=[midi_center, bend_center],
                mode='lines',
                line=dict(color='rgba(120, 120, 120, 0.3)', width=1),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Draw orange pitch bend indicator
            orange_alpha = 0.7 + 0.2*color_intensity
            y_bottom_bend = note['pitch'] - bend_height/2
            y_top_bend = note['pitch'] + bend_height/2
            
            fig.add_trace(go.Scatter(
                x=[note['start'], note['end'], note['end'], note['start'], note['start']],
                y=[y_bottom_bend, y_bottom_bend, y_top_bend, y_top_bend, y_bottom_bend],
                fill='toself',
                fillcolor=f'rgba(255, {int(140 + 40*color_intensity)}, 0, {orange_alpha})',
                line=dict(color='rgba(255,100,0,0.8)', width=1.5),
                hovertemplate=(
                    f"<b>🎯 MPE Pitch Bend</b><br>"
                    f"<b>Base MIDI:</b> {note['base_note']}<br>"
                    f"<b>Bent Pitch:</b> {note['pitch']:.2f}<br>"
                    f"<b>Deviation:</b> {deviation*100:.1f} cents ({'+' if deviation > 0 else ''}{deviation:.3f} semitones)<br>"
                    f"<b>Velocity:</b> {note['velocity']:.2f}<extra></extra>"
                ),
                showlegend=False,
                mode='lines'
            ))
    
    # Calculate axis ranges
    max_time = max(n['end'] for n in note_events)
    min_pitch = min(n['pitch'] for n in note_events)
    max_pitch = max(n['pitch'] for n in note_events)
    
    # Update layout
    duration_text = f" (first {max_duration}s)" if max_duration else ""
    fig.update_layout(
        title=f"<b>MIDI Piano Roll{duration_text}</b><br><sub>{midi_path.name} | Speed: {speed}x | Notes: {len(note_events)}</sub>",
        xaxis_title="Time (seconds)",
        yaxis_title="Pitch (MIDI Note Number + Microtonal Deviation)",
        height=500,
        hovermode='closest',
        plot_bgcolor='#f8f9fa',
        xaxis=dict(
            gridcolor='#dee2e6',
            range=[0, max_time * 1.02],
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='#dee2e6',
            range=[min_pitch - 2, max_pitch + 2],
            showgrid=True
        ),
        margin=dict(l=60, r=20, t=80, b=60)
    )
    
    return fig
