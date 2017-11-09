import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import json

# Read the input file
sampling_freq, audio = wavfile.read('input_read.wav')
# Print the params
print '\nShape:', audio.shape
print 'Datatype:', audio.dtype
print 'Duration:', round(audio.shape[0] / float(sampling_freq),3), 'seconds'
# Normalize the values
audio = audio / (2.**15)
# Extract first 30 values for plotting
audio = audio[:30]
# Build the time axis
x_values = np.arange(0, len(audio), 1) / float(sampling_freq)
# Convert to seconds
x_values *= 1000
# Plotting the chopped audio signal
plt.plot(x_values, audio, color='black')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Audio signal')
plt.show()

# Read the input file
sampling_freq, audio = wavfile.read('input_freq.wav')
# Normalize the values
audio = audio / (2.**15)
# Extract length
len_audio = len(audio)
# Apply Fourier transform
transformed_signal = np.fft.fft(audio)
half_length = int(np.ceil((len_audio + 1) / 2.0))
transformed_signal = abs(transformed_signal[0:half_length])
transformed_signal /= float(len_audio)
transformed_signal **= 2
# Extract length of transformed signal
len_ts = len(transformed_signal)
# Take care of even/odd cases
if len_audio % 2:
    transformed_signal[1:len_ts] *= 2
else:
    transformed_signal[1:len_ts-1] *= 2
# Extract power in dB
power = 10 * np.log10(transformed_signal)
# Build the time axis
x_values = np.arange(0, half_length, 1) * (sampling_freq / len_audio) / 1000.0
# Plot the figure
plt.figure()
plt.plot(x_values, power, color='black')
plt.xlabel('Freq (in kHz)')
plt.ylabel('Power (in dB)')
plt.show()

# File where the output will be saved
output_file = 'output_generated.wav'
# Specify audio parameters
duration = 3 # seconds
sampling_freq = 44100 # Hz
tone_freq = 587
min_val = -2 * np.pi
max_val = 2 * np.pi
# Generate audio
t = np.linspace(min_val, max_val, duration * sampling_freq)
audio = np.sin(2 * np.pi * tone_freq * t)
# Add some noise
noise = 0.4 * np.random.rand(duration * sampling_freq)
audio += noise
# Scale it to 16-bit integer values
scaling_factor = pow(2,15) - 1
audio_normalized = audio / np.max(np.abs(audio))
audio_scaled = np.int16(audio_normalized * scaling_factor)
# Write to output file
wavfile.write(output_file, sampling_freq, audio_scaled)
# Extract first 100 values for plotting
audio = audio[:100]
# Build the time axis
x_values = np.arange(0, len(audio), 1) / float(sampling_freq)
# Convert to seconds
x_values *= 1000
# Plotting the chopped audio signal
plt.plot(x_values, audio, color='black')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Audio signal')
plt.show()

# Synthesize tone
def synthesizer(freq, duration, amp=1.0, sampling_freq=44100):
    # Build the time axis
    t = np.linspace(0, duration, duration * sampling_freq)
    # Construct the audio signal
    audio = amp * np.sin(2 * np.pi * freq * t)
    return audio.astype(np.int16)

if __name__=='__main__':
    tone_map_file = 'tone_freq_map.json'
    # Read the frequency map
    with open(tone_map_file, 'r') as f:
        tone_freq_map = json.loads(f.read())
    # Set input parameters to generate 'G' tone
    input_tone = 'G'
    duration = 2  # seconds
    amplitude = 10000
    sampling_freq = 44100  # Hz
    # Generate the tone
    synthesized_tone = synthesizer(tone_freq_map[input_tone],duration, amplitude, sampling_freq)
    # Write to the output file
    wavfile.write('output_tone.wav', sampling_freq, synthesized_tone)
    # Tone-duration sequence
    tone_seq = [('D', 0.3), ('G', 0.6), ('C', 0.5), ('A', 0.3),('Asharp', 0.7)]
    # Construct the audio signal based on the chord sequence
    output = np.array([])
    for item in tone_seq:
        input_tone = item[0]
        duration = item[1]
        synthesized_tone = synthesizer(tone_freq_map[input_tone],duration, amplitude, sampling_freq)
        output = np.append(output, synthesized_tone, axis=0)
    # Write to the output file
    wavfile.write('output_tone_seq.wav', sampling_freq, output)