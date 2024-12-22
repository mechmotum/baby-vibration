import matplotlib.pyplot as plt


def plot_frequency_spectrum(freq, amp, rms, sample_rate, ax=None):
    """Returns plot of the amplitude versus frequency for the freqeuncy range
    of the sample rate / 2."""
    if ax is None:
        fig, ax = plt.subplots(layout='constrained')
    ax.plot(freq, amp)
    ax.axhline(rms, color=ax.get_lines()[0].get_color())
    ax.set_ylim((0.0, 1.0))
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [m/s/s]')
    ax.grid()
    return ax
