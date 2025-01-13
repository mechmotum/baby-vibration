import matplotlib.pyplot as plt


def plot_frequency_spectrum(freq, amp, ax=None, plot_kw={}):
    """Returns plot of the amplitude versus frequency."""
    if ax is None:
        fig, ax = plt.subplots(layout='constrained')
    ax.plot(freq, amp, **plot_kw)
    ax.set_ylim((0.0, 1.0))
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude [m/s$^2$]')
    ax.grid()
    return ax


def plot_iso_weights(iso_filter_df_01, iso_filter_df_02):
    ax1 = iso_filter_df_01.plot(subplots=True)
    ax1[0].set_title('iso_filter_01')
    ax2 = iso_filter_df_02.plot(subplots=True)
    ax2[0].set_title('iso_filter_02')
    return ax1, ax2
