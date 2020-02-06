# raw.plot()に嫌気が差したので全部イチから作る。そうするとビデオと同期もできる。
# scrollbarなど考えるとpyqtに埋め込んだ方が簡単に作れる。

import mne
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'


def raw_interactive_plot(raw):
    """
    arrをplotするのもやまやまだが、rawにしとかないと、annotationなどの情報がないので、rawにしとく。
    """
    data = raw.get_data()[:10, :100]  # arr
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_axes([0.05, 0.1, 0.93, 0.87])
    ax2 = fig.add_axes([0.05, 0.06, 0.93, 0.03])



    ax1.plot(data.T)



    plt.show()


if __name__ == "__main__":
    raw = mne.io.read_raw_fif("C:/Users/dokki/OneDrive/Desktop/EEG_data/EEG_test_20190602_animallab/phase_analysis/animallab0008sit_close_preprocessed.fif")
    raw_interactive_plot(raw)
