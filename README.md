# TomakunEEG
GUI eeg analyzer specific to Hyperscanning project
  
  
## Description  
This is the Python-GUI application for analyzing EEG.  
Specially **Hyperscanning project** that is based on two subjects' eeg data can be analyzed easily by this app.  
Phase-synchrony-index (PSI) or mutual information entropy analysis... are installed now.  
Sorry, but now **only Brainvision eeg file (.eeg .vhdr .vmrk) is supported**.
  
  
## Requirement
This app is based on mne-python, pyQt and other libraries.  
- So first, you should install **[mne-python](https://mne.tools/stable/install/index.html)** (Anaconda virtual environment is recommended).  
  
- Next, please install other libraries in your environment from your commandline.  
```
conda install pyqt pyqtgraph seaborn
```
  
If you cannnot run this app normaly, please update these libraries by **conda update** command.  
  
  
  
  
And please note the following.  
  
- When you'd like to use MATLAB Adjust algorithm binding. you should also install [MATLAB engine API](https://jp.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) and install [eeglab](https://sccn.ucsd.edu/eeglab/index.php),  [Adjust algorithm](https://www.nitrc.org/projects/adjust/) from their homepage. So when you use these bindings, you should give attention to their license.

- Current source density (csd) code is a bit customization of [pyCSD](https://github.com/nice-tools/pycsd). So when you use the csd function, please check the applicable web-page.
  
  
