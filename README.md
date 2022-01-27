# 2021-BrainTopogr-fNIRS-AV

The files ``av.py`` and ``mcgurk.py`` process the raw data and produce
images used in paper figures. A suitable ``conda`` environment can be
set up to run these scripts with something like::

```Console
(base) $ conda install -n mne -c conda-forge mne openpyxl mne-nirs
(base) $ conda activate mne
(mne) $ python -i av.py
```
