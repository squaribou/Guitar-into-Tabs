# Guitar-into-Tabs
Convert a guitar music into tabs and music score

The scale use here is the equal temperament

Magnitude calcul of the spectrum:<br>
We take the module of the spectrum (with complex values), and only the part with positive frequencies.
The magnitude is normalized by /N and the mirror negative part is ad by *2.


To determined a note, we use the pyin function in the librosa library. It returns us the fundamental the different 
frame of the signal (every 512 points) with probabilities about it. We keep the more likely to be the fudammental and then 
take the median and search the closest note to it.

To detecte a change of note, we use the onset_detect of the librosa library
