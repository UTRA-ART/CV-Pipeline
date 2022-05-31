Ramp Detection
==============

*******************************************************************************

Notes on Testing
----------------
* Written and tested in Python 3.8 so far. It will have to be converted to
Python 2.7.
* It has only been tested on a single image so far.

Method
------
This model is trained with a calibration image. It extracts the extremes from
each of the channels in the HLS in the given sample. It will then accept pixels
within a certain tolerance of this range. The calibration image should be
as close (in colour) to the actual competition as possible. In other words, it
should be chosen on the day of competition.

In inference, the model will accept any pixels within a certain tolerance of
the precomputed extremes.

Future work would be to group these pixels (c.f. Jason's changes) and ignore
very small patches of noise.
