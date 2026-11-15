This is file FRAUD1-Training-Data-README.txt - 10-Oct-2014
http://www.itee.uq.edu.au/sas/datasets

The training data use in the SVM for classifying reflections found
in the FRAUD1 (Face Replay Attack UQ Dataset, Version1) is structured 
as follows:

<data-line> ::= <colour><SPACE>
                1:<red-percent><SPACE>
                2:<yellow-percent><SPACE>
                3:<green-percent><SPACE>
                4:<cyan-percent><SPACE>
                5:<blue-percent><SPACE>
                6:<magenta-percent><SPACE>
                7:<black-percent><EOL>

<colour> ::= "1" | "2" | "3" | "4" | "5" | "6" | "7"

<SPACE> ::= " "

<black-percent> ::= <integer>
<red-percent> ::= <float>
<yellow-percent> ::= <float>
<green-percent> ::= <float>
<cyan-percent> ::= <float>
<blue-percent> ::= <float>
<magenta-percent> ::= <float>

<integer> ::= <digits>
<float> ::= <digits>.<digits>
<digits> ::= <digits><digit> | ""
<digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | '6" | "7" | "8" | "9"



<colour> represents the ground truth classified colour.  This is
defined as:
  1:  Black
  2:  Red
  3:  Green
  4:  Blue
  5:  Magenta
  6:  Cyan
  7:  Yellow

The <black-percent> is calculated as a percentage of pixels that have
a value of 0-2 in the V channel of the HSV colour space.

Each colour percent is calculated as a percentage of pixels that have
the following properties:
  - Value >2 in the V channel of the HSV colour space; and
  - Value in the Hue channel of the HSV colour space, as follows:
      0: Red
     30: Yellow
     60: Green
     90: Cyan
    120: Blue
    150: Magenta
The percentage is calculated by dividing the number of pixels for
each colour channel, by the total of pixels calculated in each
of the six colour channels.  That is, the sum of <red-percent> through
to <magenta-percent> will always equal 100%.
