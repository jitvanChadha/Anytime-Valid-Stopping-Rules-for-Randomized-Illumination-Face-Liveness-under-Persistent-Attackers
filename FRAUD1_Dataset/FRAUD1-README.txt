This is file README.txt - 15-Oct-2014

Title:
------
FRAUD1 - Face Replay Attack UQ Dataset, Version 1

Location:
---------
http://www.itee.uq.edu.au/sas/datasets/

Purpose:
--------
The digital representation of captured biometric signals may be injected 
into the system (replay attack) thus bypassing any liveness testing.

This FRAUD1 Dataset is the first practical attempt to counter replay 
attacks for a face recognition system that uses uncontrolled consumer 
devices.  The final operation of the system is designed to illuminate 
the face of the user with colours displayed on the computer screen, and 
then using the camera to capture the face images that will be used for 
the face recognition system.  In addition to liveness, antispoofing 
tests, the colour reflection might be observed in the face.

If the reflection sequence matches the colours that were displayed, it 
can be assumed that the video waas created at hte time that the colours 
were displayed.

In the case of this dataset, the videos have been previously captured at 
the time the nominated colours were displayed.  In its final form, it is 
expected that the colours will be nominated in real-time.

Researchers are invited to use this dataset to improve techniques for 
classifying colour reflection from objects, and to validate the results 
obtained in our experiments.

Description:
------------
The FRAUD1 dataset contains the following items:

1.  Videos of objects using a computer tablet screen that displays 
    different colours to illuminate them.  The objects consist of a 
    blank piece of white paper, a printed face on a piece of paper, and
    ten live subjects.  The colours used to illuminate the objects are 
    chosen randomly from the following:
    1  Black    RGB(0, 0, 0)
    2  Red      RGB(255, 0, 0)
    3  Green    RGB(0, 255, 0)
    4  Blue     RGB(0, 0, 255)
    5  Magenta  RGB(255, 0, 255)
    6  Cyan     RGB(0, 255, 255)
    7  Yellow   RGB(255, 255, 0)

    The white paper and paper face objects have 150 videos, and each of 
    the ten live subjects have 15 videos.  Each video consists of the 
    following:
      - A number of frames of no illumination (blank screen)
      - Four cycles of:
        . Approximately three frames of colour illumination
        . Approximately three frames of no illumination (blank screen)
      - A number of frames of no illumination (blank screen)

    The actual number of frames of illumination and no illumination will 
    vary slightly due to different responses from the camera device.

    Each illuminated object video set is captured twice - the first using 
    the in-built Microsoft LifeCam on a Microsoft Suface Pro 2 tablet, 
    and the second using an external Logitech Pro 9000 webcam connected 
    to the Microsoft Surface Pro 2 tablet.
    
    Images are captured using an application that was developed on Ubuntu
    Linux 13.10 and OpenCV 2.4.8 running on a VMware Player 6.0.2 virtual
    machine.  The VM is hosted by a Microsoft Surface Pro 2 tablet running
    Windows 8.1 Pro.  Individual frames are saved using the DIVX codec,
    with resolution 640x480@30fps.

2.  Ground truth data for each video, detailing the colours that were 
    used to illuminate the object.  Included in this file is an integer 
    representation of the colour sequence, to assist simple comparison of 
    results.  This integer is calculated using the index values above, 
    and the following formula:
    
    Result = ( ( ( Colour1 * 7) + Colour2) * 7 + Colour3) * 7 + Colour4

    For example, for the colour sequence "Red Magenta Green Green", the 
    result is:

    955 = ( ( ( 2 * 7) + 5) * 7 + 3) * 7 + 3

3.  The results obtained using the techniques developed by the authors 
    for classifying the colour reflections.  These are provided for 
    comparison purposes only.

Licence:
--------
Please see the file FRAUD1-LICENCE.txt.

Journal Article Title:
----------------------
Face Recognition on Consumer Devices: Reflections on Replay Attacks

Abstract:
---------
Widespread deployment of biometric systems supporting
consumer transactions is yet to occur. Smart consumer
devices, such as tablets and phones, have the potential to act
as biometric readers authenticating user transactions. However,
the use of these devices in uncontrolled environments is highly
susceptible to replay attacks where these biometric data are
captured and replayed at a later time. Current approaches to
counter replay attacks in this context are inadequate. In order
to show this, we demonstrate a simple replay attack that is 100%
effective against a recent state-of-the-art face recognition system;
this system was specifically designed to robustly distinguish
between live people and spoofing attempts such as photographs.
This paper proposes an approach to counter replay attacks
for face recognition on smart consumer devices using a noninvasive
challenge and response technique. The image on the
screen creates the challenge, and the dynamic reflection from
the person’s face as they look at the screen forms the response.
The sequence of screen images and their associated reflections
digitally watermarks the video. By extracting the features from
the reflection region, it is possible to determine if the reflection
matches the sequence of images that were displayed on the screen.
Experiments indicate that the face reflection sequences can be
classified under ideal conditions with a high degree of confidence.
These encouraging results may pave the way for further studies
in the use of video analysis for defeating biometric replay attacks
on consumer devices.

Colour Reflection Dataset Download:
-----------------------------------
http://dropbox.eait.uq.edu.au/e3blovel/FRAUD1_Dataset.zip
(207 Megabytes)

Ethical Clearance:
------------------
The experiment data involving humans was performed in accordance with 
the School of ITEE Ethics Approval Number:  EC201303SMI.A1.

Contact:
--------
Further enquiries can be made to:
{danny döt smith ät uq döt net döt au}, 
{a döt wiliem ät uq döt edu döt au}, or
{lovell ät itee döt uq döt edu döt au}.