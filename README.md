# SignLanguageABCRecognition
A program that uses Keras and OpenCV for understanding ASL ABC.

USE:

Find color of your skin by running SkinColorFinder.py press t when your skin is within designated rectangle and q to finalize.
You will then recieve an upper and lower bound hsv range which you can use to mask your skin.

Run hand tracker and take photo's for your dataset starting on your first s click and ending on your second.
Photos will be saved to designated path so change the path to match your machine.

Run HandTracker.py once again this time you should have your webcam running and you should be able to sign and recieve an audio guess from the machine.

Any time you wish to safely close press 'q'.
