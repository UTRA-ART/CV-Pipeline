The purpose of this branch is to set up YOLOv4 for use in object detection.

To run demo.py

1) Pass in the weights file location
2) Pass in the image file loaction
3) Turn off cuda for now
4) pass in --save-text to save labels to .txt files for each frame
    - <class> <x> <y> <width> <height>
    - x & y are the location of the center
    - all values are floating point, relative sizes to the actual image


Changes for .cfg file:

1) Change the number of classes in all the 'yolo' layer
2) Change the number of filters in the 'convolotion' layer before 'yolo'
    - filter = (num_of_classes + 4 + 1) * 3
3)