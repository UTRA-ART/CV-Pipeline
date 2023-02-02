# Tusimple Scripts
The script used to add solid white lane in tusimple dataset
## Instruction
1. Download [tusimple dataset](https://github.com/TuSimple/tusimple-benchmark/issues/3)
2. Place lanes.py in the tusimple folder and run lanes.py
3. Two new folder will be create in the tusimple folder
    * "adjusted" folder holds all tusimple images with solid white lane
    * "lanes" folder holds labels for tusimple images
### Remark
* Change [name](https://github.com/UTRA-ART/CV-Pipeline/blob/f1c8dfd43a4212f555e32cc7834fa54a6da68b76/src/lane_detection/tusimple-lane-data/lanes.py#L8) for of all label data json file to obtain adjusted image for all tusimple data
* Comment out [cv2.drawContours](https://github.com/UTRA-ART/CV-Pipeline/blob/f1c8dfd43a4212f555e32cc7834fa54a6da68b76/src/lane_detection/tusimple-lane-data/lanes.py#L71) to obtain labels in different style
