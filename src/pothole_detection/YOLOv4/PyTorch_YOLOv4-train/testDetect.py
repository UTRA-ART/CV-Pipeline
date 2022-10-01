import os

input_directory = "C:\\Users\\ammar\\Documents\\CodingProjects\\ART\\CV-Pipeline\\src\\pothole_detection\\YOLOv4\\PyTorch_YOLOv4-train\\inference\\test5"

command = 'python detect.py --weights "runs/train/exp130/weights/best.pt" --output "inference/test5-output/" --source '

for filename in os.listdir(input_directory):
    f = os.path.join(input_directory, filename)

    if os.path.isfile(f):
        os.system(command + '"' + f + '"' +
                  " --device 1 --save-txt --conf-thres 0.6")
        print(command + '"' + f + '"' + " --device 1 --save-txt --conf-thres 0.6")
