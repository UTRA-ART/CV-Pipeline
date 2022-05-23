import os

input_directory = "C:/Users/ammar/Documents/CodingProjects/ART/CV-Pipeline/YOLOv4/PyTorch_YOLOv4-train/inference/test2"

command = 'python detect.py --weights "runs/train/exp125/weights/best.pt" --output "inference/output_test2/" --source '

for filename in os.listdir(input_directory):
    f = os.path.join(input_directory, filename)

    if os.path.isfile(f):
        os.system(command + '"' + f + '"' + " --device 1 --save-txt")
        print(command + '"' + f + '"' + " --device 1 --save-txt")
