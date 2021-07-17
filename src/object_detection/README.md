# CV-Object-Detection

To use this library:
1. Create a Frame_Handler object
1. Set the camera callback to send the frame to Frame_Handler.update_from_camera(frame)
1. Call Frame_Handler.process_frame() at whatever frequency desired. Parameters for temporal consistency will need to be optimized for any given run rate.
1. Cropped images to send for classification can be found under Frame_Handler.\<object type\>
1. (x, y, z) coordinates for detected objects can be found under Frame_Handler.\<object type\>_loc
  
Updates to come:
* Integration with classification models 
* In house Haar Cascade models for detection of competition specific objects 
