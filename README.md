# DV-SLAM
## Group 1 Edit + Summary
rob 530 final project group 9
## Group 2 Edit
## Results
![alt text](https://github.com/shreysahgal/DV-SLAM/blob/main/media/res1.JPG)

The results show a substantial improvement with loop closures when compared to pure Deep VO output. These results are from running our program on KITTI rgb odometry dataset 07. To replicate them ...

Obtained Results             |  ORB-SLAM Comparison
:-------------------------:|:-------------------------:
![](https://github.com/shreysahgal/DV-SLAM/blob/main/media/res2.png)  |  ![](https://github.com/shreysahgal/DV-SLAM/blob/main/media/resulting3.png)
RMSE: 7.271161 m| RMSE: 3.194608 m
MSE: 6.651275 m| MSE: 2.960466 m
Median Error: 5.993425 m| Median Error: 2.950503 m
Min Error: 2.430470 m| Min Error: 0.476634 m
Max Error: 15.652031 m| Max Error: 6.092447 m
Standard Deviation: 5.993425 m| Standard Deviation: 1.200484 m

The results above show our DV-algorithm with a side by side comparison with ORB-SLAM modified by the Horn [[1]](#1) trajectory alignment algorithm. In comparison with a widely used monocular SLAM solution, ORB-SLAM2, our DV-SLAM does not perform as well across all metrics listed. However, we believe that the modification of other algorithm's front-ends, such as ORB-SLAM2, with the deepVO model has the potential to generate results surpassing either of the above methods. We leave this to future works.

To replicate the above results simply clone the repository, and run the deepVO_stats.py file and orb-slam alignment.py files. DeepVO_stats.py generates the error statistics and graph for DV-SLAM, and orb-slam alignment.py runs the results of ORB-SLAM through the Horn [[1]](#1) algorithm before before generating the error statistics and graph. All necessary files are provided. If you wish to generate your own, KeyFrameTrajectory.txt can be generated by running ORB-SLAM2 [[2]](#2) on the KITTI [[3]](#3) odometry dataset. The gtsam7.txt file (results of DV-SLAM) can be generated by ...
## References
<a id="1">[1]</a> 
Horn, Berthold K.P. (1987).
Closed-Form Solution of Absolute Orientation Using Unit Quaternions
Journal of the Optical Society of America A, vol. 4, no. 4, 629.

<a id="2">[2]</a> 
Mur-Artal, Raul (2016).
ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras
IEEE Transactions on Robotics.

<a id="3">[3]</a> 
Geiger et al. (2013).
Vision meets robotics: The KITTI dataset
International Journal of Robotics Research, vol. 32, 1231-1237.
