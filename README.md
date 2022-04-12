# DV-SLAM
## Group 1 Edit + Summary
rob 530 final project group 9
## Group 2 Edit
## Results
![alt text](https://github.com/shreysahgal/DV-SLAM/blob/main/media/res1.JPG)

The results show a substantial improvement with loop closures versus pure Deep VO output. These results are from running our program on KITTI rgb odometry dataset 07. To replicate them ...

Obtained Results             |  ORB-SLAM Comparison
:-------------------------:|:-------------------------:
![](https://github.com/shreysahgal/DV-SLAM/blob/main/media/res2.png)  |  ![](https://github.com/shreysahgal/DV-SLAM/blob/main/media/resulting3.png)
RMSE: 7.271161 m| RMSE: 3.194608 m
MSE: 6.651275 m| MSE: 2.960466 m
Median Error: 5.993425 m| Median Error: 2.950503 m
Min Error: 2.430470 m| Min Error: 0.476634 m
Max Error: 15.652031 m| Max Error: 6.092447 m
Standard Deviation: 5.993425 m| Standard Deviation: 1.200484 m

The results above show our DV-algorithm with a side by side comparison with ORB-SLAM modified by the Horn[[1]](#1) trajectory alignment algorithm. In comparison with a widely used monocular SLAM solution, ORB SLAM, our DV-SLAM does not perform as well across all metrics listed. However, we believe that the modification of other algorithm's front-ends, such as ORB-SLAM, with the deepVO model has the potential to generate results surpassing either of the above methods. We leave this to future works.

To replicate the above results

## References
<a id="1">[1]</a> 
Horn, Berthold K.P. (1987).
Closed-Form Solution of Absolute Orientation Using Unit Quaternions
Journal of the Optical Society of America A, vol.4, no. 4, 629.
