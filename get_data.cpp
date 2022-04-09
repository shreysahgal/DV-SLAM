#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>
#include <opencv2/core/core_c.h>
#include <opencv4/opencv2/opencv.hpp>

using namespace std;

vector<string> readLine(string filename){
	vector<string> valid_commands;
	ifstream word_file(filename);
	if(word_file.is_open()){
		string word;
		while(getline(word_file, word)){
			valid_commands.push_back(word);
		}
		word_file.close();
	}
	return valid_commands;
}

vector<double> splitString(string s){

	std::string delimiter = ", ";
	size_t pos = 0;
	std::string token;
	vector<double> v;

	pos = s.find(delimiter);
	while((pos = s.find(delimiter)) != std::string::npos){
		token = s.substr(0, pos);
		double num = stod(token);
		v.push_back(num);
		s.erase(0, pos + delimiter.length());
	}
	double num = stod(s);
	v.push_back(num);
	return v;
}

void get_data(vector<string> poses, vector<vector<double>> &DeepVO_output)
{
    for (auto pose : poses){
            
			DeepVO_output.push_back(splitString(pose));

		}
} 

cv::Mat Rotation(double thetax, double thetay, double thetaz){

    cv::Mat Rx = cv::Mat::zeros(3,3,CV_64F);
    Rx.at<double>(0,0) = 1.0;
    Rx.at<double>(1,1) = cos(thetax);
    Rx.at<double>(1,2) = -1.0 * sin(thetax);
    Rx.at<double>(2,1) = sin(thetax);
    Rx.at<double>(2,2) = cos(thetax);

    cv::Mat Ry = cv::Mat::zeros(3,3,CV_64F);
    Ry.at<double>(0,0) = cos(thetay);
    Ry.at<double>(0,2) = sin(thetay);
    Ry.at<double>(1,1) = 1.0;
    Ry.at<double>(2,0) = -1.0 * sin(thetay);
    Ry.at<double>(2,2) = cos(thetay);

    cv::Mat Rz = cv::Mat::zeros(3,3,CV_64F);
    Rz.at<double>(0,0) = cos(thetaz);
    Rz.at<double>(0,1) = -1.0 * sin(thetaz);
    Rz.at<double>(1,0) = sin(thetaz);
    Rz.at<double>(1,1) = cos(thetaz);
    Rz.at<double>(2,2) = 1.0;
    
    cout << "Rx = " << endl << " "  << Rx << endl << endl;
    cout << "Ry = " << endl << " "  << Ry << endl << endl;
    cout << "Rz = " << endl << " "  << Rz << endl << endl;

    cv::Mat R = Ry * (Rx * Rz);
    return R;

}

int main ()
{
    vector<string> poses = readLine("/home/yclai/Desktop/DeepVO/DeepVO-pytorch/result/out_07.txt");
    vector<vector<double>> DeepVO_output;
    get_data(poses, DeepVO_output);
    cv::Mat R;
    for(int i = 0; i < DeepVO_output[0].size(); i++){

        cout << "value "<< i << ": " << DeepVO_output[1][i]<<endl;

    }
    
    vector<double> rotation_matrix;
    vector<double> translation_vector;
    // R = Rotation(-0.15640373, 0.00594132, -0.00390095);
    R = Rotation(0.00594132, -0.15640373, -0.00390095);
    // R = Rotation(-0.00390095, 0.00594132, -0.15640373);
    cout << "R = " << endl << " "  << R << endl << endl;
    return 0;
}