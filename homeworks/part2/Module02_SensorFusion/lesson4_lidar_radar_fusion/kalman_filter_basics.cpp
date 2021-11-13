// Write a function 'filter()' that implements a multi-
// dimensional Kalman Filter for the example given
//============================================================================
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>

using namespace std;
using namespace Eigen;

//Kalman Filter variables
VectorXd x;	// object state
MatrixXd P;	// object covariance matrix
VectorXd u;	// external motion
MatrixXd F; // state transition matrix
MatrixXd H;	// measurement matrix
MatrixXd R;	// measurement covariance matrix
MatrixXd I; // Identity matrix
MatrixXd Q;	// process covariance matrix

vector<VectorXd> measurements;
void filter(VectorXd &x, MatrixXd &P);

int main() {
	//design the KF with 1D motion
	x = VectorXd(2);
	x << 0, 0;         // (position, velocity)

	P = MatrixXd(2, 2);
	P << 1000, 0, 0, 1000;

	u = VectorXd(2);
	u << 0, 0;

	F = MatrixXd(2, 2);
	F << 1, 1, 0, 1;   // (1, delta_t, 0, 1)

	H = MatrixXd(1, 2);
	H << 1, 0;

	R = MatrixXd(1, 1);
	R << 1;

	I = MatrixXd::Identity(2, 2);  // (1, 0, 0, 1)

	Q = MatrixXd(2, 2);
	Q << 0, 0, 0, 0;

	//create a list of measurements
	//测量值只有positon，没有velocity因为传感器测量不到速度
	VectorXd single_meas(1);
	single_meas << 1;
	measurements.push_back(single_meas);
	single_meas << 2;
	measurements.push_back(single_meas);
	single_meas << 3;
	measurements.push_back(single_meas);
	cout << "measurements: " << measurements[0] << ", " 
							 << measurements[1] << ", " 
							 << measurements[2] << endl;

	//call Kalman filter algorithm
	filter(x, P);

	return 0;
}


void filter(VectorXd &x, MatrixXd &P) {
	for (unsigned int n = 0; n < measurements.size(); ++n) {
		cout << "step " << n << ": " << endl;
		VectorXd z = measurements[n];  // 1*1

		//YOUR CODE HERE
		// KF Measurement update step
		MatrixXd y = z - H * x;  // 1*1
		MatrixXd S = H * P * H.transpose() + R;   // 1*1
		MatrixXd K = P * H.transpose() * S.inverse();   // 2*1
		
		// new state
		x = x + K * y;  // 2*1
		P = (I - K * H) * P;   // 2*2
		
		// KF Prediction step
		x = F * x + u;
		P = F * P * F.transpose() + Q;
		
		std::cout << "x=" << std::endl <<  x << std::endl;
		std::cout << "P=" << std::endl <<  P << std::endl;
		cout << endl;
	}
}