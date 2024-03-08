// ------------------------------------------------------------------------------- //
// Extended Kalman Filter project
// Advanced Kalman Filtering and Sensor Fusion - By Udemy
// ------------------------------------------------------------------------------- //


#include "kalmanfilter.h"
#include "utils.h"

// -------------------------------------------------- //
// YOU CAN USE AND MODIFY THESE CONSTANTS HERE
constexpr double ACCEL_STD = 1.0;
constexpr double GYRO_STD = 0.01/180.0 * M_PI;
constexpr double INIT_VEL_STD = 10;
constexpr double INIT_PSI_STD = 45.0/180.0 * M_PI;
constexpr double GPS_POS_STD = 3.0;
constexpr double LIDAR_RANGE_STD = 3.0;
constexpr double LIDAR_THETA_STD = 0.02;
// -------------------------------------------------- //

void KalmanFilter::handleLidarMeasurements(const std::vector<LidarMeasurement>& dataset, const BeaconMap& map)
{
    // Assume No Correlation between the Measurements and Update Sequentially
    for(const auto& meas : dataset) {handleLidarMeasurement(meas, map);}
}

void KalmanFilter::handleLidarMeasurement(LidarMeasurement meas, const BeaconMap& map)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

       

        BeaconData map_beacon = map.getBeaconWithId(meas.id); // Match Beacon with built in Data Association Id
        if (meas.id != -1 && map_beacon.id != -1)
        {           
            // The map matched beacon positions can be accessed using: map_beacon.x AND map_beacon.y

            Vector2d z = Vector2d::Zero();
            Vector2d z_meas = Vector2d::Zero();

            z_meas << meas.range, meas.theta;

            double delta_x = map_beacon.x - state[0];
            double delta_y = map_beacon.y - state[1];

            double r = sqrt(delta_x * delta_x + delta_y * delta_y);
            

            double theta = wrapAngle(atan2(delta_y,delta_x) - state[2]);

            z << r, theta;

            Vector2d v = z_meas - z;

            v[1] = wrapAngle(v[1]);

            MatrixXd R = Matrix2d::Zero();
            

            R(0,0) = LIDAR_RANGE_STD * LIDAR_RANGE_STD;
            R(1,1) = LIDAR_THETA_STD * LIDAR_THETA_STD;

            MatrixXd H(2,4);

            H(0,0) = -delta_x/r;
            H(0,1) = -delta_y/r;
            H(0,2) = 0;
            H(0,3) = 0;
            H(1,0) = delta_y/r/r;
            H(1,1) = -delta_x/r/r;
            H(1,2) = -1;
            H(1,3) = 0;

            MatrixXd s;

            s = H * cov * H.transpose() + R;

            MatrixXd K;

            K = cov * H.transpose() * s.inverse();

            state = state + K * v;

            cov = (Matrix4d::Identity() - K*H) * cov;

        }

        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    }
}

void KalmanFilter::predictionStep(GyroMeasurement gyro, double dt)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        

        double Px = state(0);
        double Py = state(1);
        double PSI = state(2);
        double V = state(3);

        double Px_k = Px + dt * V * cos(PSI);
        double Py_k = Py + dt * V * sin(PSI);
        double PSI_k = wrapAngle(PSI + dt * gyro.psi_dot) ;
        double V_k = V;

        state << Px_k, Py_k, PSI_k, V_k ;

        MatrixXd F(4,4);
        F.setZero(4,4);


        F(0,0) = 1;
        F(1,1) = 1;
        F(2,2) = 1;
        F(3,3) = 1;

        F(0,2) = -1 * dt * V * sin(PSI);
        F(0,3) = dt * cos(PSI);
        F(1,2) = V * dt * cos(PSI);
        F(1,3) = dt * sin(PSI);

        MatrixXd Q(4,4); 
        Q.setZero(4,4);

        Q(2,2) = dt * dt * GYRO_STD * GYRO_STD;
        Q(3,3) = dt * dt * ACCEL_STD * ACCEL_STD;

        cov = F * cov * F.transpose() + Q;


        
        



        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    } 
}

void KalmanFilter::handleGPSMeasurement(GPSMeasurement meas)
{
    
    if(isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        VectorXd z = Vector2d::Zero();
        MatrixXd H = MatrixXd(2,4);
        MatrixXd R = Matrix2d::Zero();

        z << meas.x,meas.y;
        H << 1,0,0,0,0,1,0,0;
        R(0,0) = GPS_POS_STD*GPS_POS_STD;
        R(1,1) = GPS_POS_STD*GPS_POS_STD;

        VectorXd z_hat = H * state;
        VectorXd y = z - z_hat;
        MatrixXd S = H * cov * H.transpose() + R;
        MatrixXd K = cov*H.transpose()*S.inverse();

        state = state + K*y;
        cov = (Matrix4d::Identity() - K*H) * cov;

        setState(state);
        setCovariance(cov);
    }
    else
    {
        VectorXd state = Vector4d::Zero();
        MatrixXd cov = Matrix4d::Zero();

        state(0) = meas.x;
        state(1) = meas.y;
        cov(0,0) = GPS_POS_STD*GPS_POS_STD;
        cov(1,1) = GPS_POS_STD*GPS_POS_STD;
        cov(2,2) = INIT_PSI_STD*INIT_PSI_STD;
        cov(3,3) = INIT_VEL_STD*INIT_VEL_STD;

        setState(state);
        setCovariance(cov);
    } 
             
}

Matrix2d KalmanFilter::getVehicleStatePositionCovariance()
{
    Matrix2d pos_cov = Matrix2d::Zero();
    MatrixXd cov = getCovariance();
    if (isInitialised() && cov.size() != 0){pos_cov << cov(0,0), cov(0,1), cov(1,0), cov(1,1);}
    return pos_cov;
}

VehicleState KalmanFilter::getVehicleState()
{
    if (isInitialised())
    {
        VectorXd state = getState(); // STATE VECTOR [X,Y,PSI,V,...]
        return VehicleState(state[0],state[1],state[2],state[3]);
    }
    return VehicleState();
}

void KalmanFilter::predictionStep(double dt){}
