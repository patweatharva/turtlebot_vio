#ifndef GRAPH_VIO_HANDLER_H
#define GRAPH_VIO_HANDLER_H

#include <boost/program_options.hpp>

#include "ros/ros.h"

#include <Eigen/Dense>

// GTSAM related includes.
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/ImuBias.h>

#include <vector>
#include <iostream>

using namespace std;
using namespace gtsam;

using symbol_shorthand::V;
using symbol_shorthand::X;

const double kGravity = 9.81;

class graph_vio_handler
{
private:
    ros::NodeHandle nh_;
    ros::Subscriber feature2D_sub_, imu_sub_;

    NonlinearFactorGraph *graph_;
    ISAM2 *isam2_;

    double gravity_;

    std::shared_ptr<PreintegrationType> preintegrated_;

public:
    graph_vio_handler(ros::NodeHandle &nh);
    ~graph_vio_handler();

    void imuCB();
    void featureCB();
    std::shared_ptr<PreintegratedCombinedMeasurements::Params> imuParams();
};

graph_vio_handler::graph_vio_handler(ros::NodeHandle &nh) : nh_(nh), graph_(new NonlinearFactorGraph()), isam2_(0), preintegrated_(nullptr)
{
    if (nh_.getParam("/gravity", gravity_))
    {
        ROS_INFO_STREAM("Gravity Read!!");
    }

}

graph_vio_handler::~graph_vio_handler()
{
    ;
}

void graph_vio_handler::imuCB()
{
    int a = 1;
    return;
};

// std::shared_ptr<PreintegratedCombinedMeasurements::Params> graph_vio_handler::imuParams()
// {
//     return
// };

#endif // GRAPH_VIO_HANDLER_H