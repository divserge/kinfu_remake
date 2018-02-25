#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <kfusion/kinfu.hpp>
#include <io/capture.hpp>

using namespace kfusion;

struct KinFuApp
{

    KinFuApp(OpenNISource& source) : exit_ (false),  iteractive_mode_(false), capture_ (source), pause_(false)
    {
        KinFuParams params = KinFuParams::default_params();
        kinfu_ = KinFu::Ptr( new KinFu(params) );

        capture_.setRegistration(true);
    }

    void show_depth(const cv::Mat& depth)
    {
        cv::Mat display;
        //cv::normalize(depth, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        depth.convertTo(display, CV_8U, 255.0/4000);
    }

    void take_cloud(KinFu& kinfu)
    {
        cuda::DeviceArray<Point> cloud = kinfu.tsdf().fetchCloud(cloud_buffer);
        cv::Mat cloud_host(1, (int)cloud.size(), CV_32FC4);
        cloud.download(cloud_host.ptr<Point>());
        //viz.showWidget("cloud", cv::viz::WCloud(cloud_host));
        //viz.showWidget("cloud", cv::viz::WPaintedCloud(cloud_host));
    }


    void show_raycasted(KinFu& kinfu, int frame)
    {
        const int mode = 3;
        kinfu.renderImage(view_device_, kinfu.getCameraPose(), mode);
        view_host_.create(view_device_.rows(), view_device_.cols(), CV_8UC4);
        view_device_.download(view_host_.ptr<void>(), view_host_.step);
        cv::imwrite(std::string("Scene.jpg"), view_host_);
    }

    bool execute()
    {
        KinFu& kinfu = *kinfu_;
        cv::Mat depth, image;
        double time_ms = 0;
        bool has_image = false;

        for (int i = 0; !exit_; ++i) {
            
            bool has_frame = capture_.grab(depth, image);
            
            if (!has_frame)
                return std::cout << "Can't grab" << std::endl, false;

            depth_device_.upload(depth.data, depth.step, depth.rows, depth.cols);

            {
                SampledScopeTime fps(time_ms); (void)fps;
                has_image = kinfu(depth_device_);
            }

            std::cout << kinfu.tsdf().getDims() << std::endl;
            std::cout << kinfu.tsdf().getVoxelSize() << std::endl;
            std::cout << kinfu.tsdf().data().sizeBytes() / (512 * 512 * 512) << std::endl;

            std::vector<float> buffer (
                kinfu.tsdf().getDims()[0] *
                kinfu.tsdf().getDims()[1] *
                kinfu.tsdf().getDims()[2], 0
            );

            kinfu.tsdf().data().download(&buffer[0]);
            std::cout << "downloaded" << std::endl;
            std::cout << buffer[1000] << std::endl;

            if (has_image)
                show_raycasted(kinfu, i);

            // siz a = kinfu.tsdf().data()

            //show_depth(depth);
            //cv::imshow("Image", image);

            
        }
        return true;
    }

    bool pause_ /*= false*/;
    bool exit_, iteractive_mode_;
    OpenNISource& capture_;
    KinFu::Ptr kinfu_;

    cv::Mat view_host_;
    cuda::Image view_device_;
    cuda::Depth depth_device_;
    cuda::DeviceArray<Point> cloud_buffer;
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main (int argc, char* argv[])
{
    int device = 0;
    cuda::setDevice (device);
    cuda::printShortCudaDeviceInfo (device);

    if(cuda::checkIfPreFermiGPU(device))
        return std::cout << std::endl << "Kinfu is not supported for pre-Fermi GPU architectures, and not built for them by default. Exiting..." << std::endl, 1;

    OpenNISource capture;
    
    std::cout << "Enter path to .oni file" << std::endl;
    std::string path;
    std::cin >> path;

    capture.open (path);
    //capture.open("d:/onis/20111013-224932.oni");
    //capture.open("d:/onis/reg20111229-180846.oni");
    //capture.open("d:/onis/white1.oni");
    //capture.open("/media/Main/onis/20111013-224932.oni");
    //capture.open("20111013-225218.oni");
    //capture.open("d:/onis/20111013-224551.oni");
    //capture.open("d:/onis/20111013-224719.oni");

    KinFuApp app (capture);

    // executing
    try { app.execute (); }
    catch (const std::bad_alloc& /*e*/) { std::cout << "Bad alloc" << std::endl; }
    catch (const std::exception& /*e*/) { std::cout << "Exception" << std::endl; }

    return 0;
}
