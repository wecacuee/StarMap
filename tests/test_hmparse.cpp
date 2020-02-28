#include <sstream>
#include <stdexcept> // std::runtime_error
#include "gtest/gtest.h" // TEST()
#include "starmap/starmap.h" // starmap::*
#include "torch/torch.h" // torch::*
#include "torch/script.h" // torch::*

#include "opencv2/opencv.hpp" // cv::*
#include "opencv2/imgcodecs.hpp" // cv::imread, cv::imwrite

namespace tjs = torch::jit::script;

class FileNotFoundError : std::runtime_error {
public:
  FileNotFoundError(const std::string& what_arg ) : std::runtime_error( what_arg ) { }
  FileNotFoundError(const char* what_arg ) : std::runtime_error( what_arg ) { }
};

cv::Mat safe_cv2_imread(const std::string fname = "tests/data/lena512.pgm") {
  cv::Mat inimg = cv::imread(fname, cv::IMREAD_UNCHANGED);
  if (inimg.data == nullptr)
    throw new FileNotFoundError(fname);
  return inimg;
}

TEST(HmParser, Nms) {
    cv::Mat img = safe_cv2_imread();
    cv::Mat det;
    img.convertTo(det, CV_32F);
    det = det / 255;
    cv::Mat pool =  starmap::nms(det);
    cv::Mat pooli8;
    pool.convertTo(pooli8, CV_8U, 255);
    cv::Mat expected = safe_cv2_imread("tests/data/test-lenna-nms-out.pgm");
    ASSERT_TRUE(pooli8.size == expected.size);
    cv::Mat diff = pooli8 != expected;
    ASSERT_TRUE(cv::countNonZero(diff) == 0) << "countNonZero: " << cv::countNonZero(diff);
}

TEST(HmParser, parseHeatmap) {
    cv::Mat hm;
    safe_cv2_imread().convertTo(hm, CV_32F, 1/ 255.);
    auto pts = starmap::parse_heatmap(hm);

    // Serialize using opencv
    cv::FileStorage fs("tests/data/test-lenna-parseHeatmap-out.cv2.yaml",
                       cv::FileStorage::READ);
    auto expected_pts = fs.getFirstTopLevelNode().mat();
    ASSERT_EQ(expected_pts.size[1], pts.size());
    for (int i = 0; i < expected_pts.size[1]; ++i) {
        ASSERT_EQ(expected_pts.at<int>(0, i), pts[i].y) << "Fail for i = " << i;
        ASSERT_EQ(expected_pts.at<int>(1, i), pts[i].x) << "Fail for i = " << i;
    }
}

