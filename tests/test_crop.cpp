#include <sstream>
#include <stdexcept> // std::runtime_error
#include "gtest/gtest.h" // TEST()
#include "starmap/starmap.h" // starmap::

#include "opencv2/opencv.hpp" // cv::*
#include "opencv2/imgcodecs.hpp" // cv::imread, cv::imwrite

class FileNotFoundError : std::runtime_error {
public:
  FileNotFoundError(const std::string& what_arg ) : std::runtime_error( what_arg ) { }
  FileNotFoundError(const char* what_arg ) : std::runtime_error( what_arg ) { }
};

cv::Mat imread_img(const cv::Size img_shape,
                   const std::string fname = "tests/data/lena512.pgm") {
  cv::Mat outimg;
  cv::Mat inimg = cv::imread(fname, cv::IMREAD_UNCHANGED);
  if (inimg.data == nullptr)
    throw new FileNotFoundError(fname);
  cv::resize(inimg, outimg, img_shape);
  return outimg;
}


/**
 * @param img_shape: (height, width)
 */
bool _test_crop(const int rows,
                const int cols,
                const int desired_side = 256,
                const int rot = 0) {
  cv::Size imgshape(rows, cols);
  auto img = imread_img(imgshape);
  auto cropped = starmap::crop(img, desired_side);
  std::ostringstream filepath;
  filepath << "tests/data/test-crop-lenna-" << rows << "-" << cols << ".pgm";
  cv::imwrite(filepath.str(), cropped);
  cv::Mat expected_img = cv::imread(filepath.str(), cv::IMREAD_UNCHANGED);
  cv::Mat diff = cropped != expected_img;
  return cv::countNonZero(diff) == 0;
}


TEST(CropTest, HandlesFatBigBig) {
  ASSERT_TRUE(_test_crop(300, 400)) << "Images do not match";
}


TEST(CropTest, HandlesFatSmallBig) {
  ASSERT_TRUE(_test_crop(100, 400)) << "Images do not match";
}

TEST(CropTest, HandlesFatSmallSmall) {
  ASSERT_TRUE(_test_crop(100, 200)) << "Images do not match";
}

TEST(CropTest, HandlesTallSmallSmall) {
  ASSERT_TRUE(_test_crop(200, 100)) << "Images do not match";
}

TEST(CropTest, HandlesTallBigSmall) {
  ASSERT_TRUE(_test_crop(400, 100)) << "Images do not match";
}

TEST(CropTest, HandlesTallBigBig) {
  ASSERT_TRUE(_test_crop(400, 300)) << "Images do not match";
}
