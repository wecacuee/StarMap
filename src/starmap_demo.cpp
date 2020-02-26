#include <iostream>
#include <tuple>


#include "torch/script.h"
#include "boost/program_options.hpp"
#include "opencv2/opencv.hpp"

#include "starmap/starmap.h" // provides Crop, parseHeatmap, horn87

using namespace std;
namespace bpo = boost::program_options;


/**
 * @brief Parse command line arguments
 * @return command line options
 */
tuple<bool, bpo::variables_map> parse_commandline(const int argc, char** const argv) {
    bpo::options_description desc("Demonstrate running starmap");
    desc.add_options()
            ("help", "produce help message")
            ("loadModel", bpo::value<string>()->required(),
             "Path to the pre-trained model file")
            ("demo", bpo::value<string>()->required(),
             "Path to an image file to test upon ")
            ("input_res", bpo::value<int>()->default_value(256),
             "The resolution of image that network accepts")
            ("GPU", bpo::value<int>()->default_value(-1),
             "GPU Id to use. For CPU specify -1")
            ;
    bpo::variables_map vm;
    bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        cout << desc << "\n";
        return make_tuple(false, vm);
    }
    bpo::notify(vm);
    return make_tuple(true, vm);
}


int main(const int argc, char** const argv) {
    // opt = opts().parse()
    bool cont;
    bpo::variables_map opt;
    tie(cont, opt) = parse_commandline(argc, argv);
    if (!cont)
        return 1;

    auto output = starmap::run_starmap_on_img(
            opt["loadModel"].as<string>(),
            opt["demo"].as<string>(),
            opt["input_res"].as<const int>(),
            opt["GPU"].as<const int>() );
    /*

  
  debugger = Debugger()
  img = (input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8).copy()
  inp = img.copy()
  star = (cv2.resize(hm[0, 0], (ref.inputRes, ref.inputRes)) * 255)
  star[star > 255] = 255
  star[star < 0] = 0
  star = np.tile(star, (3, 1, 1)).transpose(1, 2, 0)
  trans = 0.8
  star = (trans * star + (1. - trans) * img).astype(np.uint8)

   
  ps = parseHeatmap(hm[0], thresh = 0.1)
  canonical, pred, color, score = [], [], [], []
  for k in range(len(ps[0])):
    x, y, z = ((hm[0, 1:4, ps[0][k], ps[1][k]] + 0.5) * ref.outputRes).astype(np.int32)
    dep = ((hm[0, 4, ps[0][k], ps[1][k]] + 0.5) * ref.outputRes).astype(np.int32)
    canonical.append([x, y, z])
    pred.append([ps[1][k], ref.outputRes - dep, ref.outputRes - ps[0][k]])
    score.append(hm[0, 0, ps[0][k], ps[1][k]])
    color.append((1.0 * x / ref.outputRes, 1.0 * y / ref.outputRes, 1.0 * z / ref.outputRes))
    cv2.circle(img, (ps[1][k] * 4, ps[0][k] * 4), 4, (255, 255, 255), -1)
    cv2.circle(img, (ps[1][k] * 4, ps[0][k] * 4), 2, (int(z * 4), int(y * 4), int(x * 4)), -1)
  
  pred = np.array(pred).astype(np.float32)
  canonical = np.array(canonical).astype(np.float32)
  
  pointS = canonical * 1.0 / ref.outputRes
  pointT = pred * 1.0 / ref.outputRes
  R, t, s = horn87(pointS.transpose(), pointT.transpose(), score)
  
  rotated_pred = s * np.dot(R, canonical.transpose()).transpose() + t * ref.outputRes

  debugger.addImg(inp, 'inp')
  debugger.addImg(star, 'star')
  debugger.addImg(img, 'nms')
  debugger.addPoint3D(canonical / ref.outputRes - 0.5, c = color, marker = '^')
  debugger.addPoint3D(pred / ref.outputRes - 0.5, c = color, marker = 'x')
  debugger.addPoint3D(rotated_pred / ref.outputRes - 0.5, c = color, marker = '*')

  debugger.showAllImg(pause = True)
  debugger.show3D()
  */
}
