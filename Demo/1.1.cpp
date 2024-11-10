#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <opencv2/opencv.hpp>

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

#include "opencv2/core/core_c.h"
#include "opencv2/calib3d/calib3d_c.h"
#include "opencv2/core/cvdef.h"

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

static void printUsage(char** argv)
{
	cout <<
		"Rotation model images stitcher.\n\n"
		<< argv[0] << " img1 img2 [...imgN] [flags]\n\n"
		"Flags:\n"
		" --preview\n"
		" Run stitching in the preview mode. Works faster than usual mode,\n"
		" but output image will have lower resolution.\n"
		" --try_cuda (yes|no)\n"
		" Try to use CUDA. The default value is 'no'. All default values\n"
		" are for CPU mode.\n"
		"\nMotion Estimation Flags:\n"
		" --work_megapix <float>\n"
		" Resolution for image registration step. The default is 0.6 Mpx.\n"
		" --features (surf|orb|sift|akaze)\n"
		" Type of features used for images matching.\n"
		" The default is surf if available, orb otherwise.\n"
		" --matcher (homography|affine)\n"
		" Matcher used for pairwise image matching.\n"
		" --estimator (homography|affine)\n"
		" Type of estimator used for transformation estimation.\n"
		" --match_conf <float>\n"
		" Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
		" --conf_thresh <float>\n"
		" Threshold for two images are from the same panorama confidence.\n"
		" The default is 1.0.\n"
		" --ba (no|reproj|ray|affine)\n"
		" Bundle adjustment cost function. The default is ray.\n"
		" --ba_refine_mask (mask)\n"
		" Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
		" where 'x' means refine respective parameter and '_' means don't\n"
		" refine one, and has the following format:\n"
		" <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
		" adjustment doesn't support estimation of selected parameter then\n"
		" the respective flag is ignored.\n"
		" --wave_correct (no|horiz|vert)\n"
		" Perform wave effect correction. The default is 'horiz'.\n"
		" --save_graph <file_name>\n"
		" Save matches graph represented in DOT language to <file_name> file.\n"
		" Labels description: Nm is number of matches, Ni is number of inliers,\n"
		" C is confidence.\n"
		"\nCompositing Flags:\n"
		" --warp (affine|plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
		" Warp surface type. The default is 'spherical'.\n"
		" --seam_megapix <float>\n"
		" Resolution for seam estimation step. The default is 0.1 Mpx.\n"
		" --seam (no|voronoi|gc_color|gc_colorgrad)\n"
		" Seam estimation method. The default is 'gc_color'.\n"
		" --compose_megapix <float>\n"
		" Resolution for compositing step. Use -1 for original resolution.\n"
		" The default is -1.\n"
		" --expos_comp (no|gain|gain_blocks|channels|channels_blocks)\n"
		" Exposure compensation method. The default is 'gain_blocks'.\n"
		" Resolution for compositing step. Use -1 for original resolution.\n"
		" The default is -1.\n"
		" --expos_comp (no|gain|gain_blocks|channels|channels_blocks)\n"
		" Exposure compensation method. The default is 'gain_blocks'.\n"
		" --expos_comp_nr_feeds <int>\n"
		" Number of exposure compensation feed. The default is 1.\n"
		" --expos_comp_nr_filtering <int>\n"
		" Number of filtering iterations of the exposure compensation gains.\n"
		" Only used when using a block exposure compensation method.\n"
		" The default is 2.\n"
		" --expos_comp_block_size <int>\n"
		" BLock size in pixels used by the exposure compensator.\n"
		" Only used when using a block exposure compensation method.\n"
		" The default is 32.\n"
		" --blend (no|feather|multiband)\n"
		" Blending method. The default is 'multiband'.\n"
		" --blend_strength <float>\n"
		" Blending strength from [0,100] range. The default is 5.\n"
		" --output <result_img>\n"
		" The default is 'result.jpg'.\n"
		" --timelapse (as_is|crop) \n"
		" Output warped images separately as frames of a time lapse movie, with 'fixed_' prepended to input file names.\n"
		" --rangewidth <int>\n"
		" uses range_width to limit number of images to match with.\n";
}


// Default command line args
vector<String> img_names;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
#ifdef HAVE_OPENCV_XFEATURES2D
string features_type = "surf";
float match_conf = 0.65f;
#else
string features_type = "orb";
float match_conf = 0.3f;
#endif
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "reproj";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;


static int parseCmdArgs(int argc, char** argv)
{
	if (argc == 1)
	{
		printUsage(argv);
		return -1;
	}
	for (int i = 1; i < argc; ++i)
	{
		if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
		{
			printUsage(argv);
			return -1;
		}
		//����Ԥ��ģʽ��ƴ�ӹ��̸��죬�����ɵ�ͼ��ֱ��ʽϵ͡�
		else if (string(argv[i]) == "--preview")
		{
			preview = true;
		}
		else if (string(argv[i]) == "--try_cuda")
		{
			if (string(argv[i + 1]) == "no")
				try_cuda = false;
			else if (string(argv[i + 1]) == "yes")
				try_cuda = true;
			else
			{
				cout << "Bad --try_cuda flag value\n";
				return -1;
			}
			i++;
		}
		//ͼ����׼����ķֱ��ʣ�Ĭ��ֵΪ0.6�ף��������ء�
		else if (string(argv[i]) == "--work_megapix")
		{
			work_megapix = atof(argv[i + 1]);
			i++;
		}
		//�ӷ���Ʋ���ķֱ��ʣ�Ĭ����0.1������
		else if (string(argv[i]) == "--seam_megapix")
		{
			seam_megapix = atof(argv[i + 1]);
			i++;
		}
		//ƴ�Ӳ���ķֱ��ʣ�Ĭ����-1����ʾʹ�õ���ԭʼ�ֱ��ʣ���
		else if (string(argv[i]) == "--compose_megapix")
		{
			compose_megapix = atof(argv[i + 1]);
			i++;
		}
		//����ļ������ƣ�Ĭ��ֵΪ `result.jpg`
		else if (string(argv[i]) == "--result")
		{
			result_name = argv[i + 1];
			i++;
		}
		else if (string(argv[i]) == "--features")
		{
			features_type = argv[i + 1];
			if (string(features_type) == "orb")
				match_conf = 0.3f;
			i++;
		}
		else if (string(argv[i]) == "--matcher")
		{
			if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
				matcher_type = argv[i + 1];
			else
			{
				cout << "Bad --matcher flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--estimator")
		{
			if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
				estimator_type = argv[i + 1];
			else
			{
				cout << "Bad --estimator flag value\n";
				return -1;
			}
			i++;
		}
		//����ƥ������Ŷȡ�Ĭ��ֵ��0.65��������0.3��orb��
		else if (string(argv[i]) == "--match_conf")
		{
			match_conf = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
		//�ж�����ͼ������ͬһ��ȫ�������Ŷ���ֵ��Ĭ��ֵ��ȥ1.0
		else if (string(argv[i]) == "--conf_thresh")
		{
			conf_thresh = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
		//��������ĳɱ��������ͣ�Ĭ��ֵΪray������ѡ��Ϊno��
		else if (string(argv[i]) == "--ba")
		{
			ba_cost_func = argv[i + 1];
			i++;
		}
		//��������������Ż����룬��ʽΪ'x_xxx',ÿ���ַ������Ƿ���ض����������Ż���
		//'x' ��ʾ�����ò�����'_' ��ʾ������������˳��Ϊ�����ࡢƫб������ x ���ꡢ�ݺ�ȡ����� y ���ꡣ
		else if (string(argv[i]) == "--ba_refine_mask")
		{
			ba_refine_mask = argv[i + 1];
			if (ba_refine_mask.size() != 5)
			{
				cout << "Incorrect refinement mask length.\n";
				return -1;
			}
			i++;
		}
		//ִ�в��ν�����Ĭ��ֵΪhoriz
		else if (string(argv[i]) == "--wave_correct")
		{
			if (string(argv[i + 1]) == "no")
				do_wave_correct = false;
			else if (string(argv[i + 1]) == "horiz")
			{
				do_wave_correct = true;
				wave_correct = detail::WAVE_CORRECT_HORIZ;
			}
			else if (string(argv[i + 1]) == "vert")
			{
				do_wave_correct = true;
				wave_correct = detail::WAVE_CORRECT_VERT;
			}
			else
			{
				cout << "Bad --wave_correct flag value\n";
				return -1;
			}
			i++;
		}
		//����ļ��ı���·��
		else if (string(argv[i]) == "--save_graph")
		{
			save_graph = true;
			save_graph_to = argv[i + 1];
			i++;
		}
		//����ͼ��ƴ���Ե�ͶӰ���͡�
		else if (string(argv[i]) == "--warp")
		{
			warp_type = string(argv[i + 1]);
			i++;
		}
		//�عⲹ��������Ĭ��ֵΪgain_block�������棩
		else if (string(argv[i]) == "--expos_comp")
		{
			if (string(argv[i + 1]) == "no")
				expos_comp_type = ExposureCompensator::NO;
			else if (string(argv[i + 1]) == "gain")
				expos_comp_type = ExposureCompensator::GAIN;
			else if (string(argv[i + 1]) == "gain_blocks")
				expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
			else if (string(argv[i + 1]) == "channels")
				expos_comp_type = ExposureCompensator::CHANNELS;
			else if (string(argv[i + 1]) == "channels_blocks")
				expos_comp_type = ExposureCompensator::CHANNELS_BLOCKS;
			else
			{
				cout << "Bad exposure compensation method\n";
				return -1;
			}
			i++;
		}

		else if (string(argv[i]) == "--expos_comp_nr_feeds")
		{
			expos_comp_nr_feeds = atoi(argv[i + 1]);
			i++;
		}

		else if (string(argv[i]) == "--expos_comp_nr_filtering")
		{
			expos_comp_nr_filtering = atoi(argv[i + 1]);
			i++;
		}

		else if (string(argv[i]) == "--expos_comp_block_size")
		{
			expos_comp_block_size = atoi(argv[i + 1]);
			i++;
		}

		else if (string(argv[i]) == "--seam")
		{
			if (string(argv[i + 1]) == "no" ||
				string(argv[i + 1]) == "voronoi" ||
				string(argv[i + 1]) == "gc_color" ||
				string(argv[i + 1]) == "gc_colorgrad" ||
				string(argv[i + 1]) == "dp_color" ||
				string(argv[i + 1]) == "dp_colorgrad")
				seam_find_type = argv[i + 1];
			else
			{
				cout << "Bad seam finding method\n";
				return -1;
			}
			i++;
		}

		else if (string(argv[i]) == "--blend")
		{
			if (string(argv[i + 1]) == "no")
				blend_type = Blender::NO;
			else if (string(argv[i + 1]) == "feather")
				blend_type = Blender::FEATHER;
			else if (string(argv[i + 1]) == "multiband")
				blend_type = Blender::MULTI_BAND;
			else
			{
				cout << "Bad blending method\n";
				return -1;
			}
			i++;
		}

		else if (string(argv[i]) == "--timelapse")
		{
			timelapse = true;

			if (string(argv[i + 1]) == "as_is")
				timelapse_type = Timelapser::AS_IS;
			else if (string(argv[i + 1]) == "crop")
				timelapse_type = Timelapser::CROP;
			else
			{
				cout << "Bad timelapse method\n";
				return -1;
			}
			i++;
		}

		else if (string(argv[i]) == "--rangewidth")
		{
			range_width = atoi(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--blend_strength")
		{
			blend_strength = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
		else if (string(argv[i]) == "--output")
		{
			result_name = argv[i + 1];
			i++;
		}
		else
			img_names.push_back(argv[i]);
	}
	if (preview)
	{
		compose_megapix = 0.6;
	}
	return 0;
}

void calcDeriv(const Mat& err1, const Mat& err2, double h, Mat res)
{
	for (int i = 0; i < err1.rows; ++i)
		res.at<double>(i, 0) = (err2.at<double>(i, 0) - err1.at<double>(i, 0)) / h;
}

void setUpInitialCameraParams(const std::vector<CameraParams>& cameras, cv::Mat& cam_params)
{
	int num_images_ = static_cast<int>(cameras.size());
	SVD svd;

	for (int i = 0; i < num_images_; ++i)
	{
		cam_params.at<double>(i * 7, 0) = cameras[i].focal;
		cam_params.at<double>(i * 7 + 1, 0) = cameras[i].ppx;
		cam_params.at<double>(i * 7 + 2, 0) = cameras[i].ppy;
		cam_params.at<double>(i * 7 + 3, 0) = cameras[i].aspect;

		svd(cameras[i].R, SVD::FULL_UV);
		Mat R = svd.u * svd.vt;
		if (determinant(R) < 0)
			R *= -1;

		Mat rvec;
		Rodrigues(R, rvec);
		CV_Assert(rvec.type() == CV_32F);
		cam_params.at<double>(i * 7 + 4, 0) = rvec.at<float>(0, 0);
		cam_params.at<double>(i * 7 + 5, 0) = rvec.at<float>(1, 0);
		cam_params.at<double>(i * 7 + 6, 0) = rvec.at<float>(2, 0);
	}
}


void obtainRefinedCameraParams(std::vector<CameraParams>& cameras, cv::Mat& cam_params)
{
	
	int num_images_ = static_cast<int>(cameras.size());

	for (int i = 0; i < num_images_; ++i)
	{
		cameras[i].focal = cam_params.at<double>(i * 7, 0);
		cameras[i].ppx = cam_params.at<double>(i * 7 + 1, 0);
		cameras[i].ppy = cam_params.at<double>(i * 7 + 2, 0);
		cameras[i].aspect = cam_params.at<double>(i * 7 + 3, 0);

		Mat rvec(3, 1, CV_64F);
		rvec.at<double>(0, 0) = cam_params.at<double>(i * 7 + 4, 0);
		rvec.at<double>(1, 0) = cam_params.at<double>(i * 7 + 5, 0);
		rvec.at<double>(2, 0) = cam_params.at<double>(i * 7 + 6, 0);
		Rodrigues(rvec, cameras[i].R);

		Mat tmp;
		cameras[i].R.convertTo(tmp, CV_32F);
		cameras[i].R = tmp;
	}
}

void calcError(Mat& err, cv::Mat& cam_params_, int total_num_matches_,
	const std::vector<std::pair<int, int>>& edges_,
	const std::vector<ImageFeatures>& features_,
	const std::vector<MatchesInfo>& pairwise_matches_, int num_images_)
{
	err.create(total_num_matches_ * 2, 1, CV_64F);

	int match_idx = 0;
	for (size_t edge_idx = 0; edge_idx < edges_.size(); ++edge_idx)
	{
		int i = edges_[edge_idx].first;
		int j = edges_[edge_idx].second;
		double f1 = cam_params_.at<double>(i * 7, 0);
		double f2 = cam_params_.at<double>(j * 7, 0);
		double ppx1 = cam_params_.at<double>(i * 7 + 1, 0);
		double ppx2 = cam_params_.at<double>(j * 7 + 1, 0);
		double ppy1 = cam_params_.at<double>(i * 7 + 2, 0);
		double ppy2 = cam_params_.at<double>(j * 7 + 2, 0);
		double a1 = cam_params_.at<double>(i * 7 + 3, 0);
		double a2 = cam_params_.at<double>(j * 7 + 3, 0);

		double R1[9];
		Mat R1_(3, 3, CV_64F, R1);
		Mat rvec(3, 1, CV_64F);
		rvec.at<double>(0, 0) = cam_params_.at<double>(i * 7 + 4, 0);
		rvec.at<double>(1, 0) = cam_params_.at<double>(i * 7 + 5, 0);
		rvec.at<double>(2, 0) = cam_params_.at<double>(i * 7 + 6, 0);
		Rodrigues(rvec, R1_);

		double R2[9];
		Mat R2_(3, 3, CV_64F, R2);
		rvec.at<double>(0, 0) = cam_params_.at<double>(j * 7 + 4, 0);
		rvec.at<double>(1, 0) = cam_params_.at<double>(j * 7 + 5, 0);
		rvec.at<double>(2, 0) = cam_params_.at<double>(j * 7 + 6, 0);
		Rodrigues(rvec, R2_);

		const ImageFeatures& features1 = features_[i];
		const ImageFeatures& features2 = features_[j];
		const MatchesInfo& matches_info = pairwise_matches_[i * num_images_ + j];

		Mat_<double> K1 = Mat::eye(3, 3, CV_64F);
		K1(0, 0) = f1; K1(0, 2) = ppx1;
		K1(1, 1) = f1 * a1; K1(1, 2) = ppy1;

		Mat_<double> K2 = Mat::eye(3, 3, CV_64F);
		K2(0, 0) = f2; K2(0, 2) = ppx2;
		K2(1, 1) = f2 * a2; K2(1, 2) = ppy2;

		Mat_<double> H = K2 * R2_.inv() * R1_ * K1.inv();

		for (size_t k = 0; k < matches_info.matches.size(); ++k)
		{
			if (!matches_info.inliers_mask[k])
				continue;

			const DMatch& m = matches_info.matches[k];
			Point2f p1 = features1.keypoints[m.queryIdx].pt;
			Point2f p2 = features2.keypoints[m.trainIdx].pt;
			double x = H(0, 0) * p1.x + H(0, 1) * p1.y + H(0, 2);
			double y = H(1, 0) * p1.x + H(1, 1) * p1.y + H(1, 2);
			double z = H(2, 0) * p1.x + H(2, 1) * p1.y + H(2, 2);

			err.at<double>(2 * match_idx, 0) = p2.x - x / z;
			err.at<double>(2 * match_idx + 1, 0) = p2.y - y / z;
			match_idx++;
		}
	}
}

void calcJacobian(Mat& jac, Mat& cam_params_, int num_images_, int total_num_matches_,
	const std::vector<std::pair<int, int>>& edges_,
	const std::vector<ImageFeatures>& features_,
	const std::vector<MatchesInfo>& pairwise_matches_,
	Mat& refinement_mask_, Mat& err1_, Mat& err2_)
{
	jac.create(total_num_matches_ * 2, num_images_ * 7, CV_64F);
	jac.setTo(0);

	double val;
	const double step = 1e-4;

	for (int i = 0; i < num_images_; ++i)
	{
		if (refinement_mask_.at<uchar>(0, 0))
		{
			val = cam_params_.at<double>(i * 7, 0);
			cam_params_.at<double>(i * 7, 0) = val - step;
			calcError(err1_, cam_params_, total_num_matches_, edges_, features_, pairwise_matches_, num_images_);
			cam_params_.at<double>(i * 7, 0) = val + step;
			calcError(err2_, cam_params_, total_num_matches_, edges_, features_, pairwise_matches_, num_images_);
			calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7));
			cam_params_.at<double>(i * 7, 0) = val;
		}
		if (refinement_mask_.at<uchar>(0, 2))
		{
			val = cam_params_.at<double>(i * 7 + 1, 0);
			cam_params_.at<double>(i * 7 + 1, 0) = val - step;
			calcError(err1_, cam_params_, total_num_matches_, edges_, features_, pairwise_matches_, num_images_);
			cam_params_.at<double>(i * 7 + 1, 0) = val + step;
			calcError(err2_, cam_params_, total_num_matches_, edges_, features_, pairwise_matches_, num_images_);
			calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + 1));
			cam_params_.at<double>(i * 7 + 1, 0) = val;
		}
		if (refinement_mask_.at<uchar>(1, 2))
		{
			val = cam_params_.at<double>(i * 7 + 2, 0);
			cam_params_.at<double>(i * 7 + 2, 0) = val - step;
			calcError(err1_, cam_params_, total_num_matches_, edges_, features_, pairwise_matches_, num_images_);
			cam_params_.at<double>(i * 7 + 2, 0) = val + step;
			calcError(err2_, cam_params_, total_num_matches_, edges_, features_, pairwise_matches_, num_images_);
			calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + 2));
			cam_params_.at<double>(i * 7 + 2, 0) = val;
		}
		if (refinement_mask_.at<uchar>(1, 1))
		{
			val = cam_params_.at<double>(i * 7 + 3, 0);
			cam_params_.at<double>(i * 7 + 3, 0) = val - step;
			calcError(err1_, cam_params_, total_num_matches_, edges_, features_, pairwise_matches_, num_images_);
			cam_params_.at<double>(i * 7 + 3, 0) = val + step;
			calcError(err2_, cam_params_, total_num_matches_, edges_, features_, pairwise_matches_, num_images_);
			calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + 3));
			cam_params_.at<double>(i * 7 + 3, 0) = val;
		}
		for (int j = 4; j < 7; ++j)
		{
			val = cam_params_.at<double>(i * 7 + j, 0);
			cam_params_.at<double>(i * 7 + j, 0) = val - step;
			calcError(err1_, cam_params_, total_num_matches_, edges_, features_, pairwise_matches_, num_images_);
			cam_params_.at<double>(i * 7 + j, 0) = val + step;
			calcError(err2_, cam_params_, total_num_matches_, edges_, features_, pairwise_matches_, num_images_);
			calcDeriv(err1_, err2_, 2 * step, jac.col(i * 7 + j));
			cam_params_.at<double>(i * 7 + j, 0) = val;
		}
	}
}


int main(int argc, char* argv[])
{
#if ENABLE_LOG
	int64 app_start_time = getTickCount();
#endif

#if 0
	cv::setBreakOnError(true);
#endif

	int retval = parseCmdArgs(argc, argv);
	if (retval)
		return retval;

	// Check if have enough images
	int num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		return -1;
	}

	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

	LOGLN("Finding features...");
#if ENABLE_LOG
	int64 t = getTickCount();
#endif

	Ptr<Feature2D> finder;
	if (features_type == "orb")
	{
		finder = ORB::create();
	}
	else if (features_type == "akaze")
	{
		finder = AKAZE::create();
	}
#ifdef HAVE_OPENCV_XFEATURES2D
	else if (features_type == "surf")
	{
		finder = xfeatures2d::SURF::create();
	}
#endif
	else if (features_type == "sift")
	{
		finder = SIFT::create();
	}
	else
	{
		cout << "Unknown 2D features type: '" << features_type << "'.\n";
		return -1;
	}

	Mat full_img, img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;    

	for (int i = 0; i < num_images; ++i)
	{
		full_img = imread(samples::findFile(img_names[i])); 
		full_img_sizes[i] = full_img.size();

		if (full_img.empty())
		{
			LOGLN("Can't open image " << img_names[i]);
			return -1;
		}
		if (work_megapix < 0)
		{
			img = full_img;
			work_scale = 1;
			is_work_scale_set = true;
		}
		else
		{
			if (!is_work_scale_set)
			{
				work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
				is_work_scale_set = true;
			}
			resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
		}
		if (!is_seam_scale_set)
		{
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
			seam_work_aspect = seam_scale / work_scale;   
			is_seam_scale_set = true;
		}

		computeImageFeatures(finder, img, features[i]);
		features[i].img_idx = i;
		LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());

		resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
		images[i] = img.clone();
	}

	full_img.release();
	img.release();

	LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	LOG("Pairwise matching") << endl;
#if ENABLE_LOG
	t = getTickCount();
#endif
	vector<MatchesInfo> pairwise_matches;
	Ptr<FeaturesMatcher> matcher;
	if (matcher_type == "affine")
		matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
	else if (range_width == -1)
		matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
	else
		matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();

	LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	// Check if we should save matches graph
	if (save_graph)
	{
		LOGLN("Saving matches graph...");
		ofstream f(save_graph_to.c_str());
		f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
	}

	// Leave only images we are sure are from the same panorama
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	vector<Mat> img_subset;
	vector<String> img_names_subset;
	vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		img_names_subset.push_back(img_names[indices[i]]);
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}

	images = img_subset;
	img_names = img_names_subset;
	full_img_sizes = full_img_sizes_subset;

	// Check if we still have enough images
	num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return -1;
	}

	Ptr<Estimator> estimator;
	if (estimator_type == "affine")
		estimator = makePtr<AffineBasedEstimator>();
	else
		estimator = makePtr<HomographyBasedEstimator>();

	vector<CameraParams> cameras;
	if (!(*estimator)(features, pairwise_matches, cameras))
	{
		cout << "Homography estimation failed.\n";
		return -1;
	}

	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		LOGLN("Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
	}

	/*Ptr<detail::BundleAdjusterBase> adjuster;
	if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
	else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
	else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
	else
	{
		cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
		return -1;
	}
	adjuster->setConfThresh(conf_thresh);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
	adjuster->setRefinementMask(refine_mask);


	if (!(*adjuster)(features, pairwise_matches, cameras))
	{
		cout << "Camera parameters adjusting failed.\n";
		return -1;
	}*/


	LOGLN("Bundle adjustment");
#if ENABLE_LOG
	t = getTickCount();
#endif

	int num_images_ = static_cast<int>(features.size());
	ImageFeatures *features_ = &features[0];
	MatchesInfo *pairwise_matches_ = &pairwise_matches[0];

	Mat cam_params_;
	cam_params_.create(num_images_ * 7, 1, CV_64F);
	setUpInitialCameraParams(cameras, cam_params_);

	// Leave only consistent image pairs
	vector<pair<int,int>> edges_;
	for (int i = 0; i < num_images_ - 1; ++i)
	{
		for (int j = i + 1; j < num_images_; ++j)
		{
			const MatchesInfo& matches_info = pairwise_matches_[i * num_images_ + j];
			if (matches_info.confidence > conf_thresh)
				edges_.push_back(make_pair(i, j));
		}
	}

	// Compute number of correspondences
	int total_num_matches_ = 0;
	for (size_t i = 0; i < edges_.size(); ++i)
		total_num_matches_ += static_cast<int>(pairwise_matches_[edges_[i].first * num_images_ + 
			edges_[i].second].num_inliers);

	int num_params_per_cam_ = 7;
	int num_errs_per_measurement_ = 2;
	TermCriteria term_criteria_(TermCriteria::EPS + TermCriteria::COUNT, 1000, DBL_EPSILON);

	CvLevMarq solver(num_images_ * num_params_per_cam_,
		total_num_matches_ * num_errs_per_measurement_,
		cvTermCriteria(term_criteria_));

	Mat err, jac;
	CvMat matParams = cvMat(cam_params_);
	cvCopy(&matParams, solver.param);

#if ENABLE_LOG
	int iter = 0;
#endif
	

	Mat_<uchar>refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
	Mat_<uchar>refinement_mask_ = refine_mask.clone();

	for (;;)
	{
		const CvMat* _param = 0;
		CvMat* _jac = 0;
		CvMat* _err = 0;
		Mat err1_;
		Mat err2_;

		bool proceed = solver.update(_param, _jac, _err);

		cvCopy(_param, &matParams);

		if (!proceed || !_err)
			break;

		if (_jac)
		{
			calcJacobian(jac, cam_params_, num_images_, total_num_matches_, edges_, features, pairwise_matches, refinement_mask_, err1_, err2_);
			CvMat tmp = cvMat(jac);
			cvCopy(&tmp, _jac);
		}

		if (_err)
		{
			calcError(err, cam_params_, total_num_matches_, edges_, features, pairwise_matches, num_images_);
			//LOGLN(".");
#if ENABLE_LOG
			iter++;
#endif
			CvMat tmp = cvMat(err);
			cvCopy(&tmp, _err);
		}
	}

	LOGLN("");
	LOGLN("Bundle adjustment, final RMS error: " << std::sqrt(err.dot(err) / total_num_matches_));
	LOGLN("Bundle adjustment, iterations done: " << iter);

	// Check if all camera parameters are valid
	bool ok = true;
	for (int i = 0; i < cam_params_.rows; ++i)
	{
		if (cvIsNaN(cam_params_.at<double>(i, 0)))
		{
			ok = false;
			break;
		}
	}
	if (!ok)
		return false;

	obtainRefinedCameraParams(cameras, cam_params_);

	// Normalize motion to center image
	Graph span_tree;
	std::vector<int> span_tree_centers;
	findMaxSpanningTree(num_images_, pairwise_matches, span_tree, span_tree_centers);
	Mat R_inv = cameras[span_tree_centers[0]].R.inv();
	for (int i = 0; i < num_images_; ++i)
		cameras[i].R = R_inv * cameras[i].R;

	LOGLN("Bundle adjustment, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");


	// Find median focal length
	
	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		LOGLN("Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
		focals.push_back(cameras[i].focal);
	}

	sort(focals.begin(), focals.end());
	float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
	

	if (do_wave_correct)
	{
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R.clone());
		waveCorrect(rmats, wave_correct);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}

	LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
	t = getTickCount();
#endif

	vector<Point> corners(num_images);		
	vector<UMat> masks_warped(num_images);		
	vector<UMat> images_warped(num_images);		
	vector<Size> sizes(num_images);		
	vector<UMat> masks(num_images);		
	// Prepare images masks
	for (int i = 0; i < num_images; ++i)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	// Warp images and their masks

	Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
	if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
	{
		if (warp_type == "plane")
			warper_creator = makePtr<cv::PlaneWarperGpu>();
		else if (warp_type == "cylindrical")
			warper_creator = makePtr<cv::CylindricalWarperGpu>();
		else if (warp_type == "spherical")
			warper_creator = makePtr<cv::SphericalWarperGpu>();
	}
	else
#endif
	{
		if (warp_type == "plane")
			warper_creator = makePtr<cv::PlaneWarper>();
		else if (warp_type == "affine")
			warper_creator = makePtr<cv::AffineWarper>();
		else if (warp_type == "cylindrical")
			warper_creator = makePtr<cv::CylindricalWarper>();
		else if (warp_type == "spherical")
			warper_creator = makePtr<cv::SphericalWarper>();
		else if (warp_type == "fisheye")
			warper_creator = makePtr<cv::FisheyeWarper>();
		else if (warp_type == "stereographic")
			warper_creator = makePtr<cv::StereographicWarper>();
		else if (warp_type == "compressedPlaneA2B1")
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
		else if (warp_type == "compressedPlaneA1.5B1")
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
		else if (warp_type == "compressedPlanePortraitA2B1")
			warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
		else if (warp_type == "compressedPlanePortraitA1.5B1")
			warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
		else if (warp_type == "paniniA2B1")
			warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
		else if (warp_type == "paniniA1.5B1")
			warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
		else if (warp_type == "paniniPortraitA2B1")
			warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
		else if (warp_type == "paniniPortraitA1.5B1")
			warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
		else if (warp_type == "mercator")
			warper_creator = makePtr<cv::MercatorWarper>();
		else if (warp_type == "transverseMercator")
			warper_creator = makePtr<cv::TransverseMercatorWarper>();
	}

	if (!warper_creator)
	{
		cout << "Can't create the following warper '" << warp_type << "'\n";
		return 1;
	}

	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_work_aspect;    
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	LOGLN("Compensating exposure...");
#if ENABLE_LOG
	t = getTickCount();
#endif

	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	if (dynamic_cast<GainCompensator*>(compensator.get()))
	{
		GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
		gcompensator->setNrFeeds(expos_comp_nr_feeds);
	}

	if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
	{
		ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
		ccompensator->setNrFeeds(expos_comp_nr_feeds);
	}

	if (dynamic_cast<BlocksCompensator*>(compensator.get()))
	{
		BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
		bcompensator->setNrFeeds(expos_comp_nr_feeds);
		bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
		bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
	}

	compensator->feed(corners, images_warped, masks_warped);

	LOGLN("Compensating exposure, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	LOGLN("Finding seams...");
#if ENABLE_LOG
	t = getTickCount();
#endif

	Ptr<SeamFinder> seam_finder;
	if (seam_find_type == "no")
		seam_finder = makePtr<detail::NoSeamFinder>();
	else if (seam_find_type == "voronoi")
		seam_finder = makePtr<detail::VoronoiSeamFinder>();
	else if (seam_find_type == "gc_color")
	{
#ifdef HAVE_OPENCV_CUDALEGACY
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
		else
#endif
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
	}
	else if (seam_find_type == "gc_colorgrad")
	{
#ifdef HAVE_OPENCV_CUDALEGACY
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		else
#endif
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
	}
	else if (seam_find_type == "dp_color")
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
	else if (seam_find_type == "dp_colorgrad")
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
	if (!seam_finder)
	{
		cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
		return 1;
	}

	seam_finder->find(images_warped_f, corners, masks_warped);

	LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	// Release unused memory
	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

	LOGLN("Compositing...");
#if ENABLE_LOG
	t = getTickCount();
#endif

	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;
	Ptr<Blender> blender;
	Ptr<Timelapser> timelapser;
	//double compose_seam_aspect = 1;
	double compose_work_aspect = 1;

	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		LOGLN("Compositing image #" << indices[img_idx] + 1);

		// Read image and resize it if necessary
		full_img = imread(samples::findFile(img_names[img_idx]));
		if (!is_compose_scale_set)
		{
			if (compose_megapix > 0)
				compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
			is_compose_scale_set = true;

			// Compute relative scales
			//compose_seam_aspect = compose_scale / seam_scale;
			compose_work_aspect = compose_scale / work_scale;

			// Update warped image scale
			warped_image_scale *= static_cast<float>(compose_work_aspect);
			warper = warper_creator->create(warped_image_scale);

			// Update corners and sizes
			for (int i = 0; i < num_images; ++i)
			{
				// Update intrinsics
				cameras[i].focal *= compose_work_aspect;
				cameras[i].ppx *= compose_work_aspect;
				cameras[i].ppy *= compose_work_aspect;

				// Update corner and size
				Size sz = full_img_sizes[i];
				if (std::abs(compose_scale - 1) > 1e-1)
				{
					sz.width = cvRound(full_img_sizes[i].width * compose_scale);
					sz.height = cvRound(full_img_sizes[i].height * compose_scale);
				}

				Mat K;
				cameras[i].K().convertTo(K, CV_32F);
				Rect roi = warper->warpRoi(sz, K, cameras[i].R);
				corners[i] = roi.tl();
				sizes[i] = roi.size();
			}
		}
		if (abs(compose_scale - 1) > 1e-1)
			resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
		else
			img = full_img;
		full_img.release();
		Size img_size = img.size();

		Mat K;
		cameras[img_idx].K().convertTo(K, CV_32F);

		// Warp the current image
		warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

		// Compensate exposure
		compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		img.release();
		mask.release();

		dilate(masks_warped[img_idx], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
		mask_warped = seam_mask & mask_warped;

		if (!blender && !timelapse)
		{
			blender = Blender::createDefault(blend_type, try_cuda);
			Size dst_sz = resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
			if (blend_width < 1.f)
				blender = Blender::createDefault(Blender::NO, try_cuda);
			else if (blend_type == Blender::MULTI_BAND)
			{
				MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
				mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
				LOGLN("Multi-band blender, number of bands: " << mb->numBands());
			}
			else if (blend_type == Blender::FEATHER)
			{
				FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
				fb->setSharpness(1.f / blend_width);
				LOGLN("Feather blender, sharpness: " << fb->sharpness());
			}
			blender->prepare(corners, sizes);
		}
		else if (!timelapser && timelapse)
		{
			timelapser = Timelapser::createDefault(timelapse_type);
			timelapser->initialize(corners, sizes);
		}

		// Blend the current image
		if (timelapse)
		{
			timelapser->process(img_warped_s, Mat::ones(img_warped_s.size(), CV_8UC1), corners[img_idx]);
			String fixedFileName;
			size_t pos_s = String(img_names[img_idx]).find_last_of("/\\");
			if (pos_s == String::npos)
			{
				fixedFileName = "fixed_" + img_names[img_idx];
			}
			else
			{
				fixedFileName = "fixed_" + String(img_names[img_idx]).substr(pos_s + 1, String(img_names[img_idx]).length() - pos_s);
			}
			imwrite(fixedFileName, timelapser->getDst());
		}
		else
		{
			blender->feed(img_warped_s, mask_warped, corners[img_idx]);
		}
	}

	if (!timelapse)
	{
		Mat result, result_mask;
		blender->blend(result, result_mask);

		LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

		imwrite(result_name, result);
	}

	LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
	return 0;
}