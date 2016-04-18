#include "replaces.hpp"
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) cascadedetect_INTEGER_13_1_2008-cascadedetect_INTEGER_13_2_2013, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Itseez Inc. may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <cstdio>

#include "cascadedetect.hpp"
#include "opencv2/objdetect/objdetect_c.h"
#include "opencl_kernels_objdetect.hpp"

namespace cv
{

template<typename _Tp> void copyVectorToUMat(const std::vector<_Tp>& v, UMat& um)
{
if(v.empty())
um.release();
Mat(cascadedetect_INTEGER_56_1_1, (int)(v.size()*sizeof(v[cascadedetect_INTEGER_56_2_0])), CV_8U, (void*)&v[cascadedetect_INTEGER_56_3_0]).copyTo(um);
}

void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps,
std::vector<int>* weights, std::vector<double>* levelWeights)
{
if( groupThreshold <= cascadedetect_INTEGER_62_1_0 || rectList.empty() )
{
if( weights )
{
size_t i, sz = rectList.size();
weights->resize(sz);
for( i = cascadedetect_INTEGER_68_1_0; i < sz; i+=cascadedetect_INTEGER_68_2_1 )
(*weights)[i] = cascadedetect_INTEGER_69_1_1;
}
return;
}

std::vector<int> labels;
int nclasses = partition(rectList, labels, SimilarRects(eps));

std::vector<Rect> rrects(nclasses);
std::vector<int> rweights(nclasses, cascadedetect_INTEGER_78_1_0);
std::vector<int> rejectLevels(nclasses, cascadedetect_INTEGER_79_1_0);
std::vector<double> rejectWeights(nclasses, DBL_MIN);
int i, j, nlabels = (int)labels.size();
for( i = cascadedetect_INTEGER_82_1_0; i < nlabels; i+=cascadedetect_INTEGER_82_2_1 )
{
int cls = labels[i];
rrects[cls].x += rectList[i].x;
rrects[cls].y += rectList[i].y;
rrects[cls].width += rectList[i].width;
rrects[cls].height += rectList[i].height;
rweights[cls]+=cascadedetect_INTEGER_89_1_1;
}

bool useDefaultWeights = false;

if ( levelWeights && weights && !weights->empty() && !levelWeights->empty() )
{
for( i = cascadedetect_INTEGER_96_1_0; i < nlabels; i+=cascadedetect_INTEGER_96_2_1 )
{
int cls = labels[i];
if( (*weights)[i] > rejectLevels[cls] )
{
rejectLevels[cls] = (*weights)[i];
rejectWeights[cls] = (*levelWeights)[i];
}
else if( ( (*weights)[i] == rejectLevels[cls] ) && ( (*levelWeights)[i] > rejectWeights[cls] ) )
rejectWeights[cls] = (*levelWeights)[i];
}
}
else
useDefaultWeights = true;

for( i = cascadedetect_INTEGER_111_1_0; i < nclasses; i+=cascadedetect_INTEGER_111_2_1 )
{
Rect r = rrects[i];
float s = 1.f/rweights[i];
rrects[i] = Rect(saturate_cast<int>(r.x*s),
saturate_cast<int>(r.y*s),
saturate_cast<int>(r.width*s),
saturate_cast<int>(r.height*s));
}

rectList.clear();
if( weights )
weights->clear();
if( levelWeights )
levelWeights->clear();

for( i = cascadedetect_INTEGER_127_1_0; i < nclasses; i+=cascadedetect_INTEGER_127_2_1 )
{
Rect r1 = rrects[i];
int n1 = rweights[i];
double w1 = rejectWeights[i];
int l1 = rejectLevels[i];

// filter out rectangles which don't have enough similar rectangles
if( n1 <= groupThreshold )
continue;
// filter out small face rectangles inside large rectangles
for( j = cascadedetect_INTEGER_138_1_0; j < nclasses; j+=cascadedetect_INTEGER_138_2_1 )
{
int n2 = rweights[j];

if( j == i || n2 <= groupThreshold )
continue;
Rect r2 = rrects[j];

int dx = saturate_cast<int>( r2.width * eps );
int dy = saturate_cast<int>( r2.height * eps );

if( i != j &&
r1.x >= r2.x - dx &&
r1.y >= r2.y - dy &&
r1.x + r1.width <= r2.x + r2.width + dx &&
r1.y + r1.height <= r2.y + r2.height + dy &&
(n2 > std::max(cascadedetect_INTEGER_154_1_3, n1) || n1 < cascadedetect_INTEGER_154_2_3) )
break;
}

if( j == nclasses )
{
rectList.push_back(r1);
if( weights )
weights->push_back(useDefaultWeights ? n1 : l1);
if( levelWeights )
levelWeights->push_back(w1);
}
}
}

class MeanshiftGrouping
{
public:
MeanshiftGrouping(const Point3d& densKer, const std::vector<Point3d>& posV,
const std::vector<double>& wV, double eps, int maxIter = cascadedetect_INTEGER_173_1_20)
{
densityKernel = densKer;
weightsV = wV;
positionsV = posV;
positionsCount = (int)posV.size();
meanshiftV.resize(positionsCount);
distanceV.resize(positionsCount);
iterMax = maxIter;
modeEps = eps;

for (unsigned i = cascadedetect_INTEGER_184_1_0; i<positionsV.size(); i+=cascadedetect_INTEGER_184_2_1)
{
meanshiftV[i] = getNewValue(positionsV[i]);
distanceV[i] = moveToMode(meanshiftV[i]);
meanshiftV[i] -= positionsV[i];
}
}

void getModes(std::vector<Point3d>& modesV, std::vector<double>& resWeightsV, const double eps)
{
for (size_t i=cascadedetect_INTEGER_194_1_0; i <distanceV.size(); i+=cascadedetect_INTEGER_194_2_1)
{
bool is_found = false;
for(size_t j=cascadedetect_INTEGER_197_1_0; j<modesV.size(); j+=cascadedetect_INTEGER_197_2_1)
{
if ( getDistance(distanceV[i], modesV[j]) < eps)
{
is_found=true;
break;
}
}
if (!is_found)
{
modesV.push_back(distanceV[i]);
}
}

resWeightsV.resize(modesV.size());

for (size_t i=cascadedetect_INTEGER_213_1_0; i<modesV.size(); i+=cascadedetect_INTEGER_213_2_1)
{
resWeightsV[i] = getResultWeight(modesV[i]);
}
}

protected:
std::vector<Point3d> positionsV;
std::vector<double> weightsV;

Point3d densityKernel;
int positionsCount;

std::vector<Point3d> meanshiftV;
std::vector<Point3d> distanceV;
int iterMax;
double modeEps;

Point3d getNewValue(const Point3d& inPt) const
{
Point3d resPoint(.0);
Point3d ratPoint(.0);
for (size_t i=cascadedetect_INTEGER_235_1_0; i<positionsV.size(); i+=cascadedetect_INTEGER_235_2_1)
{
Point3d aPt= positionsV[i];
Point3d bPt = inPt;
Point3d sPt = densityKernel;

sPt.x *= std::exp(aPt.z);
sPt.y *= std::exp(aPt.z);

aPt.x /= sPt.x;
aPt.y /= sPt.y;
aPt.z /= sPt.z;

bPt.x /= sPt.x;
bPt.y /= sPt.y;
bPt.z /= sPt.z;

double w = (weightsV[i])*std::exp(-((aPt-bPt).dot(aPt-bPt))/cascadedetect_INTEGER_252_1_2)/std::sqrt(sPt.dot(Point3d(cascadedetect_INTEGER_252_2_1,cascadedetect_INTEGER_252_3_1,cascadedetect_INTEGER_252_4_1)));

resPoint += w*aPt;

ratPoint.x += w/sPt.x;
ratPoint.y += w/sPt.y;
ratPoint.z += w/sPt.z;
}
resPoint.x /= ratPoint.x;
resPoint.y /= ratPoint.y;
resPoint.z /= ratPoint.z;
return resPoint;
}

double getResultWeight(const Point3d& inPt) const
{
double sumW=cascadedetect_INTEGER_268_1_0;
for (size_t i=cascadedetect_INTEGER_269_1_0; i<positionsV.size(); i+=cascadedetect_INTEGER_269_2_1)
{
Point3d aPt = positionsV[i];
Point3d sPt = densityKernel;

sPt.x *= std::exp(aPt.z);
sPt.y *= std::exp(aPt.z);

aPt -= inPt;

aPt.x /= sPt.x;
aPt.y /= sPt.y;
aPt.z /= sPt.z;

sumW+=(weightsV[i])*std::exp(-(aPt.dot(aPt))/cascadedetect_INTEGER_283_1_2)/std::sqrt(sPt.dot(Point3d(cascadedetect_INTEGER_283_2_1,cascadedetect_INTEGER_283_3_1,cascadedetect_INTEGER_283_4_1)));
}
return sumW;
}

Point3d moveToMode(Point3d aPt) const
{
Point3d bPt;
for (int i = cascadedetect_INTEGER_291_1_0; i<iterMax; i+=cascadedetect_INTEGER_291_2_1)
{
bPt = aPt;
aPt = getNewValue(bPt);
if ( getDistance(aPt, bPt) <= modeEps )
{
break;
}
}
return aPt;
}

double getDistance(Point3d p1, Point3d p2) const
{
Point3d ns = densityKernel;
ns.x *= std::exp(p2.z);
ns.y *= std::exp(p2.z);
p2 -= p1;
p2.x /= ns.x;
p2.y /= ns.y;
p2.z /= ns.z;
return p2.dot(p2);
}
};
//new grouping function with using meanshift
static void groupRectangles_meanshift(std::vector<Rect>& rectList, double detectThreshold, std::vector<double>* foundWeights,
std::vector<double>& scales, Size winDetSize)
{
int detectionCount = (int)rectList.size();
std::vector<Point3d> hits(detectionCount), resultHits;
std::vector<double> hitWeights(detectionCount), resultWeights;
Point2d hitCenter;

for (int i=cascadedetect_INTEGER_324_1_0; i < detectionCount; i+=cascadedetect_INTEGER_324_2_1)
{
hitWeights[i] = (*foundWeights)[i];
hitCenter = (rectList[i].tl() + rectList[i].br())*(0.5); //center of rectangles
hits[i] = Point3d(hitCenter.x, hitCenter.y, std::log(scales[i]));
}

rectList.clear();
if (foundWeights)
foundWeights->clear();

double logZ = std::log(1.3);
Point3d smothing(cascadedetect_INTEGER_336_1_8, cascadedetect_INTEGER_336_2_16, logZ);

MeanshiftGrouping msGrouping(smothing, hits, hitWeights, 1e-5, cascadedetect_INTEGER_338_1_100);

msGrouping.getModes(resultHits, resultWeights, cascadedetect_INTEGER_340_1_1);

for (unsigned i=cascadedetect_INTEGER_342_1_0; i < resultHits.size(); ++i)
{

double scale = std::exp(resultHits[i].z);
hitCenter.x = resultHits[i].x;
hitCenter.y = resultHits[i].y;
Size s( int(winDetSize.width * scale), int(winDetSize.height * scale) );
Rect resultRect( int(hitCenter.x-s.width/cascadedetect_INTEGER_349_1_2), int(hitCenter.y-s.height/cascadedetect_INTEGER_349_2_2),
int(s.width), int(s.height) );

if (resultWeights[i] > detectThreshold)
{
rectList.push_back(resultRect);
foundWeights->push_back(resultWeights[i]);
}
}
}

void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps)
{
groupRectangles(rectList, groupThreshold, eps, cascadedetect_INTEGER_362_1_0, cascadedetect_INTEGER_362_2_0);
}

void groupRectangles(std::vector<Rect>& rectList, std::vector<int>& weights, int groupThreshold, double eps)
{
groupRectangles(rectList, groupThreshold, eps, &weights, cascadedetect_INTEGER_367_1_0);
}
//used for cascade detection algorithm for ROC-curve calculating
void groupRectangles(std::vector<Rect>& rectList, std::vector<int>& rejectLevels,
std::vector<double>& levelWeights, int groupThreshold, double eps)
{
groupRectangles(rectList, groupThreshold, eps, &rejectLevels, &levelWeights);
}
//can be used for HOG detection algorithm only
void groupRectangles_meanshift(std::vector<Rect>& rectList, std::vector<double>& foundWeights,
std::vector<double>& foundScales, double detectThreshold, Size winDetSize)
{
groupRectangles_meanshift(rectList, detectThreshold, &foundWeights, foundScales, winDetSize);
}


FeatureEvaluator::~FeatureEvaluator() {}

bool FeatureEvaluator::read(const FileNode&, Size _origWinSize)
{
origWinSize = _origWinSize;
localSize = lbufSize = Size(cascadedetect_INTEGER_388_1_0, cascadedetect_INTEGER_388_2_0);
if (scaleData.empty())
scaleData = makePtr<std::vector<ScaleData> >();
else
scaleData->clear();
return true;
}

Ptr<FeatureEvaluator> FeatureEvaluator::clone() const { return Ptr<FeatureEvaluator>(); }
int FeatureEvaluator::getFeatureType() const {return -cascadedetect_INTEGER_397_1_1;}
bool FeatureEvaluator::setWindow(Point, int) { return true; }
void FeatureEvaluator::getUMats(std::vector<UMat>& bufs)
{
if (!(sbufFlag & USBUF_VALID))
{
sbuf.copyTo(usbuf);
sbufFlag |= USBUF_VALID;
}

bufs.clear();
bufs.push_back(uscaleData);
bufs.push_back(usbuf);
bufs.push_back(ufbuf);
}

void FeatureEvaluator::getMats()
{
if (!(sbufFlag & SBUF_VALID))
{
usbuf.copyTo(sbuf);
sbufFlag |= SBUF_VALID;
}
}

float FeatureEvaluator::calcOrd(int) const { return 0.; }
int FeatureEvaluator::calcCat(int) const { return cascadedetect_INTEGER_423_1_0; }

bool FeatureEvaluator::updateScaleData( Size imgsz, const std::vector<float>& _scales )
{
if( scaleData.empty() )
scaleData = makePtr<std::vector<ScaleData> >();

size_t i, nscales = _scales.size();
bool recalcOptFeatures = nscales != scaleData->size();
scaleData->resize(nscales);

int layer_dy = cascadedetect_INTEGER_434_1_0;
Point layer_ofs(cascadedetect_INTEGER_435_1_0,cascadedetect_INTEGER_435_2_0);
Size prevBufSize = sbufSize;
sbufSize.width = std::max(sbufSize.width, (int)alignSize(cvRound(imgsz.width/_scales[cascadedetect_INTEGER_437_1_0]) + cascadedetect_INTEGER_437_2_31, cascadedetect_INTEGER_437_3_32));
recalcOptFeatures = recalcOptFeatures || sbufSize.width != prevBufSize.width;

for( i = cascadedetect_INTEGER_440_1_0; i < nscales; i+=cascadedetect_INTEGER_440_2_1 )
{
FeatureEvaluator::ScaleData& s = scaleData->at(i);
if( !recalcOptFeatures && fabs(s.scale - _scales[i]) > FLT_EPSILON*cascadedetect_INTEGER_443_1_100*_scales[i] )
recalcOptFeatures = true;
float sc = _scales[i];
Size sz;
sz.width = cvRound(imgsz.width/sc);
sz.height = cvRound(imgsz.height/sc);
s.ystep = sc >= cascadedetect_INTEGER_449_1_2 ? cascadedetect_INTEGER_449_2_1 : cascadedetect_INTEGER_449_3_2;
s.scale = sc;
s.szi = Size(sz.width+cascadedetect_INTEGER_451_1_1, sz.height+cascadedetect_INTEGER_451_2_1);

if( i == cascadedetect_INTEGER_453_1_0 )
{
layer_dy = s.szi.height;
}

if( layer_ofs.x + s.szi.width > sbufSize.width )
{
layer_ofs = Point(cascadedetect_INTEGER_460_1_0, layer_ofs.y + layer_dy);
layer_dy = s.szi.height;
}
s.layer_ofs = layer_ofs.y*sbufSize.width + layer_ofs.x;
layer_ofs.x += s.szi.width;
}

layer_ofs.y += layer_dy;
sbufSize.height = std::max(sbufSize.height, layer_ofs.y);
recalcOptFeatures = recalcOptFeatures || sbufSize.height != prevBufSize.height;
return recalcOptFeatures;
}


bool FeatureEvaluator::setImage( InputArray _image, const std::vector<float>& _scales )
{
Size imgsz = _image.size();
bool recalcOptFeatures = updateScaleData(imgsz, _scales);

size_t i, nscales = scaleData->size();
if (nscales == cascadedetect_INTEGER_480_1_0)
{
return false;
}
Size sz0 = scaleData->at(cascadedetect_INTEGER_484_1_0).szi;
sz0 = Size(std::max(rbuf.cols, (int)alignSize(sz0.width, cascadedetect_INTEGER_485_1_16)), std::max(rbuf.rows, sz0.height));

if (recalcOptFeatures)
{
computeOptFeatures();
copyVectorToUMat(*scaleData, uscaleData);
}

if (_image.isUMat() && localSize.area() > cascadedetect_INTEGER_493_1_0)
{
usbuf.create(sbufSize.height*nchannels, sbufSize.width, CV_32S);
urbuf.create(sz0, CV_8U);

for (i = cascadedetect_INTEGER_498_1_0; i < nscales; i+=cascadedetect_INTEGER_498_2_1)
{
const ScaleData& s = scaleData->at(i);
UMat dst(urbuf, Rect(cascadedetect_INTEGER_501_1_0, cascadedetect_INTEGER_501_2_0, s.szi.width - cascadedetect_INTEGER_501_3_1, s.szi.height - cascadedetect_INTEGER_501_4_1));
resize(_image, dst, dst.size(), 1. / s.scale, 1. / s.scale, INTER_LINEAR);
computeChannels((int)i, dst);
}
sbufFlag = USBUF_VALID;
}
else
{
Mat image = _image.getMat();
sbuf.create(sbufSize.height*nchannels, sbufSize.width, CV_32S);
rbuf.create(sz0, CV_8U);

for (i = cascadedetect_INTEGER_513_1_0; i < nscales; i+=cascadedetect_INTEGER_513_2_1)
{
const ScaleData& s = scaleData->at(i);
Mat dst(s.szi.height - cascadedetect_INTEGER_516_1_1, s.szi.width - cascadedetect_INTEGER_516_2_1, CV_8U, rbuf.ptr());
resize(image, dst, dst.size(), 1. / s.scale, 1. / s.scale, INTER_LINEAR);
computeChannels((int)i, dst);
}
sbufFlag = SBUF_VALID;
}

return true;
}

//----------------------------------------------  HaarEvaluator ---------------------------------------

bool HaarEvaluator::Feature :: read( const FileNode& node )
{
FileNode rnode = node[CC_RECTS];
FileNodeIterator it = rnode.begin(), it_end = rnode.end();

int ri;
for( ri = cascadedetect_INTEGER_534_1_0; ri < RECT_NUM; ri+=cascadedetect_INTEGER_534_2_1 )
{
rect[ri].r = Rect();
rect[ri].weight = 0.f;
}

for(ri = cascadedetect_INTEGER_540_1_0; it != it_end; ++it, ri+=cascadedetect_INTEGER_540_2_1)
{
FileNodeIterator it2 = (*it).begin();
it2 >> rect[ri].r.x >> rect[ri].r.y >>
rect[ri].r.width >> rect[ri].r.height >> rect[ri].weight;
}

tilted = (int)node[CC_TILTED] != cascadedetect_INTEGER_547_1_0;
return true;
}

HaarEvaluator::HaarEvaluator()
{
optfeaturesPtr = cascadedetect_INTEGER_553_1_0;
pwin = cascadedetect_INTEGER_554_1_0;
localSize = Size(cascadedetect_INTEGER_555_1_4, cascadedetect_INTEGER_555_2_2);
lbufSize = Size(cascadedetect_INTEGER_556_1_0, cascadedetect_INTEGER_556_2_0);
nchannels = cascadedetect_INTEGER_557_1_0;
tofs = cascadedetect_INTEGER_558_1_0;
}

HaarEvaluator::~HaarEvaluator()
{
}

bool HaarEvaluator::read(const FileNode& node, Size _origWinSize)
{
if (!FeatureEvaluator::read(node, _origWinSize))
return false;
size_t i, n = node.size();
CV_Assert(n > cascadedetect_INTEGER_570_1_0);
if(features.empty())
features = makePtr<std::vector<Feature> >();
if(optfeatures.empty())
optfeatures = makePtr<std::vector<OptFeature> >();
if (optfeatures_lbuf.empty())
optfeatures_lbuf = makePtr<std::vector<OptFeature> >();
features->resize(n);
FileNodeIterator it = node.begin();
hasTiltedFeatures = false;
std::vector<Feature>& ff = *features;
sbufSize = Size();
ufbuf.release();

for(i = cascadedetect_INTEGER_584_1_0; i < n; i+=cascadedetect_INTEGER_584_2_1, ++it)
{
if(!ff[i].read(*it))
return false;
if( ff[i].tilted )
hasTiltedFeatures = true;
}
nchannels = hasTiltedFeatures ? cascadedetect_INTEGER_591_1_3 : cascadedetect_INTEGER_591_2_2;
normrect = Rect(cascadedetect_INTEGER_592_1_1, cascadedetect_INTEGER_592_2_1, origWinSize.width - cascadedetect_INTEGER_592_3_2, origWinSize.height - cascadedetect_INTEGER_592_4_2);

localSize = lbufSize = Size(cascadedetect_INTEGER_594_1_0, cascadedetect_INTEGER_594_2_0);
if (ocl::haveOpenCL())
{
if (ocl::Device::getDefault().isAMD() || ocl::Device::getDefault().isIntel())
{
localSize = Size(cascadedetect_INTEGER_599_1_8, cascadedetect_INTEGER_599_2_8);
lbufSize = Size(origWinSize.width + localSize.width,
origWinSize.height + localSize.height);
if (lbufSize.area() > cascadedetect_INTEGER_602_1_1024)
lbufSize = Size(cascadedetect_INTEGER_603_1_0, cascadedetect_INTEGER_603_2_0);
}
}

return true;
}

Ptr<FeatureEvaluator> HaarEvaluator::clone() const
{
Ptr<HaarEvaluator> ret = makePtr<HaarEvaluator>();
*ret = *this;
return ret;
}


void HaarEvaluator::computeChannels(int scaleIdx, InputArray img)
{
const ScaleData& s = scaleData->at(scaleIdx);
sqofs = hasTiltedFeatures ? sbufSize.area() * cascadedetect_INTEGER_621_1_2 : sbufSize.area();

if (img.isUMat())
{
int sx = s.layer_ofs % sbufSize.width;
int sy = s.layer_ofs / sbufSize.width;
int sqy = sy + (sqofs / sbufSize.width);
UMat sum(usbuf, Rect(sx, sy, s.szi.width, s.szi.height));
UMat sqsum(usbuf, Rect(sx, sqy, s.szi.width, s.szi.height));
sqsum.flags = (sqsum.flags & ~UMat::DEPTH_MASK) | CV_32S;

if (hasTiltedFeatures)
{
int sty = sy + (tofs / sbufSize.width);
UMat tilted(usbuf, Rect(sx, sty, s.szi.width, s.szi.height));
integral(img, sum, sqsum, tilted, CV_32S, CV_32S);
}
else
{
UMatData* u = sqsum.u;
integral(img, sum, sqsum, noArray(), CV_32S, CV_32S);
CV_Assert(sqsum.u == u && sqsum.size() == s.szi && sqsum.type()==CV_32S);
}
}
else
{
Mat sum(s.szi, CV_32S, sbuf.ptr<int>() + s.layer_ofs, sbuf.step);
Mat sqsum(s.szi, CV_32S, sum.ptr<int>() + sqofs, sbuf.step);

if (hasTiltedFeatures)
{
Mat tilted(s.szi, CV_32S, sum.ptr<int>() + tofs, sbuf.step);
integral(img, sum, sqsum, tilted, CV_32S, CV_32S);
}
else
integral(img, sum, sqsum, noArray(), CV_32S, CV_32S);
}
}

void HaarEvaluator::computeOptFeatures()
{
if (hasTiltedFeatures)
tofs = sbufSize.area();

int sstep = sbufSize.width;
CV_SUM_OFS( nofs[cascadedetect_INTEGER_666_1_0], nofs[cascadedetect_INTEGER_666_2_1], nofs[cascadedetect_INTEGER_666_3_2], nofs[cascadedetect_INTEGER_666_4_3], cascadedetect_INTEGER_666_5_0, normrect, sstep );

size_t fi, nfeatures = features->size();
const std::vector<Feature>& ff = *features;
optfeatures->resize(nfeatures);
optfeaturesPtr = &(*optfeatures)[cascadedetect_INTEGER_671_1_0];
for( fi = cascadedetect_INTEGER_672_1_0; fi < nfeatures; fi+=cascadedetect_INTEGER_672_2_1 )
optfeaturesPtr[fi].setOffsets( ff[fi], sstep, tofs );
optfeatures_lbuf->resize(nfeatures);

for( fi = cascadedetect_INTEGER_676_1_0; fi < nfeatures; fi+=cascadedetect_INTEGER_676_2_1 )
optfeatures_lbuf->at(fi).setOffsets(ff[fi], lbufSize.width > cascadedetect_INTEGER_677_1_0 ? lbufSize.width : sstep, tofs);

copyVectorToUMat(*optfeatures_lbuf, ufbuf);
}

bool HaarEvaluator::setWindow( Point pt, int scaleIdx )
{
const ScaleData& s = getScaleData(scaleIdx);

if( pt.x < cascadedetect_INTEGER_686_1_0 || pt.y < cascadedetect_INTEGER_686_2_0 ||
pt.x + origWinSize.width >= s.szi.width ||
pt.y + origWinSize.height >= s.szi.height )
return false;

pwin = &sbuf.at<int>(pt) + s.layer_ofs;
const int* pq = (const int*)(pwin + sqofs);
int valsum = CALC_SUM_OFS(nofs, pwin);
unsigned valsqsum = (unsigned)(CALC_SUM_OFS(nofs, pq));

double area = normrect.area();
double nf = area * valsqsum - (double)valsum * valsum;
if( nf > 0. )
{
nf = std::sqrt(nf);
varianceNormFactor = (float)(1./nf);
return area*varianceNormFactor < 1e-1;
}
else
{
varianceNormFactor = 1.f;
return false;
}
}


void HaarEvaluator::OptFeature::setOffsets( const Feature& _f, int step, int _tofs )
{
weight[cascadedetect_INTEGER_714_1_0] = _f.rect[cascadedetect_INTEGER_714_2_0].weight;
weight[cascadedetect_INTEGER_715_1_1] = _f.rect[cascadedetect_INTEGER_715_2_1].weight;
weight[cascadedetect_INTEGER_716_1_2] = _f.rect[cascadedetect_INTEGER_716_2_2].weight;

if( _f.tilted )
{
CV_TILTED_OFS( ofs[cascadedetect_INTEGER_720_1_0][cascadedetect_INTEGER_720_2_0], ofs[cascadedetect_INTEGER_720_3_0][cascadedetect_INTEGER_720_4_1], ofs[cascadedetect_INTEGER_720_5_0][cascadedetect_INTEGER_720_6_2], ofs[cascadedetect_INTEGER_720_7_0][cascadedetect_INTEGER_720_8_3], _tofs, _f.rect[cascadedetect_INTEGER_720_9_0].r, step );
CV_TILTED_OFS( ofs[cascadedetect_INTEGER_721_1_1][cascadedetect_INTEGER_721_2_0], ofs[cascadedetect_INTEGER_721_3_1][cascadedetect_INTEGER_721_4_1], ofs[cascadedetect_INTEGER_721_5_1][cascadedetect_INTEGER_721_6_2], ofs[cascadedetect_INTEGER_721_7_1][cascadedetect_INTEGER_721_8_3], _tofs, _f.rect[cascadedetect_INTEGER_721_9_1].r, step );
CV_TILTED_OFS( ofs[cascadedetect_INTEGER_722_1_2][cascadedetect_INTEGER_722_2_0], ofs[cascadedetect_INTEGER_722_3_2][cascadedetect_INTEGER_722_4_1], ofs[cascadedetect_INTEGER_722_5_2][cascadedetect_INTEGER_722_6_2], ofs[cascadedetect_INTEGER_722_7_2][cascadedetect_INTEGER_722_8_3], _tofs, _f.rect[cascadedetect_INTEGER_722_9_2].r, step );
}
else
{
CV_SUM_OFS( ofs[cascadedetect_INTEGER_726_1_0][cascadedetect_INTEGER_726_2_0], ofs[cascadedetect_INTEGER_726_3_0][cascadedetect_INTEGER_726_4_1], ofs[cascadedetect_INTEGER_726_5_0][cascadedetect_INTEGER_726_6_2], ofs[cascadedetect_INTEGER_726_7_0][cascadedetect_INTEGER_726_8_3], cascadedetect_INTEGER_726_9_0, _f.rect[cascadedetect_INTEGER_726_10_0].r, step );
CV_SUM_OFS( ofs[cascadedetect_INTEGER_727_1_1][cascadedetect_INTEGER_727_2_0], ofs[cascadedetect_INTEGER_727_3_1][cascadedetect_INTEGER_727_4_1], ofs[cascadedetect_INTEGER_727_5_1][cascadedetect_INTEGER_727_6_2], ofs[cascadedetect_INTEGER_727_7_1][cascadedetect_INTEGER_727_8_3], cascadedetect_INTEGER_727_9_0, _f.rect[cascadedetect_INTEGER_727_10_1].r, step );
CV_SUM_OFS( ofs[cascadedetect_INTEGER_728_1_2][cascadedetect_INTEGER_728_2_0], ofs[cascadedetect_INTEGER_728_3_2][cascadedetect_INTEGER_728_4_1], ofs[cascadedetect_INTEGER_728_5_2][cascadedetect_INTEGER_728_6_2], ofs[cascadedetect_INTEGER_728_7_2][cascadedetect_INTEGER_728_8_3], cascadedetect_INTEGER_728_9_0, _f.rect[cascadedetect_INTEGER_728_10_2].r, step );
}
}

Rect HaarEvaluator::getNormRect() const
{
return normrect;
}

int HaarEvaluator::getSquaresOffset() const
{
return sqofs;
}

//----------------------------------------------  LBPEvaluator -------------------------------------
bool LBPEvaluator::Feature :: read(const FileNode& node )
{
FileNode rnode = node[CC_RECT];
FileNodeIterator it = rnode.begin();
it >> rect.x >> rect.y >> rect.width >> rect.height;
return true;
}

LBPEvaluator::LBPEvaluator()
{
features = makePtr<std::vector<Feature> >();
optfeatures = makePtr<std::vector<OptFeature> >();
scaleData = makePtr<std::vector<ScaleData> >();
}

LBPEvaluator::~LBPEvaluator()
{
}

bool LBPEvaluator::read( const FileNode& node, Size _origWinSize )
{
if (!FeatureEvaluator::read(node, _origWinSize))
return false;
if(features.empty())
features = makePtr<std::vector<Feature> >();
if(optfeatures.empty())
optfeatures = makePtr<std::vector<OptFeature> >();
if (optfeatures_lbuf.empty())
optfeatures_lbuf = makePtr<std::vector<OptFeature> >();

features->resize(node.size());
optfeaturesPtr = cascadedetect_INTEGER_774_1_0;
FileNodeIterator it = node.begin(), it_end = node.end();
std::vector<Feature>& ff = *features;
for(int i = cascadedetect_INTEGER_777_1_0; it != it_end; ++it, i+=cascadedetect_INTEGER_777_2_1)
{
if(!ff[i].read(*it))
return false;
}
nchannels = cascadedetect_INTEGER_782_1_1;
localSize = lbufSize = Size(cascadedetect_INTEGER_783_1_0, cascadedetect_INTEGER_783_2_0);
if (ocl::haveOpenCL())
localSize = Size(cascadedetect_INTEGER_785_1_8, cascadedetect_INTEGER_785_2_8);

return true;
}

Ptr<FeatureEvaluator> LBPEvaluator::clone() const
{
Ptr<LBPEvaluator> ret = makePtr<LBPEvaluator>();
*ret = *this;
return ret;
}

void LBPEvaluator::computeChannels(int scaleIdx, InputArray _img)
{
const ScaleData& s = scaleData->at(scaleIdx);

if (_img.isUMat())
{
int sx = s.layer_ofs % sbufSize.width;
int sy = s.layer_ofs / sbufSize.width;
UMat sum(usbuf, Rect(sx, sy, s.szi.width, s.szi.height));
integral(_img, sum, noArray(), noArray(), CV_32S);
}
else
{
Mat sum(s.szi, CV_32S, sbuf.ptr<int>() + s.layer_ofs, sbuf.step);
integral(_img, sum, noArray(), noArray(), CV_32S);
}
}

void LBPEvaluator::computeOptFeatures()
{
int sstep = sbufSize.width;

size_t fi, nfeatures = features->size();
const std::vector<Feature>& ff = *features;
optfeatures->resize(nfeatures);
optfeaturesPtr = &(*optfeatures)[cascadedetect_INTEGER_822_1_0];
for( fi = cascadedetect_INTEGER_823_1_0; fi < nfeatures; fi+=cascadedetect_INTEGER_823_2_1 )
optfeaturesPtr[fi].setOffsets( ff[fi], sstep );
copyVectorToUMat(*optfeatures, ufbuf);
}


void LBPEvaluator::OptFeature::setOffsets( const Feature& _f, int step )
{
Rect tr = _f.rect;
int w0 = tr.width;
int h0 = tr.height;

CV_SUM_OFS( ofs[cascadedetect_INTEGER_835_1_0], ofs[cascadedetect_INTEGER_835_2_1], ofs[cascadedetect_INTEGER_835_3_4], ofs[cascadedetect_INTEGER_835_4_5], cascadedetect_INTEGER_835_5_0, tr, step );
tr.x += cascadedetect_INTEGER_836_1_2*w0;
CV_SUM_OFS( ofs[cascadedetect_INTEGER_837_1_2], ofs[cascadedetect_INTEGER_837_2_3], ofs[cascadedetect_INTEGER_837_3_6], ofs[cascadedetect_INTEGER_837_4_7], cascadedetect_INTEGER_837_5_0, tr, step );
tr.y += cascadedetect_INTEGER_838_1_2*h0;
CV_SUM_OFS( ofs[cascadedetect_INTEGER_839_1_10], ofs[cascadedetect_INTEGER_839_2_11], ofs[cascadedetect_INTEGER_839_3_14], ofs[cascadedetect_INTEGER_839_4_15], cascadedetect_INTEGER_839_5_0, tr, step );
tr.x -= cascadedetect_INTEGER_840_1_2*w0;
CV_SUM_OFS( ofs[cascadedetect_INTEGER_841_1_8], ofs[cascadedetect_INTEGER_841_2_9], ofs[cascadedetect_INTEGER_841_3_12], ofs[cascadedetect_INTEGER_841_4_13], cascadedetect_INTEGER_841_5_0, tr, step );
}


bool LBPEvaluator::setWindow( Point pt, int scaleIdx )
{
CV_Assert(cascadedetect_INTEGER_847_1_0 <= scaleIdx && scaleIdx < (int)scaleData->size());
const ScaleData& s = scaleData->at(scaleIdx);

if( pt.x < cascadedetect_INTEGER_850_1_0 || pt.y < cascadedetect_INTEGER_850_2_0 ||
pt.x + origWinSize.width >= s.szi.width ||
pt.y + origWinSize.height >= s.szi.height )
return false;

pwin = &sbuf.at<int>(pt) + s.layer_ofs;
return true;
}


Ptr<FeatureEvaluator> FeatureEvaluator::create( int featureType )
{
return featureType == HAAR ? Ptr<FeatureEvaluator>(new HaarEvaluator) :
featureType == LBP ? Ptr<FeatureEvaluator>(new LBPEvaluator) :
Ptr<FeatureEvaluator>();
}

//---------------------------------------- Classifier Cascade --------------------------------------------

CascadeClassifierImpl::CascadeClassifierImpl()
{
}

CascadeClassifierImpl::~CascadeClassifierImpl()
{
}

bool CascadeClassifierImpl::empty() const
{
return !oldCascade && data.stages.empty();
}

bool CascadeClassifierImpl::load(const String& filename)
{
oldCascade.release();
data = Data();
featureEvaluator.release();

FileStorage fs(filename, FileStorage::READ);
if( !fs.isOpened() )
return false;

if( read_(fs.getFirstTopLevelNode()) )
return true;

fs.release();

oldCascade.reset((CvHaarClassifierCascade*)cvLoad(filename.c_str(), cascadedetect_INTEGER_897_1_0, cascadedetect_INTEGER_897_2_0, cascadedetect_INTEGER_897_3_0));
return !oldCascade.empty();
}

void CascadeClassifierImpl::read(const FileNode& node)
{
read_(node);
}

int CascadeClassifierImpl::runAt( Ptr<FeatureEvaluator>& evaluator, Point pt, int scaleIdx, double& weight )
{
assert( !oldCascade &&
(data.featureType == FeatureEvaluator::HAAR ||
data.featureType == FeatureEvaluator::LBP ||
data.featureType == FeatureEvaluator::HOG) );

if( !evaluator->setWindow(pt, scaleIdx) )
return -cascadedetect_INTEGER_914_1_1;
if( data.maxNodesPerTree == cascadedetect_INTEGER_915_1_1 )
{
if( data.featureType == FeatureEvaluator::HAAR )
return predictOrderedStump<HaarEvaluator>( *this, evaluator, weight );
else if( data.featureType == FeatureEvaluator::LBP )
return predictCategoricalStump<LBPEvaluator>( *this, evaluator, weight );
else
return -cascadedetect_INTEGER_922_1_2;
}
else
{
if( data.featureType == FeatureEvaluator::HAAR )
return predictOrdered<HaarEvaluator>( *this, evaluator, weight );
else if( data.featureType == FeatureEvaluator::LBP )
return predictCategorical<LBPEvaluator>( *this, evaluator, weight );
else
return -cascadedetect_INTEGER_931_1_2;
}
}

void CascadeClassifierImpl::setMaskGenerator(const Ptr<MaskGenerator>& _maskGenerator)
{
maskGenerator=_maskGenerator;
}
Ptr<CascadeClassifierImpl::MaskGenerator> CascadeClassifierImpl::getMaskGenerator()
{
return maskGenerator;
}

Ptr<BaseCascadeClassifier::MaskGenerator> createFaceDetectionMaskGenerator()
{
#ifdef HAVE_TEGRA_OPTIMIZATION
if (tegra::useTegra())
return tegra::getCascadeClassifierMaskGenerator();
#endif
return Ptr<BaseCascadeClassifier::MaskGenerator>();
}

class CascadeClassifierInvoker : public ParallelLoopBody
{
public:
CascadeClassifierInvoker( CascadeClassifierImpl& _cc, int _nscales, int _nstripes,
const FeatureEvaluator::ScaleData* _scaleData,
const int* _stripeSizes, std::vector<Rect>& _vec,
std::vector<int>& _levels, std::vector<double>& _weights,
bool outputLevels, const Mat& _mask, Mutex* _mtx)
{
classifier = &_cc;
nscales = _nscales;
nstripes = _nstripes;
scaleData = _scaleData;
stripeSizes = _stripeSizes;
rectangles = &_vec;
rejectLevels = outputLevels ? &_levels : cascadedetect_INTEGER_968_1_0;
levelWeights = outputLevels ? &_weights : cascadedetect_INTEGER_969_1_0;
mask = _mask;
mtx = _mtx;
}

void operator()(const Range& range) const
{
Ptr<FeatureEvaluator> evaluator = classifier->featureEvaluator->clone();
double gypWeight = 0.;
Size origWinSize = classifier->data.origWinSize;

for( int scaleIdx = cascadedetect_INTEGER_980_1_0; scaleIdx < nscales; scaleIdx+=cascadedetect_INTEGER_980_2_1 )
{
const FeatureEvaluator::ScaleData& s = scaleData[scaleIdx];
float scalingFactor = s.scale;
int yStep = s.ystep;
int stripeSize = stripeSizes[scaleIdx];
int y0 = range.start*stripeSize;
Size szw = s.getWorkingSize(origWinSize);
int y1 = std::min(range.end*stripeSize, szw.height);
Size winSize(cvRound(origWinSize.width * scalingFactor),
cvRound(origWinSize.height * scalingFactor));

for( int y = y0; y < y1; y += yStep )
{
for( int x = cascadedetect_INTEGER_994_1_0; x < szw.width; x += yStep )
{
int result = classifier->runAt(evaluator, Point(x, y), scaleIdx, gypWeight);
if( rejectLevels )
{
if( result == cascadedetect_INTEGER_999_1_1 )
result = -(int)classifier->data.stages.size();
if( classifier->data.stages.size() + result == cascadedetect_INTEGER_1001_1_0 )
{
mtx->lock();
rectangles->push_back(Rect(cvRound(x*scalingFactor),
cvRound(y*scalingFactor),
winSize.width, winSize.height));
rejectLevels->push_back(-result);
levelWeights->push_back(gypWeight);
mtx->unlock();
}
}
else if( result > cascadedetect_INTEGER_1012_1_0 )
{
mtx->lock();
rectangles->push_back(Rect(cvRound(x*scalingFactor),
cvRound(y*scalingFactor),
winSize.width, winSize.height));
mtx->unlock();
}
if( result == cascadedetect_INTEGER_1020_1_0 )
x += yStep;
}
}
}
}

CascadeClassifierImpl* classifier;
std::vector<Rect>* rectangles;
int nscales, nstripes;
const FeatureEvaluator::ScaleData* scaleData;
const int* stripeSizes;
std::vector<int> *rejectLevels;
std::vector<double> *levelWeights;
std::vector<float> scales;
Mat mask;
Mutex* mtx;
};


struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };
struct getNeighbors { int operator ()(const CvAvgComp& e) const { return e.neighbors; } };

#ifdef HAVE_OPENCL
bool CascadeClassifierImpl::ocl_detectMultiScaleNoGrouping( const std::vector<float>& scales,
std::vector<Rect>& candidates )
{
int featureType = getFeatureType();
std::vector<UMat> bufs;
featureEvaluator->getUMats(bufs);
Size localsz = featureEvaluator->getLocalSize();
if( localsz.area() == cascadedetect_INTEGER_1051_1_0 )
return false;
Size lbufSize = featureEvaluator->getLocalBufSize();
size_t localsize[] = { (size_t)localsz.width, (size_t)localsz.height };
const int grp_per_CU = cascadedetect_INTEGER_1055_1_12;
size_t globalsize[] = { grp_per_CU*ocl::Device::getDefault().maxComputeUnits()*localsize[cascadedetect_INTEGER_1056_1_0], localsize[cascadedetect_INTEGER_1056_2_1] };
bool ok = false;

ufacepos.create(cascadedetect_INTEGER_1059_1_1, MAX_FACES*cascadedetect_INTEGER_1059_2_3+cascadedetect_INTEGER_1059_3_1, CV_32S);
UMat ufacepos_count(ufacepos, Rect(cascadedetect_INTEGER_1060_1_0, cascadedetect_INTEGER_1060_2_0, cascadedetect_INTEGER_1060_3_1, cascadedetect_INTEGER_1060_4_1));
ufacepos_count.setTo(Scalar::all(cascadedetect_INTEGER_1061_1_0));

if( ustages.empty() )
{
copyVectorToUMat(data.stages, ustages);
if (!data.stumps.empty())
copyVectorToUMat(data.stumps, unodes);
else
copyVectorToUMat(data.nodes, unodes);
copyVectorToUMat(data.leaves, uleaves);
if( !data.subsets.empty() )
copyVectorToUMat(data.subsets, usubsets);
}

int nstages = (int)data.stages.size();
int splitstage_ocl = cascadedetect_INTEGER_1076_1_1;

if( featureType == FeatureEvaluator::HAAR )
{
Ptr<HaarEvaluator> haar = featureEvaluator.dynamicCast<HaarEvaluator>();
if( haar.empty() )
return false;

if( haarKernel.empty() )
{
String opts;
if (lbufSize.area())
opts = format("-D LOCAL_SIZE_X=%d -D LOCAL_SIZE_Y=%d -D SUM_BUF_SIZE=%d -D SUM_BUF_STEP=%d -D NODE_COUNT=%d -D SPLIT_STAGE=%d -D N_STAGES=%d -D MAX_FACES=%d -D HAAR",
localsz.width, localsz.height, lbufSize.area(), lbufSize.width, data.maxNodesPerTree, splitstage_ocl, nstages, MAX_FACES);
else
opts = format("-D LOCAL_SIZE_X=%d -D LOCAL_SIZE_Y=%d -D NODE_COUNT=%d -D SPLIT_STAGE=%d -D N_STAGES=%d -D MAX_FACES=%d -D HAAR",
localsz.width, localsz.height, data.maxNodesPerTree, splitstage_ocl, nstages, MAX_FACES);
haarKernel.create("runHaarClassifier", ocl::objdetect::cascadedetect_oclsrc, opts);
if( haarKernel.empty() )
return false;
}

Rect normrect = haar->getNormRect();
int sqofs = haar->getSquaresOffset();

haarKernel.args((int)scales.size(),
ocl::KernelArg::PtrReadOnly(bufs[cascadedetect_INTEGER_1102_1_0]), // scaleData
ocl::KernelArg::ReadOnlyNoSize(bufs[cascadedetect_INTEGER_1103_1_1]), // sum
ocl::KernelArg::PtrReadOnly(bufs[cascadedetect_INTEGER_1104_1_2]), // optfeatures

// cascade classifier
ocl::KernelArg::PtrReadOnly(ustages),
ocl::KernelArg::PtrReadOnly(unodes),
ocl::KernelArg::PtrReadOnly(uleaves),

ocl::KernelArg::PtrWriteOnly(ufacepos), // positions
normrect, sqofs, data.origWinSize);
ok = haarKernel.run(cascadedetect_INTEGER_1113_1_2, globalsize, localsize, true);
}
else if( featureType == FeatureEvaluator::LBP )
{
if (data.maxNodesPerTree > cascadedetect_INTEGER_1117_1_1)
return false;

Ptr<LBPEvaluator> lbp = featureEvaluator.dynamicCast<LBPEvaluator>();
if( lbp.empty() )
return false;

if( lbpKernel.empty() )
{
String opts;
if (lbufSize.area())
opts = format("-D LOCAL_SIZE_X=%d -D LOCAL_SIZE_Y=%d -D SUM_BUF_SIZE=%d -D SUM_BUF_STEP=%d -D SPLIT_STAGE=%d -D N_STAGES=%d -D MAX_FACES=%d -D LBP",
localsz.width, localsz.height, lbufSize.area(), lbufSize.width, splitstage_ocl, nstages, MAX_FACES);
else
opts = format("-D LOCAL_SIZE_X=%d -D LOCAL_SIZE_Y=%d -D SPLIT_STAGE=%d -D N_STAGES=%d -D MAX_FACES=%d -D LBP",
localsz.width, localsz.height, splitstage_ocl, nstages, MAX_FACES);
lbpKernel.create("runLBPClassifierStumpSimple", ocl::objdetect::cascadedetect_oclsrc, opts);
if( lbpKernel.empty() )
return false;
}

int subsetSize = (data.ncategories + cascadedetect_INTEGER_1138_1_31)/cascadedetect_INTEGER_1138_2_32;
lbpKernel.args((int)scales.size(),
ocl::KernelArg::PtrReadOnly(bufs[cascadedetect_INTEGER_1140_1_0]), // scaleData
ocl::KernelArg::ReadOnlyNoSize(bufs[cascadedetect_INTEGER_1141_1_1]), // sum
ocl::KernelArg::PtrReadOnly(bufs[cascadedetect_INTEGER_1142_1_2]), // optfeatures

// cascade classifier
ocl::KernelArg::PtrReadOnly(ustages),
ocl::KernelArg::PtrReadOnly(unodes),
ocl::KernelArg::PtrReadOnly(usubsets),
subsetSize,

ocl::KernelArg::PtrWriteOnly(ufacepos), // positions
data.origWinSize);

ok = lbpKernel.run(cascadedetect_INTEGER_1153_1_2, globalsize, localsize, true);
}

if( ok )
{
Mat facepos = ufacepos.getMat(ACCESS_READ);
const int* fptr = facepos.ptr<int>();
int nfaces = fptr[cascadedetect_INTEGER_1160_1_0];
nfaces = std::min(nfaces, (int)MAX_FACES);

for( int i = cascadedetect_INTEGER_1163_1_0; i < nfaces; i+=cascadedetect_INTEGER_1163_2_1 )
{
const FeatureEvaluator::ScaleData& s = featureEvaluator->getScaleData(fptr[i*cascadedetect_INTEGER_1165_1_3 + cascadedetect_INTEGER_1165_2_1]);
candidates.push_back(Rect(cvRound(fptr[i*cascadedetect_INTEGER_1166_1_3 + cascadedetect_INTEGER_1166_2_2]*s.scale),
cvRound(fptr[i*cascadedetect_INTEGER_1167_1_3 + cascadedetect_INTEGER_1167_2_3]*s.scale),
cvRound(data.origWinSize.width*s.scale),
cvRound(data.origWinSize.height*s.scale)));
}
}
return ok;
}
#endif

bool CascadeClassifierImpl::isOldFormatCascade() const
{
return !oldCascade.empty();
}

int CascadeClassifierImpl::getFeatureType() const
{
return featureEvaluator->getFeatureType();
}

Size CascadeClassifierImpl::getOriginalWindowSize() const
{
return data.origWinSize;
}

void* CascadeClassifierImpl::getOldCascade()
{
return oldCascade;
}

static void detectMultiScaleOldFormat( const Mat& image, Ptr<CvHaarClassifierCascade> oldCascade,
std::vector<Rect>& objects,
std::vector<int>& rejectLevels,
std::vector<double>& levelWeights,
std::vector<CvAvgComp>& vecAvgComp,
double scaleFactor, int minNeighbors,
int flags, Size minObjectSize, Size maxObjectSize,
bool outputRejectLevels = false )
{
MemStorage storage(cvCreateMemStorage(cascadedetect_INTEGER_1205_1_0));
CvMat _image = image;
CvSeq* _objects = cvHaarDetectObjectsForROC( &_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
minNeighbors, flags, minObjectSize, maxObjectSize, outputRejectLevels );
Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
objects.resize(vecAvgComp.size());
std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());
}


void CascadeClassifierImpl::detectMultiScaleNoGrouping( InputArray _image, std::vector<Rect>& candidates,
std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
double scaleFactor, Size minObjectSize, Size maxObjectSize,
bool outputRejectLevels )
{
Size imgsz = _image.size();

Mat grayImage;
_InputArray gray;

candidates.clear();
rejectLevels.clear();
levelWeights.clear();

if( maxObjectSize.height == cascadedetect_INTEGER_1229_1_0 || maxObjectSize.width == cascadedetect_INTEGER_1229_2_0 )
maxObjectSize = imgsz;

#ifdef HAVE_OPENCL
bool use_ocl = tryOpenCL && ocl::useOpenCL() &&
featureEvaluator->getLocalSize().area() > cascadedetect_INTEGER_1234_1_0 &&
ocl::Device::getDefault().type() != ocl::Device::TYPE_CPU &&
(data.minNodesPerTree == data.maxNodesPerTree) &&
!isOldFormatCascade() &&
maskGenerator.empty() &&
!outputRejectLevels;
#endif

/*if( use_ocl )
{
if (_image.channels() > cascadedetect_INTEGER_1244_1_1)
cvtColor(_image, ugrayImage, COLOR_BGR2GRAY);
else if (_image.isUMat())
ugrayImage = _image.getUMat();
else
_image.copyTo(ugrayImage);
gray = ugrayImage;
}
else*/
{
if (_image.channels() > cascadedetect_INTEGER_1254_1_1)
cvtColor(_image, grayImage, COLOR_BGR2GRAY);
else if (_image.isMat())
grayImage = _image.getMat();
else
_image.copyTo(grayImage);
gray = grayImage;
}

std::vector<float> scales;
scales.reserve(cascadedetect_INTEGER_1264_1_1024);

for( double factor = cascadedetect_INTEGER_1266_1_1; ; factor *= scaleFactor )
{
Size originalWindowSize = getOriginalWindowSize();

Size windowSize( cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
if( windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height ||
windowSize.width > imgsz.width || windowSize.height > imgsz.height )
break;
if( windowSize.width < minObjectSize.width || windowSize.height < minObjectSize.height )
continue;
scales.push_back((float)factor);
}

if( scales.size() == cascadedetect_INTEGER_1279_1_0 || !featureEvaluator->setImage(gray, scales) )
return;

#ifdef HAVE_OPENCL
// OpenCL code
CV_OCL_RUN(use_ocl, ocl_detectMultiScaleNoGrouping( scales, candidates ))

tryOpenCL = false;
#endif

// CPU code
featureEvaluator->getMats();
{
Mat currentMask;
if (maskGenerator)
currentMask = maskGenerator->generateMask(gray.getMat());

size_t i, nscales = scales.size();
cv::AutoBuffer<int> stripeSizeBuf(nscales);
int* stripeSizes = stripeSizeBuf;
const FeatureEvaluator::ScaleData* s = &featureEvaluator->getScaleData(cascadedetect_INTEGER_1299_1_0);
Size szw = s->getWorkingSize(data.origWinSize);
int nstripes = cvCeil(szw.width/32.);
for( i = cascadedetect_INTEGER_1302_1_0; i < nscales; i+=cascadedetect_INTEGER_1302_2_1 )
{
szw = s[i].getWorkingSize(data.origWinSize);
stripeSizes[i] = std::max((szw.height/s[i].ystep + nstripes-cascadedetect_INTEGER_1305_1_1)/nstripes, cascadedetect_INTEGER_1305_2_1)*s[i].ystep;
}

CascadeClassifierInvoker invoker(*this, (int)nscales, nstripes, s, stripeSizes,
candidates, rejectLevels, levelWeights,
outputRejectLevels, currentMask, &mtx);
parallel_for_(Range(cascadedetect_INTEGER_1311_1_0, nstripes), invoker);
}
}


void CascadeClassifierImpl::detectMultiScale( InputArray _image, std::vector<Rect>& objects,
std::vector<int>& rejectLevels,
std::vector<double>& levelWeights,
double scaleFactor, int minNeighbors,
int flags, Size minObjectSize, Size maxObjectSize,
bool outputRejectLevels )
{
CV_Assert( scaleFactor > cascadedetect_INTEGER_1323_1_1 && _image.depth() == CV_8U );

if( empty() )
return;

if( isOldFormatCascade() )
{
Mat image = _image.getMat();
std::vector<CvAvgComp> fakeVecAvgComp;
detectMultiScaleOldFormat( image, oldCascade, objects, rejectLevels, levelWeights, fakeVecAvgComp, scaleFactor,
minNeighbors, flags, minObjectSize, maxObjectSize, outputRejectLevels );
}
else
{
detectMultiScaleNoGrouping( _image, objects, rejectLevels, levelWeights, scaleFactor, minObjectSize, maxObjectSize,
outputRejectLevels );
const double GROUP_EPS = 0.2;
if( outputRejectLevels )
{
groupRectangles( objects, rejectLevels, levelWeights, minNeighbors, GROUP_EPS );
}
else
{
groupRectangles( objects, minNeighbors, GROUP_EPS );
}
}
}

void CascadeClassifierImpl::detectMultiScale( InputArray _image, std::vector<Rect>& objects,
double scaleFactor, int minNeighbors,
int flags, Size minObjectSize, Size maxObjectSize)
{
std::vector<int> fakeLevels;
std::vector<double> fakeWeights;
detectMultiScale( _image, objects, fakeLevels, fakeWeights, scaleFactor,
minNeighbors, flags, minObjectSize, maxObjectSize );
}

void CascadeClassifierImpl::detectMultiScale( InputArray _image, std::vector<Rect>& objects,
std::vector<int>& numDetections, double scaleFactor,
int minNeighbors, int flags, Size minObjectSize,
Size maxObjectSize )
{
Mat image = _image.getMat();
CV_Assert( scaleFactor > cascadedetect_INTEGER_1367_1_1 && image.depth() == CV_8U );

if( empty() )
return;

std::vector<int> fakeLevels;
std::vector<double> fakeWeights;
if( isOldFormatCascade() )
{
std::vector<CvAvgComp> vecAvgComp;
detectMultiScaleOldFormat( image, oldCascade, objects, fakeLevels, fakeWeights, vecAvgComp, scaleFactor,
minNeighbors, flags, minObjectSize, maxObjectSize );
numDetections.resize(vecAvgComp.size());
std::transform(vecAvgComp.begin(), vecAvgComp.end(), numDetections.begin(), getNeighbors());
}
else
{
detectMultiScaleNoGrouping( image, objects, fakeLevels, fakeWeights, scaleFactor, minObjectSize, maxObjectSize );
const double GROUP_EPS = 0.2;
groupRectangles( objects, numDetections, minNeighbors, GROUP_EPS );
}
}


CascadeClassifierImpl::Data::Data()
{
stageType = featureType = ncategories = maxNodesPerTree = cascadedetect_INTEGER_1393_1_0;
}

bool CascadeClassifierImpl::Data::read(const FileNode &root)
{
static const float THRESHOLD_EPS = 1e-5f;

// load stage params
String stageTypeStr = (String)root[CC_STAGE_TYPE];
if( stageTypeStr == CC_BOOST )
stageType = BOOST;
else
return false;

String featureTypeStr = (String)root[CC_FEATURE_TYPE];
if( featureTypeStr == CC_HAAR )
featureType = FeatureEvaluator::HAAR;
else if( featureTypeStr == CC_LBP )
featureType = FeatureEvaluator::LBP;
else if( featureTypeStr == CC_HOG )
{
featureType = FeatureEvaluator::HOG;
CV_Error(Error::StsNotImplemented, "HOG cascade is not supported in 3.0");
}
else
return false;

origWinSize.width = (int)root[CC_WIDTH];
origWinSize.height = (int)root[CC_HEIGHT];
CV_Assert( origWinSize.height > cascadedetect_INTEGER_1422_1_0 && origWinSize.width > cascadedetect_INTEGER_1422_2_0 );

// load feature params
FileNode fn = root[CC_FEATURE_PARAMS];
if( fn.empty() )
return false;

ncategories = fn[CC_MAX_CAT_COUNT];
int subsetSize = (ncategories + cascadedetect_INTEGER_1430_1_31)/cascadedetect_INTEGER_1430_2_32,
nodeStep = cascadedetect_INTEGER_1431_1_3 + ( ncategories>cascadedetect_INTEGER_1431_2_0 ? subsetSize : cascadedetect_INTEGER_1431_3_1 );

// load stages
fn = root[CC_STAGES];
if( fn.empty() )
return false;

stages.reserve(fn.size());
classifiers.clear();
nodes.clear();
stumps.clear();

FileNodeIterator it = fn.begin(), it_end = fn.end();
minNodesPerTree = INT_MAX;
maxNodesPerTree = cascadedetect_INTEGER_1445_1_0;

for( int si = cascadedetect_INTEGER_1447_1_0; it != it_end; si+=cascadedetect_INTEGER_1447_2_1, ++it )
{
FileNode fns = *it;
Stage stage;
stage.threshold = (float)fns[CC_STAGE_THRESHOLD] - THRESHOLD_EPS;
fns = fns[CC_WEAK_CLASSIFIERS];
if(fns.empty())
return false;
stage.ntrees = (int)fns.size();
stage.first = (int)classifiers.size();
stages.push_back(stage);
classifiers.reserve(stages[si].first + stages[si].ntrees);

FileNodeIterator it1 = fns.begin(), it1_end = fns.end();
for( ; it1 != it1_end; ++it1 ) // weak trees
{
FileNode fnw = *it1;
FileNode internalNodes = fnw[CC_INTERNAL_NODES];
FileNode leafValues = fnw[CC_LEAF_VALUES];
if( internalNodes.empty() || leafValues.empty() )
return false;

DTree tree;
tree.nodeCount = (int)internalNodes.size()/nodeStep;
minNodesPerTree = std::min(minNodesPerTree, tree.nodeCount);
maxNodesPerTree = std::max(maxNodesPerTree, tree.nodeCount);

classifiers.push_back(tree);

nodes.reserve(nodes.size() + tree.nodeCount);
leaves.reserve(leaves.size() + leafValues.size());
if( subsetSize > cascadedetect_INTEGER_1478_1_0 )
subsets.reserve(subsets.size() + tree.nodeCount*subsetSize);

FileNodeIterator internalNodesIter = internalNodes.begin(), internalNodesEnd = internalNodes.end();

for( ; internalNodesIter != internalNodesEnd; ) // nodes
{
DTreeNode node;
node.left = (int)*internalNodesIter; ++internalNodesIter;
node.right = (int)*internalNodesIter; ++internalNodesIter;
node.featureIdx = (int)*internalNodesIter; ++internalNodesIter;
if( subsetSize > cascadedetect_INTEGER_1489_1_0 )
{
for( int j = cascadedetect_INTEGER_1491_1_0; j < subsetSize; j+=cascadedetect_INTEGER_1491_2_1, ++internalNodesIter )
subsets.push_back((int)*internalNodesIter);
node.threshold = 0.f;
}
else
{
node.threshold = (float)*internalNodesIter; ++internalNodesIter;
}
nodes.push_back(node);
}

internalNodesIter = leafValues.begin(), internalNodesEnd = leafValues.end();

for( ; internalNodesIter != internalNodesEnd; ++internalNodesIter ) // leaves
leaves.push_back((float)*internalNodesIter);
}
}

if( maxNodesPerTree == cascadedetect_INTEGER_1509_1_1 )
{
int nodeOfs = cascadedetect_INTEGER_1511_1_0, leafOfs = cascadedetect_INTEGER_1511_2_0;
size_t nstages = stages.size();
for( size_t stageIdx = cascadedetect_INTEGER_1513_1_0; stageIdx < nstages; stageIdx+=cascadedetect_INTEGER_1513_2_1 )
{
const Stage& stage = stages[stageIdx];

int ntrees = stage.ntrees;
for( int i = cascadedetect_INTEGER_1518_1_0; i < ntrees; i+=cascadedetect_INTEGER_1518_2_1, nodeOfs+=cascadedetect_INTEGER_1518_3_1, leafOfs+= cascadedetect_INTEGER_1518_4_2 )
{
const DTreeNode& node = nodes[nodeOfs];
stumps.push_back(Stump(node.featureIdx, node.threshold,
leaves[leafOfs], leaves[leafOfs+cascadedetect_INTEGER_1522_1_1]));
}
}
}

return true;
}


bool CascadeClassifierImpl::read_(const FileNode& root)
{
#ifdef HAVE_OPENCL
tryOpenCL = true;
haarKernel = ocl::Kernel();
lbpKernel = ocl::Kernel();
#endif
ustages.release();
unodes.release();
uleaves.release();
if( !data.read(root) )
return false;

// load features
featureEvaluator = FeatureEvaluator::create(data.featureType);
FileNode fn = root[CC_FEATURES];
if( fn.empty() )
return false;

return featureEvaluator->read(fn, data.origWinSize);
}

template<> void DefaultDeleter<CvHaarClassifierCascade>::operator ()(CvHaarClassifierCascade* obj) const
{ cvReleaseHaarClassifierCascade(&obj); }


BaseCascadeClassifier::~BaseCascadeClassifier()
{
}

CascadeClassifier::CascadeClassifier() {}
CascadeClassifier::CascadeClassifier(const String& filename)
{
load(filename);
}

CascadeClassifier::~CascadeClassifier()
{
}

bool CascadeClassifier::empty() const
{
return cc.empty() || cc->empty();
}

bool CascadeClassifier::load( const String& filename )
{
cc = makePtr<CascadeClassifierImpl>();
if(!cc->load(filename))
cc.release();
return !empty();
}

bool CascadeClassifier::read(const FileNode &root)
{
Ptr<CascadeClassifierImpl> ccimpl = makePtr<CascadeClassifierImpl>();
bool ok = ccimpl->read_(root);
if( ok )
cc = ccimpl.staticCast<BaseCascadeClassifier>();
else
cc.release();
return ok;
}

void clipObjects(Size sz, std::vector<Rect>& objects,
std::vector<int>* a, std::vector<double>* b)
{
size_t i, j = cascadedetect_INTEGER_1598_1_0, n = objects.size();
Rect win0 = Rect(cascadedetect_INTEGER_1599_1_0, cascadedetect_INTEGER_1599_2_0, sz.width, sz.height);
if(a)
{
CV_Assert(a->size() == n);
}
if(b)
{
CV_Assert(b->size() == n);
}

for( i = cascadedetect_INTEGER_1609_1_0; i < n; i+=cascadedetect_INTEGER_1609_2_1 )
{
Rect r = win0 & objects[i];
if( r.area() > cascadedetect_INTEGER_1612_1_0 )
{
objects[j] = r;
if( i > j )
{
if(a) a->at(j) = a->at(i);
if(b) b->at(j) = b->at(i);
}
j+=cascadedetect_INTEGER_1620_1_1;
}
}

if( j < n )
{
objects.resize(j);
if(a) a->resize(j);
if(b) b->resize(j);
}
}

void CascadeClassifier::detectMultiScale( InputArray image,
CV_OUT std::vector<Rect>& objects,
double scaleFactor,
int minNeighbors, int flags,
Size minSize,
Size maxSize )
{
CV_Assert(!empty());
cc->detectMultiScale(image, objects, scaleFactor, minNeighbors, flags, minSize, maxSize);
clipObjects(image.size(), objects, cascadedetect_INTEGER_1641_1_0, cascadedetect_INTEGER_1641_2_0);
}

void CascadeClassifier::detectMultiScale( InputArray image,
CV_OUT std::vector<Rect>& objects,
CV_OUT std::vector<int>& numDetections,
double scaleFactor,
int minNeighbors, int flags,
Size minSize, Size maxSize )
{
CV_Assert(!empty());
cc->detectMultiScale(image, objects, numDetections,
scaleFactor, minNeighbors, flags, minSize, maxSize);
clipObjects(image.size(), objects, &numDetections, cascadedetect_INTEGER_1654_1_0);
}

void CascadeClassifier::detectMultiScale( InputArray image,
CV_OUT std::vector<Rect>& objects,
CV_OUT std::vector<int>& rejectLevels,
CV_OUT std::vector<double>& levelWeights,
double scaleFactor,
int minNeighbors, int flags,
Size minSize, Size maxSize,
bool outputRejectLevels )
{
CV_Assert(!empty());
cc->detectMultiScale(image, objects, rejectLevels, levelWeights,
scaleFactor, minNeighbors, flags,
minSize, maxSize, outputRejectLevels);
clipObjects(image.size(), objects, &rejectLevels, &levelWeights);
}

bool CascadeClassifier::isOldFormatCascade() const
{
CV_Assert(!empty());
return cc->isOldFormatCascade();
}

Size CascadeClassifier::getOriginalWindowSize() const
{
CV_Assert(!empty());
return cc->getOriginalWindowSize();
}

int CascadeClassifier::getFeatureType() const
{
CV_Assert(!empty());
return cc->getFeatureType();
}

void* CascadeClassifier::getOldCascade()
{
CV_Assert(!empty());
return cc->getOldCascade();
}

void CascadeClassifier::setMaskGenerator(const Ptr<BaseCascadeClassifier::MaskGenerator>& maskGenerator)
{
CV_Assert(!empty());
cc->setMaskGenerator(maskGenerator);
}

Ptr<BaseCascadeClassifier::MaskGenerator> CascadeClassifier::getMaskGenerator()
{
CV_Assert(!empty());
return cc->getMaskGenerator();
}

} // namespace cv
