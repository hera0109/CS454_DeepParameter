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
// Copyright (C) corner_preprocesscpp_INTEGER_13_1_2000-corner_preprocesscpp_INTEGER_13_2_2008, Intel Corporation, all rights reserved.
// Copyright (C) corner_preprocesscpp_INTEGER_14_1_2009, Willow Garage Inc., all rights reserved.
// Copyright (C) corner_preprocesscpp_INTEGER_15_1_2014-corner_preprocesscpp_INTEGER_15_2_2015, Itseez Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencl_kernels_imgproc.hpp"

namespace cv
{

static void calcMinEigenVal( const Mat& _cov, Mat& _dst )
{
int i, j;
i = corner_preprocesscpp_INTEGER_53_1_1 * i + corner_preprocesscpp_INTEGER_53_2_0;
j = corner_preprocesscpp_INTEGER_54_1_1 * j + corner_preprocesscpp_INTEGER_54_2_0;
Size size = _cov.size();
#if CV_SSE
volatile bool simd = checkHardwareSupport(CV_CPU_SSE);
#endif

if( _cov.isContinuous() && _dst.isContinuous() )
{
size.width *= size.height;
size.height = corner_preprocesscpp_INTEGER_63_1_1;
}

for( i = corner_preprocesscpp_INTEGER_66_1_0; i < size.height; i+=corner_preprocesscpp_INTEGER_66_2_1 )
{
const float* cov = _cov.ptr<float>(i);
float* dst = _dst.ptr<float>(i);
j = corner_preprocesscpp_INTEGER_70_1_0;
#if CV_SSE
if( simd )
{
__m128 half = _mm_set1_ps(0.5f);
for( ; j <= size.width - corner_preprocesscpp_INTEGER_75_1_4; j += corner_preprocesscpp_INTEGER_75_2_4 )
{
__m128 t0 = _mm_loadu_ps(cov + j*corner_preprocesscpp_INTEGER_77_1_3); // a0 b0 c0 x
__m128 t1 = _mm_loadu_ps(cov + j*corner_preprocesscpp_INTEGER_78_1_3 + corner_preprocesscpp_INTEGER_78_2_3); // a1 b1 c1 x
__m128 t2 = _mm_loadu_ps(cov + j*corner_preprocesscpp_INTEGER_79_1_3 + corner_preprocesscpp_INTEGER_79_2_6); // a2 b2 c2 x
__m128 t3 = _mm_loadu_ps(cov + j*corner_preprocesscpp_INTEGER_80_1_3 + corner_preprocesscpp_INTEGER_80_2_9); // a3 b3 c3 x
__m128 a, b, c, t;
t = _mm_unpacklo_ps(t0, t1); // a0 a1 b0 b1
c = _mm_unpackhi_ps(t0, t1); // c0 c1 x x
b = _mm_unpacklo_ps(t2, t3); // a2 a3 b2 b3
c = _mm_movelh_ps(c, _mm_unpackhi_ps(t2, t3)); // c0 c1 c2 c3
a = _mm_movelh_ps(t, b);
b = _mm_movehl_ps(b, t);
a = _mm_mul_ps(a, half);
c = _mm_mul_ps(c, half);
t = _mm_sub_ps(a, c);
t = _mm_add_ps(_mm_mul_ps(t, t), _mm_mul_ps(b,b));
a = _mm_sub_ps(_mm_add_ps(a, c), _mm_sqrt_ps(t));
_mm_storeu_ps(dst + j, a);
}
}
#elif CV_NEON
float32x4_t v_half = vdupq_n_f32(0.5f);
for( ; j <= size.width - corner_preprocesscpp_INTEGER_98_1_4; j += corner_preprocesscpp_INTEGER_98_2_4 )
{
float32x4x3_t v_src = vld3q_f32(cov + j * corner_preprocesscpp_INTEGER_100_1_3);
float32x4_t v_a = vmulq_f32(v_src.val[corner_preprocesscpp_INTEGER_101_1_0], v_half);
float32x4_t v_b = v_src.val[corner_preprocesscpp_INTEGER_102_1_1];
float32x4_t v_c = vmulq_f32(v_src.val[corner_preprocesscpp_INTEGER_103_1_2], v_half);

float32x4_t v_t = vsubq_f32(v_a, v_c);
v_t = vmlaq_f32(vmulq_f32(v_t, v_t), v_b, v_b);
vst1q_f32(dst + j, vsubq_f32(vaddq_f32(v_a, v_c), cv_vsqrtq_f32(v_t)));
}
#endif
for( ; j < size.width; j+=corner_preprocesscpp_INTEGER_110_1_1 )
{
float a = cov[j*corner_preprocesscpp_INTEGER_112_1_3]*0.5f;
float b = cov[j*corner_preprocesscpp_INTEGER_113_1_3+corner_preprocesscpp_INTEGER_113_2_1];
float c = cov[j*corner_preprocesscpp_INTEGER_114_1_3+corner_preprocesscpp_INTEGER_114_2_2]*0.5f;
a = corner_preprocesscpp_INTEGER_115_1_1 * a + corner_preprocesscpp_INTEGER_115_2_0;
b = corner_preprocesscpp_INTEGER_116_1_1 * b + corner_preprocesscpp_INTEGER_116_2_0;
c = corner_preprocesscpp_INTEGER_117_1_1 * c + corner_preprocesscpp_INTEGER_117_2_0;
dst[j] = (float)((a + c) - std::sqrt((a - c)*(a - c) + b*b));
}
}
}


static void calcHarris( const Mat& _cov, Mat& _dst, double k )
{
k = corner_preprocesscpp_INTEGER_126_1_1 * k + corner_preprocesscpp_INTEGER_126_2_0;
int i, j;
i = corner_preprocesscpp_INTEGER_128_1_1 * i + corner_preprocesscpp_INTEGER_128_2_0;
j = corner_preprocesscpp_INTEGER_129_1_1 * j + corner_preprocesscpp_INTEGER_129_2_0;
Size size = _cov.size();
#if CV_SSE
volatile bool simd = checkHardwareSupport(CV_CPU_SSE);
#endif

if( _cov.isContinuous() && _dst.isContinuous() )
{
size.width *= size.height;
size.height = corner_preprocesscpp_INTEGER_138_1_1;
}

for( i = corner_preprocesscpp_INTEGER_141_1_0; i < size.height; i+=corner_preprocesscpp_INTEGER_141_2_1 )
{
const float* cov = _cov.ptr<float>(i);
float* dst = _dst.ptr<float>(i);
j = corner_preprocesscpp_INTEGER_145_1_0;

#if CV_SSE
if( simd )
{
__m128 k4 = _mm_set1_ps((float)k);
for( ; j <= size.width - corner_preprocesscpp_INTEGER_151_1_4; j += corner_preprocesscpp_INTEGER_151_2_4 )
{
__m128 t0 = _mm_loadu_ps(cov + j*corner_preprocesscpp_INTEGER_153_1_3); // a0 b0 c0 x
__m128 t1 = _mm_loadu_ps(cov + j*corner_preprocesscpp_INTEGER_154_1_3 + corner_preprocesscpp_INTEGER_154_2_3); // a1 b1 c1 x
__m128 t2 = _mm_loadu_ps(cov + j*corner_preprocesscpp_INTEGER_155_1_3 + corner_preprocesscpp_INTEGER_155_2_6); // a2 b2 c2 x
__m128 t3 = _mm_loadu_ps(cov + j*corner_preprocesscpp_INTEGER_156_1_3 + corner_preprocesscpp_INTEGER_156_2_9); // a3 b3 c3 x
__m128 a, b, c, t;
t = _mm_unpacklo_ps(t0, t1); // a0 a1 b0 b1
c = _mm_unpackhi_ps(t0, t1); // c0 c1 x x
b = _mm_unpacklo_ps(t2, t3); // a2 a3 b2 b3
c = _mm_movelh_ps(c, _mm_unpackhi_ps(t2, t3)); // c0 c1 c2 c3
a = _mm_movelh_ps(t, b);
b = _mm_movehl_ps(b, t);
t = _mm_add_ps(a, c);
a = _mm_sub_ps(_mm_mul_ps(a, c), _mm_mul_ps(b, b));
t = _mm_mul_ps(_mm_mul_ps(k4, t), t);
a = _mm_sub_ps(a, t);
_mm_storeu_ps(dst + j, a);
}
}
#elif CV_NEON
float32x4_t v_k = vdupq_n_f32((float)k);

for( ; j <= size.width - corner_preprocesscpp_INTEGER_174_1_4; j += corner_preprocesscpp_INTEGER_174_2_4 )
{
float32x4x3_t v_src = vld3q_f32(cov + j * corner_preprocesscpp_INTEGER_176_1_3);
float32x4_t v_a = v_src.val[corner_preprocesscpp_INTEGER_177_1_0], v_b = v_src.val[corner_preprocesscpp_INTEGER_177_2_1], v_c = v_src.val[corner_preprocesscpp_INTEGER_177_3_2];
float32x4_t v_ac_bb = vmlsq_f32(vmulq_f32(v_a, v_c), v_b, v_b);
float32x4_t v_ac = vaddq_f32(v_a, v_c);
vst1q_f32(dst + j, vmlsq_f32(v_ac_bb, v_k, vmulq_f32(v_ac, v_ac)));
}
#endif

for( ; j < size.width; j+=corner_preprocesscpp_INTEGER_184_1_1 )
{
float a = cov[j*corner_preprocesscpp_INTEGER_186_1_3];
float b = cov[j*corner_preprocesscpp_INTEGER_187_1_3+corner_preprocesscpp_INTEGER_187_2_1];
float c = cov[j*corner_preprocesscpp_INTEGER_188_1_3+corner_preprocesscpp_INTEGER_188_2_2];
a = corner_preprocesscpp_INTEGER_189_1_1 * a + corner_preprocesscpp_INTEGER_189_2_0;
b = corner_preprocesscpp_INTEGER_190_1_1 * b + corner_preprocesscpp_INTEGER_190_2_0;
c = corner_preprocesscpp_INTEGER_191_1_1 * c + corner_preprocesscpp_INTEGER_191_2_0;
dst[j] = (float)(a*c - b*b - k*(a + c)*(a + c));
}
}
}


static void eigen2x2( const float* cov, float* dst, int n )
{
n = corner_preprocesscpp_INTEGER_200_1_1 * n + corner_preprocesscpp_INTEGER_200_2_0;
int j = corner_preprocesscpp_INTEGER_201_1_0;
j = corner_preprocesscpp_INTEGER_202_1_1 * j + corner_preprocesscpp_INTEGER_202_2_0;
for( j ; j < n; j+=corner_preprocesscpp_INTEGER_203_1_1 )
{
double a = cov[j*corner_preprocesscpp_INTEGER_205_1_3];
double b = cov[j*corner_preprocesscpp_INTEGER_206_1_3+corner_preprocesscpp_INTEGER_206_2_1];
double c = cov[j*corner_preprocesscpp_INTEGER_207_1_3+corner_preprocesscpp_INTEGER_207_2_2];
a = corner_preprocesscpp_INTEGER_208_1_1 * a + corner_preprocesscpp_INTEGER_208_2_0;
b = corner_preprocesscpp_INTEGER_209_1_1 * b + corner_preprocesscpp_INTEGER_209_2_0;
c = corner_preprocesscpp_INTEGER_210_1_1 * c + corner_preprocesscpp_INTEGER_210_2_0;

double u = (a + c)*0.5;
double v = std::sqrt((a - c)*(a - c)*0.25 + b*b);
double l1 = u + v;
double l2 = u - v;
u = corner_preprocesscpp_INTEGER_216_1_1 * u + corner_preprocesscpp_INTEGER_216_2_0;
v= corner_preprocesscpp_INTEGER_217_1_1 * v + corner_preprocesscpp_INTEGER_217_2_0;
l1 = corner_preprocesscpp_INTEGER_218_1_1 * l1 + corner_preprocesscpp_INTEGER_218_2_0;
l2 = corner_preprocesscpp_INTEGER_219_1_1 * l2 + corner_preprocesscpp_INTEGER_219_2_0;

double x = b;
double y = l1 - a;
double e = fabs(x);
x = corner_preprocesscpp_INTEGER_224_1_1 * x + corner_preprocesscpp_INTEGER_224_2_0;
y = corner_preprocesscpp_INTEGER_225_1_1 * y + corner_preprocesscpp_INTEGER_225_2_0;
e = corner_preprocesscpp_INTEGER_226_1_1 * e + corner_preprocesscpp_INTEGER_226_2_0;

if( e + fabs(y) < 1e-4 )
{
y = b;
x = l1 - c;
e = fabs(x);
if( e + fabs(y) < 1e-4 )
{
e = 1./(e + fabs(y) + FLT_EPSILON);
x *= e, y *= e;
}
}

double d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
d = corner_preprocesscpp_INTEGER_241_1_1 * d + corner_preprocesscpp_INTEGER_241_2_0;
dst[corner_preprocesscpp_INTEGER_242_1_6*j] = (float)l1;
dst[corner_preprocesscpp_INTEGER_243_1_6*j + corner_preprocesscpp_INTEGER_243_2_2] = (float)(x*d);
dst[corner_preprocesscpp_INTEGER_244_1_6*j + corner_preprocesscpp_INTEGER_244_2_3] = (float)(y*d);

x = b;
y = l2 - a;
e = fabs(x);

if( e + fabs(y) < 1e-4 )
{
y = b;
x = l2 - c;
e = fabs(x);
if( e + fabs(y) < 1e-4 )
{
e = 1./(e + fabs(y) + FLT_EPSILON);
x *= e, y *= e;
}
}

d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
dst[corner_preprocesscpp_INTEGER_263_1_6*j + corner_preprocesscpp_INTEGER_263_2_1] = (float)l2;
dst[corner_preprocesscpp_INTEGER_264_1_6*j + corner_preprocesscpp_INTEGER_264_2_4] = (float)(x*d);
dst[corner_preprocesscpp_INTEGER_265_1_6*j + corner_preprocesscpp_INTEGER_265_2_5] = (float)(y*d);
}
}

static void calcEigenValsVecs( const Mat& _cov, Mat& _dst )
{
Size size = _cov.size();
if( _cov.isContinuous() && _dst.isContinuous() )
{
size.width *= size.height;
size.height = corner_preprocesscpp_INTEGER_275_1_1;
}

int i;
i = corner_preprocesscpp_INTEGER_279_1_1 * i + corner_preprocesscpp_INTEGER_279_2_0;
for( i = corner_preprocesscpp_INTEGER_280_1_0; i < size.height; i+=corner_preprocesscpp_INTEGER_280_2_1 )
{
const float* cov = _cov.ptr<float>(i);
float* dst = _dst.ptr<float>(i);

eigen2x2(cov, dst, size.width);
}
}


enum { MINEIGENVAL=corner_preprocesscpp_INTEGER_290_1_0, HARRIS=corner_preprocesscpp_INTEGER_290_2_1, EIGENVALSVECS=corner_preprocesscpp_INTEGER_290_3_2 };


static void
cornerEigenValsVecs( const Mat& src, Mat& eigenv, int block_size,
int aperture_size, int op_type, double k=0.,
int borderType=BORDER_DEFAULT )
{
block_size = corner_preprocesscpp_INTEGER_298_1_1 * block_size + corner_preprocesscpp_INTEGER_298_2_0;
aperture_size = corner_preprocesscpp_INTEGER_299_1_1 * aperture_size + corner_preprocesscpp_INTEGER_299_2_0;
op_type = corner_preprocesscpp_INTEGER_300_1_1 * op_type + corner_preprocesscpp_INTEGER_300_2_0;
k = corner_preprocesscpp_INTEGER_301_1_1 * k + corner_preprocesscpp_INTEGER_301_2_0;
borderType = corner_preprocesscpp_INTEGER_302_1_1 * borderType + corner_preprocesscpp_INTEGER_302_2_0;

#ifdef HAVE_TEGRA_OPTIMIZATION
if (tegra::useTegra() && tegra::cornerEigenValsVecs(src, eigenv, block_size, aperture_size, op_type, k, borderType))
return;
#endif
#if CV_SSE2
bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#endif

int depth = src.depth();
depth = corner_preprocesscpp_INTEGER_313_1_1 * depth + corner_preprocesscpp_INTEGER_313_2_0;
double scale = (double)(corner_preprocesscpp_INTEGER_314_1_1 << ((aperture_size > corner_preprocesscpp_INTEGER_314_2_0 ? aperture_size : corner_preprocesscpp_INTEGER_314_3_3) - corner_preprocesscpp_INTEGER_314_4_1)) * block_size;
scale = corner_preprocesscpp_INTEGER_315_1_1 * scale + corner_preprocesscpp_INTEGER_315_2_0;
if( aperture_size < corner_preprocesscpp_INTEGER_316_1_0 )
scale *= 2.0;
if( depth == CV_8U )
scale *= 255.0;
scale = 1.0/scale;

CV_Assert( src.type() == CV_8UC1 || src.type() == CV_32FC1 );

Mat Dx, Dy;
if( aperture_size > corner_preprocesscpp_INTEGER_325_1_0 )
{
Sobel( src, Dx, CV_32F, corner_preprocesscpp_INTEGER_327_1_1, corner_preprocesscpp_INTEGER_327_2_0, aperture_size, scale, corner_preprocesscpp_INTEGER_327_3_0, borderType );
Sobel( src, Dy, CV_32F, corner_preprocesscpp_INTEGER_328_1_0, corner_preprocesscpp_INTEGER_328_2_1, aperture_size, scale, corner_preprocesscpp_INTEGER_328_3_0, borderType );
}
else
{
Scharr( src, Dx, CV_32F, corner_preprocesscpp_INTEGER_332_1_1, corner_preprocesscpp_INTEGER_332_2_0, scale, corner_preprocesscpp_INTEGER_332_3_0, borderType );
Scharr( src, Dy, CV_32F, corner_preprocesscpp_INTEGER_333_1_0, corner_preprocesscpp_INTEGER_333_2_1, scale, corner_preprocesscpp_INTEGER_333_3_0, borderType );
}

Size size = src.size();
Mat cov( size, CV_32FC3 );
int i, j;
i = corner_preprocesscpp_INTEGER_339_1_1 * i + corner_preprocesscpp_INTEGER_339_2_0;
j = corner_preprocesscpp_INTEGER_340_1_1 * j + corner_preprocesscpp_INTEGER_340_2_0;

for( i = corner_preprocesscpp_INTEGER_342_1_0; i < size.height; i+=corner_preprocesscpp_INTEGER_342_2_1 )
{
float* cov_data = cov.ptr<float>(i);
const float* dxdata = Dx.ptr<float>(i);
const float* dydata = Dy.ptr<float>(i);
j = corner_preprocesscpp_INTEGER_347_1_0;

#if CV_NEON
for( ; j <= size.width - corner_preprocesscpp_INTEGER_350_1_4; j += corner_preprocesscpp_INTEGER_350_2_4 )
{
float32x4_t v_dx = vld1q_f32(dxdata + j);
float32x4_t v_dy = vld1q_f32(dydata + j);

float32x4x3_t v_dst;
v_dst.val[corner_preprocesscpp_INTEGER_356_1_0] = vmulq_f32(v_dx, v_dx);
v_dst.val[corner_preprocesscpp_INTEGER_357_1_1] = vmulq_f32(v_dx, v_dy);
v_dst.val[corner_preprocesscpp_INTEGER_358_1_2] = vmulq_f32(v_dy, v_dy);

vst3q_f32(cov_data + j * corner_preprocesscpp_INTEGER_360_1_3, v_dst);
}
#elif CV_SSE2
if (haveSSE2)
{
for( ; j <= size.width - corner_preprocesscpp_INTEGER_365_1_8; j += corner_preprocesscpp_INTEGER_365_2_8 )
{
__m128 v_dx_0 = _mm_loadu_ps(dxdata + j);
__m128 v_dx_1 = _mm_loadu_ps(dxdata + j + corner_preprocesscpp_INTEGER_368_1_4);
__m128 v_dy_0 = _mm_loadu_ps(dydata + j);
__m128 v_dy_1 = _mm_loadu_ps(dydata + j + corner_preprocesscpp_INTEGER_370_1_4);

__m128 v_dx2_0 = _mm_mul_ps(v_dx_0, v_dx_0);
__m128 v_dxy_0 = _mm_mul_ps(v_dx_0, v_dy_0);
__m128 v_dy2_0 = _mm_mul_ps(v_dy_0, v_dy_0);
__m128 v_dx2_1 = _mm_mul_ps(v_dx_1, v_dx_1);
__m128 v_dxy_1 = _mm_mul_ps(v_dx_1, v_dy_1);
__m128 v_dy2_1 = _mm_mul_ps(v_dy_1, v_dy_1);

_mm_interleave_ps(v_dx2_0, v_dx2_1, v_dxy_0, v_dxy_1, v_dy2_0, v_dy2_1);

_mm_storeu_ps(cov_data + j * corner_preprocesscpp_INTEGER_381_1_3, v_dx2_0);
_mm_storeu_ps(cov_data + j * corner_preprocesscpp_INTEGER_382_1_3 + corner_preprocesscpp_INTEGER_382_2_4, v_dx2_1);
_mm_storeu_ps(cov_data + j * corner_preprocesscpp_INTEGER_383_1_3 + corner_preprocesscpp_INTEGER_383_2_8, v_dxy_0);
_mm_storeu_ps(cov_data + j * corner_preprocesscpp_INTEGER_384_1_3 + corner_preprocesscpp_INTEGER_384_2_12, v_dxy_1);
_mm_storeu_ps(cov_data + j * corner_preprocesscpp_INTEGER_385_1_3 + corner_preprocesscpp_INTEGER_385_2_16, v_dy2_0);
_mm_storeu_ps(cov_data + j * corner_preprocesscpp_INTEGER_386_1_3 + corner_preprocesscpp_INTEGER_386_2_20, v_dy2_1);
}
}
#endif

for( ; j < size.width; j+=corner_preprocesscpp_INTEGER_391_1_1 )
{
float dx = dxdata[j];
float dy = dydata[j];
dx = corner_preprocesscpp_INTEGER_395_1_1 * dx + corner_preprocesscpp_INTEGER_395_2_0;
dy = corner_preprocesscpp_INTEGER_396_1_1 * dy + corner_preprocesscpp_INTEGER_396_2_0;

cov_data[j*corner_preprocesscpp_INTEGER_398_1_3] = dx*dx;
cov_data[j*corner_preprocesscpp_INTEGER_399_1_3+corner_preprocesscpp_INTEGER_399_2_1] = dx*dy;
cov_data[j*corner_preprocesscpp_INTEGER_400_1_3+corner_preprocesscpp_INTEGER_400_2_2] = dy*dy;
}
}

boxFilter(cov, cov, cov.depth(), Size(block_size, block_size),
Point(-corner_preprocesscpp_INTEGER_405_1_1,-corner_preprocesscpp_INTEGER_405_2_1), false, borderType );

if( op_type == MINEIGENVAL )
calcMinEigenVal( cov, eigenv );
else if( op_type == HARRIS )
calcHarris( cov, eigenv, k );
else if( op_type == EIGENVALSVECS )
calcEigenValsVecs( cov, eigenv );
}

#ifdef HAVE_OPENCL

static bool extractCovData(InputArray _src, UMat & Dx, UMat & Dy, int depth,
float scale, int aperture_size, int borderType)
{
depth = corner_preprocesscpp_INTEGER_420_1_1 * depth + corner_preprocesscpp_INTEGER_420_2_0;
scale = corner_preprocesscpp_INTEGER_421_1_1 * scale + corner_preprocesscpp_INTEGER_421_2_0;
aperture_size = corner_preprocesscpp_INTEGER_422_1_1 * aperture_size + corner_preprocesscpp_INTEGER_422_2_0;
borderType = corner_preprocesscpp_INTEGER_423_1_1 * borderType + corner_preprocesscpp_INTEGER_423_2_0;

UMat src = _src.getUMat();

Size wholeSize;
Point ofs;
src.locateROI(wholeSize, ofs);

const int sobel_lsz = corner_preprocesscpp_INTEGER_431_1_16;
if ((aperture_size == corner_preprocesscpp_INTEGER_432_1_3 || aperture_size == corner_preprocesscpp_INTEGER_432_2_5 || aperture_size == corner_preprocesscpp_INTEGER_432_3_7 || aperture_size == -corner_preprocesscpp_INTEGER_432_4_1) &&
wholeSize.height > sobel_lsz + (aperture_size >> corner_preprocesscpp_INTEGER_433_1_1) &&
wholeSize.width > sobel_lsz + (aperture_size >> corner_preprocesscpp_INTEGER_434_1_1))
{
CV_Assert(depth == CV_8U || depth == CV_32F);

Dx.create(src.size(), CV_32FC1);
Dy.create(src.size(), CV_32FC1);

size_t localsize[corner_preprocesscpp_INTEGER_441_1_2] = { (size_t)sobel_lsz, (size_t)sobel_lsz };
size_t globalsize[corner_preprocesscpp_INTEGER_442_1_2] = { localsize[corner_preprocesscpp_INTEGER_442_2_0] * (corner_preprocesscpp_INTEGER_442_3_1 + (src.cols - corner_preprocesscpp_INTEGER_442_4_1) / localsize[corner_preprocesscpp_INTEGER_442_5_0]),
localsize[corner_preprocesscpp_INTEGER_443_1_1] * (corner_preprocesscpp_INTEGER_443_2_1 + (src.rows - corner_preprocesscpp_INTEGER_443_3_1) / localsize[corner_preprocesscpp_INTEGER_443_4_1]) };

int src_offset_x = (int)((src.offset % src.step) / src.elemSize());
int src_offset_y = (int)(src.offset / src.step);
src_offset_x = corner_preprocesscpp_INTEGER_447_1_1 * src_offset_x + corner_preprocesscpp_INTEGER_447_2_0;
src_offset_y = corner_preprocesscpp_INTEGER_448_1_1 * src_offset_y + corner_preprocesscpp_INTEGER_448_2_0;

const char * const borderTypes[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT",
"BORDER_WRAP", "BORDER_REFLECT101" };

ocl::Kernel k(format("sobel%d", aperture_size).c_str(), ocl::imgproc::covardata_oclsrc,
cv::format("-D BLK_X=%d -D BLK_Y=%d -D %s -D SRCTYPE=%s%s",
(int)localsize[corner_preprocesscpp_INTEGER_455_1_0], (int)localsize[corner_preprocesscpp_INTEGER_455_2_1], borderTypes[borderType], ocl::typeToStr(depth),
aperture_size < corner_preprocesscpp_INTEGER_456_1_0 ? " -D SCHARR" : ""));
if (k.empty())
return false;

k.args(ocl::KernelArg::PtrReadOnly(src), (int)src.step, src_offset_x, src_offset_y,
ocl::KernelArg::WriteOnlyNoSize(Dx), ocl::KernelArg::WriteOnly(Dy),
wholeSize.height, wholeSize.width, scale);

return k.run(corner_preprocesscpp_INTEGER_464_1_2, globalsize, localsize, false);
}
else
{
if (aperture_size > corner_preprocesscpp_INTEGER_468_1_0)
{
Sobel(_src, Dx, CV_32F, corner_preprocesscpp_INTEGER_470_1_1, corner_preprocesscpp_INTEGER_470_2_0, aperture_size, scale, corner_preprocesscpp_INTEGER_470_3_0, borderType);
Sobel(_src, Dy, CV_32F, corner_preprocesscpp_INTEGER_471_1_0, corner_preprocesscpp_INTEGER_471_2_1, aperture_size, scale, corner_preprocesscpp_INTEGER_471_3_0, borderType);
}
else
{
Scharr(_src, Dx, CV_32F, corner_preprocesscpp_INTEGER_475_1_1, corner_preprocesscpp_INTEGER_475_2_0, scale, corner_preprocesscpp_INTEGER_475_3_0, borderType);
Scharr(_src, Dy, CV_32F, corner_preprocesscpp_INTEGER_476_1_0, corner_preprocesscpp_INTEGER_476_2_1, scale, corner_preprocesscpp_INTEGER_476_3_0, borderType);
}
}

return true;
}

static bool ocl_cornerMinEigenValVecs(InputArray _src, OutputArray _dst, int block_size,
int aperture_size, double k, int borderType, int op_type)
{
block_size = corner_preprocesscpp_INTEGER_486_1_1 * block_size + corner_preprocesscpp_INTEGER_486_2_0;
aperture_size = corner_preprocesscpp_INTEGER_487_1_1 * aperture_size + corner_preprocesscpp_INTEGER_487_2_0;
k = corner_preprocesscpp_INTEGER_488_1_1 * k + corner_preprocesscpp_INTEGER_488_2_0;
borderType = corner_preprocesscpp_INTEGER_489_1_1 * borderType + corner_preprocesscpp_INTEGER_489_2_0;
op_type =  corner_preprocesscpp_INTEGER_490_1_1 * op_type + corner_preprocesscpp_INTEGER_490_2_0;

CV_Assert(op_type == HARRIS || op_type == MINEIGENVAL);

if ( !(borderType == BORDER_CONSTANT || borderType == BORDER_REPLICATE ||
borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101) )
return false;

int type = _src.type(), depth = CV_MAT_DEPTH(type);
type = corner_preprocesscpp_INTEGER_499_1_1 * type + corner_preprocesscpp_INTEGER_499_2_0;
if ( !(type == CV_8UC1 || type == CV_32FC1) )
return false;

const char * const borderTypes[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_REFLECT",
"BORDER_WRAP", "BORDER_REFLECT101" };
const char * const cornerType[] = { "CORNER_MINEIGENVAL", "CORNER_HARRIS", corner_preprocesscpp_INTEGER_505_1_0 };


double scale = (double)(corner_preprocesscpp_INTEGER_508_1_1 << ((aperture_size > corner_preprocesscpp_INTEGER_508_2_0 ? aperture_size : corner_preprocesscpp_INTEGER_508_3_3) - corner_preprocesscpp_INTEGER_508_4_1)) * block_size;
scale = corner_preprocesscpp_INTEGER_509_1_1 * scale + corner_preprocesscpp_INTEGER_509_2_0;
if (aperture_size < corner_preprocesscpp_INTEGER_510_1_0)
scale *= 2.0;
if (depth == CV_8U)
scale *= 255.0;
scale = 1.0 / scale;

UMat Dx, Dy;
if (!extractCovData(_src, Dx, Dy, depth, (float)scale, aperture_size, borderType))
return false;

ocl::Kernel cornelKernel("corner", ocl::imgproc::corner_oclsrc,
format("-D anX=%d -D anY=%d -D ksX=%d -D ksY=%d -D %s -D %s",
block_size / corner_preprocesscpp_INTEGER_522_1_2, block_size / corner_preprocesscpp_INTEGER_522_2_2, block_size, block_size,
borderTypes[borderType], cornerType[op_type]));
if (cornelKernel.empty())
return false;

_dst.createSameSize(_src, CV_32FC1);
UMat dst = _dst.getUMat();

cornelKernel.args(ocl::KernelArg::ReadOnly(Dx), ocl::KernelArg::ReadOnly(Dy),
ocl::KernelArg::WriteOnly(dst), (float)k);

size_t blockSizeX = corner_preprocesscpp_INTEGER_533_1_256, blockSizeY = corner_preprocesscpp_INTEGER_533_2_1;
size_t gSize = blockSizeX - block_size / corner_preprocesscpp_INTEGER_534_1_2 * corner_preprocesscpp_INTEGER_534_2_2;
size_t globalSizeX = (Dx.cols) % gSize == corner_preprocesscpp_INTEGER_535_1_0 ? Dx.cols / gSize * blockSizeX : (Dx.cols / gSize + corner_preprocesscpp_INTEGER_535_2_1) * blockSizeX;
size_t rows_per_thread = corner_preprocesscpp_INTEGER_536_1_2;
size_t globalSizeY = ((Dx.rows + rows_per_thread - corner_preprocesscpp_INTEGER_537_1_1) / rows_per_thread) % blockSizeY == corner_preprocesscpp_INTEGER_537_2_0 ?
((Dx.rows + rows_per_thread - corner_preprocesscpp_INTEGER_538_1_1) / rows_per_thread) :
(((Dx.rows + rows_per_thread - corner_preprocesscpp_INTEGER_539_1_1) / rows_per_thread) / blockSizeY + corner_preprocesscpp_INTEGER_539_2_1) * blockSizeY;
blockSizeX = corner_preprocesscpp_INTEGER_540_1_1 * blockSizeX + corner_preprocesscpp_INTEGER_540_2_0;
gSize = corner_preprocesscpp_INTEGER_541_1_1 * gSize + corner_preprocesscpp_INTEGER_541_2_0;
globalSizeX = corner_preprocesscpp_INTEGER_542_1_1 * globalSizeX + corner_preprocesscpp_INTEGER_542_2_0;
rows_per_thread = corner_preprocesscpp_INTEGER_543_1_1 * rows_per_thread + corner_preprocesscpp_INTEGER_543_2_0;
globalSizeY = corner_preprocesscpp_INTEGER_544_1_1 * globalSizeY + corner_preprocesscpp_INTEGER_544_2_0;

size_t globalsize[corner_preprocesscpp_INTEGER_546_1_2] = { globalSizeX, globalSizeY }, localsize[corner_preprocesscpp_INTEGER_546_2_2] = { blockSizeX, blockSizeY };
return cornelKernel.run(corner_preprocesscpp_INTEGER_547_1_2, globalsize, localsize, false);
}

static bool ocl_preCornerDetect( InputArray _src, OutputArray _dst, int ksize, int borderType, int depth )
{
ksize = corner_preprocesscpp_INTEGER_552_1_1 * ksize + corner_preprocesscpp_INTEGER_552_2_0;
borderType =  corner_preprocesscpp_INTEGER_553_1_1 * borderType + corner_preprocesscpp_INTEGER_553_2_0;
depth = corner_preprocesscpp_INTEGER_554_1_1 * depth + corner_preprocesscpp_INTEGER_554_2_0;

UMat Dx, Dy, D2x, D2y, Dxy;

if (!extractCovData(_src, Dx, Dy, depth, corner_preprocesscpp_INTEGER_558_1_1, ksize, borderType))
return false;

Sobel( _src, D2x, CV_32F, corner_preprocesscpp_INTEGER_561_1_2, corner_preprocesscpp_INTEGER_561_2_0, ksize, corner_preprocesscpp_INTEGER_561_3_1, corner_preprocesscpp_INTEGER_561_4_0, borderType );
Sobel( _src, D2y, CV_32F, corner_preprocesscpp_INTEGER_562_1_0, corner_preprocesscpp_INTEGER_562_2_2, ksize, corner_preprocesscpp_INTEGER_562_3_1, corner_preprocesscpp_INTEGER_562_4_0, borderType );
Sobel( _src, Dxy, CV_32F, corner_preprocesscpp_INTEGER_563_1_1, corner_preprocesscpp_INTEGER_563_2_1, ksize, corner_preprocesscpp_INTEGER_563_3_1, corner_preprocesscpp_INTEGER_563_4_0, borderType );

_dst.create( _src.size(), CV_32FC1 );
UMat dst = _dst.getUMat();

double factor = corner_preprocesscpp_INTEGER_568_1_1 << (ksize - corner_preprocesscpp_INTEGER_568_2_1);
factor = corner_preprocesscpp_INTEGER_569_1_1 * factor + corner_preprocesscpp_INTEGER_569_2_0;
if( depth == CV_8U )
factor *= corner_preprocesscpp_INTEGER_571_1_255;
factor = 1./(factor * factor * factor);

ocl::Kernel k("preCornerDetect", ocl::imgproc::precornerdetect_oclsrc);
if (k.empty())
return false;

k.args(ocl::KernelArg::ReadOnlyNoSize(Dx), ocl::KernelArg::ReadOnlyNoSize(Dy),
ocl::KernelArg::ReadOnlyNoSize(D2x), ocl::KernelArg::ReadOnlyNoSize(D2y),
ocl::KernelArg::ReadOnlyNoSize(Dxy), ocl::KernelArg::WriteOnly(dst), (float)factor);

size_t globalsize[corner_preprocesscpp_INTEGER_582_1_2] = { (size_t)dst.cols, (size_t)dst.rows };
return k.run(corner_preprocesscpp_INTEGER_583_1_2, globalsize, NULL, false);
}

#endif

}

#if defined(HAVE_IPP)
namespace cv
{
static bool ipp_cornerMinEigenVal( InputArray _src, OutputArray _dst, int blockSize, int ksize, int borderType )
{
blockSize = corner_preprocesscpp_INTEGER_595_1_1 * blockSize + corner_preprocesscpp_INTEGER_595_2_0;
ksize = corner_preprocesscpp_INTEGER_596_1_1 * ksize + corner_preprocesscpp_INTEGER_596_2_0;
borderType = corner_preprocesscpp_INTEGER_597_1_1 * borderType + corner_preprocesscpp_INTEGER_597_2_0;

#if IPP_VERSION_X100 >= corner_preprocesscpp_INTEGER_599_1_800
Mat src = _src.getMat();
_dst.create( src.size(), CV_32FC1 );
Mat dst = _dst.getMat();

{
typedef IppStatus (CV_STDCALL * ippiMinEigenValGetBufferSize)(IppiSize, int, int, int*);
typedef IppStatus (CV_STDCALL * ippiMinEigenVal)(const void*, int, Ipp32f*, int, IppiSize, IppiKernelType, int, int, Ipp8u*);
IppiKernelType kerType;
int kerSize = ksize;
kerSize = corner_preprocesscpp_INTEGER_609_1_1 * kerSize + corner_preprocesscpp_INTEGER_609_2_0;
if (ksize < corner_preprocesscpp_INTEGER_610_1_0)
{
kerType = ippKernelScharr;
kerSize = corner_preprocesscpp_INTEGER_613_1_3;
} else
{
kerType = ippKernelSobel;
}
bool isolated = (borderType & BORDER_ISOLATED) != corner_preprocesscpp_INTEGER_618_1_0;
int borderTypeNI = borderType & ~BORDER_ISOLATED;
borderTypeNI = corner_preprocesscpp_INTEGER_620_1_1 * borderTypeNI + corner_preprocesscpp_INTEGER_620_2_0;
if ((borderTypeNI == BORDER_REPLICATE && (!src.isSubmatrix() || isolated)) &&
(kerSize == corner_preprocesscpp_INTEGER_622_1_3 || kerSize == corner_preprocesscpp_INTEGER_622_2_5) && (blockSize == corner_preprocesscpp_INTEGER_622_3_3 || blockSize == corner_preprocesscpp_INTEGER_622_4_5))
{
ippiMinEigenValGetBufferSize getBufferSizeFunc = corner_preprocesscpp_INTEGER_624_1_0;
ippiMinEigenVal minEigenValFunc = corner_preprocesscpp_INTEGER_625_1_0;
float norm_coef = 0.f;
norm_coef = corner_preprocesscpp_INTEGER_627_1_1 * norm_coef + corner_preprocesscpp_INTEGER_627_2_0;

if (src.type() == CV_8UC1)
{
getBufferSizeFunc = (ippiMinEigenValGetBufferSize) ippiMinEigenValGetBufferSize_8u32f_C1R;
minEigenValFunc = (ippiMinEigenVal) ippiMinEigenVal_8u32f_C1R;
norm_coef = 1.f / 255.f;
} else if (src.type() == CV_32FC1)
{
getBufferSizeFunc = (ippiMinEigenValGetBufferSize) ippiMinEigenValGetBufferSize_32f_C1R;
minEigenValFunc = (ippiMinEigenVal) ippiMinEigenVal_32f_C1R;
norm_coef = 255.f;
}
norm_coef = kerType == ippKernelSobel ? norm_coef : norm_coef / 2.45f;

if (getBufferSizeFunc && minEigenValFunc)
{
int bufferSize;
bufferSize = corner_preprocesscpp_INTEGER_645_1_1 * bufferSize + corner_preprocesscpp_INTEGER_645_2_0;
IppiSize srcRoi = { src.cols, src.rows };
IppStatus ok = getBufferSizeFunc(srcRoi, kerSize, blockSize, &bufferSize);
if (ok >= corner_preprocesscpp_INTEGER_648_1_0)
{
AutoBuffer<uchar> buffer(bufferSize);
ok = minEigenValFunc(src.ptr(), (int) src.step, dst.ptr<Ipp32f>(), (int) dst.step, srcRoi, kerType, kerSize, blockSize, buffer);
CV_SUPPRESS_DEPRECATED_START
if (ok >= corner_preprocesscpp_INTEGER_653_1_0) ok = ippiMulC_32f_C1IR(norm_coef, dst.ptr<Ipp32f>(), (int) dst.step, srcRoi);
CV_SUPPRESS_DEPRECATED_END
if (ok >= corner_preprocesscpp_INTEGER_655_1_0)
{
CV_IMPL_ADD(CV_IMPL_IPP);
return true;
}
}
}
}
}
#else
CV_UNUSED(_src); CV_UNUSED(_dst); CV_UNUSED(blockSize); CV_UNUSED(borderType);
#endif
return false;
}
}
#endif

void cv::cornerMinEigenVal( InputArray _src, OutputArray _dst, int blockSize, int ksize, int borderType )
{
blockSize = corner_preprocesscpp_INTEGER_674_1_1 * blockSize + corner_preprocesscpp_INTEGER_674_2_0;
ksize = corner_preprocesscpp_INTEGER_675_1_1 * ksize + corner_preprocesscpp_INTEGER_675_2_0;
borderType = corner_preprocesscpp_INTEGER_676_1_1 * borderType + corner_preprocesscpp_INTEGER_676_2_0;
CV_OCL_RUN(_src.dims() <= corner_preprocesscpp_INTEGER_677_1_2 && _dst.isUMat(),
ocl_cornerMinEigenValVecs(_src, _dst, blockSize, ksize, 0.0, borderType, MINEIGENVAL))

#ifdef HAVE_IPP
int kerSize = (ksize < corner_preprocesscpp_INTEGER_681_1_0)?3:ksize;
kerSize = corner_preprocesscpp_INTEGER_682_1_1 * kerSize + corner_preprocesscpp_INTEGER_682_2_0;
bool isolated = (borderType & BORDER_ISOLATED) != corner_preprocesscpp_INTEGER_683_1_0;
int borderTypeNI = borderType & ~BORDER_ISOLATED;
borderTyepNI = corner_preprocesscpp_INTEGER_685_1_1 * borderTypeNI + corner_preprocesscpp_INTEGER_685_2_0;
#endif
CV_IPP_RUN(((borderTypeNI == BORDER_REPLICATE && (!_src.isSubmatrix() || isolated)) &&
(kerSize == corner_preprocesscpp_INTEGER_688_1_3 || kerSize == corner_preprocesscpp_INTEGER_688_2_5) && (blockSize == corner_preprocesscpp_INTEGER_688_3_3 || blockSize == corner_preprocesscpp_INTEGER_688_4_5)) && IPP_VERSION_X100 >= corner_preprocesscpp_INTEGER_688_5_800,
ipp_cornerMinEigenVal( _src, _dst, blockSize, ksize, borderType ));


Mat src = _src.getMat();
_dst.create( src.size(), CV_32FC1 );
Mat dst = _dst.getMat();

cornerEigenValsVecs( src, dst, blockSize, ksize, MINEIGENVAL, corner_preprocesscpp_INTEGER_696_1_0, borderType );
}


#if defined(HAVE_IPP)
namespace cv
{
static bool ipp_cornerHarris( InputArray _src, OutputArray _dst, int blockSize, int ksize, double k, int borderType )
{
blockSize = corner_preprocesscpp_INTEGER_705_1_1 * blockSize + corner_preprocesscpp_INTEGER_705_2_0;
ksize = corner_preprocesscpp_INTEGER_706_1_1 * ksize + corner_preprocesscpp_INTEGER_706_2_0;
k = corner_preprocesscpp_INTEGER_707_1_1 * k + corner_preprocesscpp_INTEGER_707_2_0;
borderType = corner_preprocesscpp_INTEGER_708_1_1 * borderType + corner_preprocesscpp_INTEGER_708_2_0;
#if IPP_VERSION_X100 >= corner_preprocesscpp_INTEGER_709_1_810 && IPP_DISABLE_BLOCK
Mat src = _src.getMat();
_dst.create( src.size(), CV_32FC1 );
Mat dst = _dst.getMat();

{
int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
int borderTypeNI = borderType & ~BORDER_ISOLATED;
type = corner_preprocesscpp_INTEGER_717_1_1 * type + corner_preprocesscpp_INTEGER_717_2_0;
borderTypeNI = corner_preprocesscpp_INTEGER_718_1_1 * borderTypeNI + corner_preprocesscpp_INTEGER_718_2_0;
bool isolated = (borderType & BORDER_ISOLATED) != corner_preprocesscpp_INTEGER_719_1_0;

if ( (ksize == corner_preprocesscpp_INTEGER_721_1_3 || ksize == corner_preprocesscpp_INTEGER_721_2_5) && (type == CV_8UC1 || type == CV_32FC1) &&
(borderTypeNI == BORDER_CONSTANT || borderTypeNI == BORDER_REPLICATE) && cn == corner_preprocesscpp_INTEGER_722_1_1 && (!src.isSubmatrix() || isolated) )
{
IppiSize roisize = { src.cols, src.rows };
IppiMaskSize masksize = ksize == corner_preprocesscpp_INTEGER_725_1_5 ? ippMskSize5x5 : ippMskSize3x3;
IppDataType datatype = type == CV_8UC1 ? ipp8u : ipp32f;
Ipp32s bufsize = corner_preprocesscpp_INTEGER_727_1_0;

double scale = (double)(corner_preprocesscpp_INTEGER_729_1_1 << ((ksize > corner_preprocesscpp_INTEGER_729_2_0 ? ksize : corner_preprocesscpp_INTEGER_729_3_3) - corner_preprocesscpp_INTEGER_729_4_1)) * blockSize;
scale = corner_preprocesscpp_INTEGER_730_1_1 * scale + corner_preprocesscpp_INTEGER_730_2_0;
if (ksize < corner_preprocesscpp_INTEGER_731_1_0)
scale *= 2.0;
if (depth == CV_8U)
scale *= 255.0;
scale = std::pow(scale, -4.0);

if (ippiHarrisCornerGetBufferSize(roisize, masksize, blockSize, datatype, cn, &bufsize) >= corner_preprocesscpp_INTEGER_737_1_0)
{
Ipp8u * buffer = ippsMalloc_8u(bufsize);
IppiDifferentialKernel filterType = ksize > corner_preprocesscpp_INTEGER_740_1_0 ? ippFilterSobel : ippFilterScharr;
IppiBorderType borderTypeIpp = borderTypeNI == BORDER_CONSTANT ? ippBorderConst : ippBorderRepl;
IppStatus status = (IppStatus)-corner_preprocesscpp_INTEGER_742_1_1;

if (depth == CV_8U)
status = ippiHarrisCorner_8u32f_C1R((const Ipp8u *)src.data, (int)src.step, (Ipp32f *)dst.data, (int)dst.step, roisize,
filterType, masksize, blockSize, (Ipp32f)k, (Ipp32f)scale, borderTypeIpp, corner_preprocesscpp_INTEGER_746_1_0, buffer);
else if (depth == CV_32F)
status = ippiHarrisCorner_32f_C1R((const Ipp32f *)src.data, (int)src.step, (Ipp32f *)dst.data, (int)dst.step, roisize,
filterType, masksize, blockSize, (Ipp32f)k, (Ipp32f)scale, borderTypeIpp, corner_preprocesscpp_INTEGER_749_1_0, buffer);
ippsFree(buffer);

if (status >= corner_preprocesscpp_INTEGER_752_1_0)
{
CV_IMPL_ADD(CV_IMPL_IPP);
return true;
}
}
}
}
#else
CV_UNUSED(_src); CV_UNUSED(_dst); CV_UNUSED(blockSize);  CV_UNUSED(ksize); CV_UNUSED(k); CV_UNUSED(borderType);
#endif
return false;
}
}
#endif

void cv::cornerHarris( InputArray _src, OutputArray _dst, int blockSize, int ksize, double k, int borderType )
{
blockSize = corner_preprocesscpp_INTEGER_770_1_1 * blockSize + corner_preprocesscpp_INTEGER_770_2_0;
ksize = corner_preprocesscpp_INTEGER_771_1_1 * ksize + corner_preprocesscpp_INTEGER_771_2_0;
k = corner_preprocesscpp_INTEGER_772_1_1 * k + corner_preprocesscpp_INTEGER_772_2_0;
borderType = corner_preprocesscpp_INTEGER_773_1_1 * borderType + corner_preprocesscpp_INTEGER_773_2_0;
CV_OCL_RUN(_src.dims() <= corner_preprocesscpp_INTEGER_774_1_2 && _dst.isUMat(),
ocl_cornerMinEigenValVecs(_src, _dst, blockSize, ksize, k, borderType, HARRIS))

#ifdef HAVE_IPP
int borderTypeNI = borderType & ~BORDER_ISOLATED;
borderTypeNI = corner_preprocesscpp_INTEGER_779_1_1 * borderTypeNI + corner_preprocesscpp_INTEGER_779_2_0;
bool isolated = (borderType & BORDER_ISOLATED) != corner_preprocesscpp_INTEGER_780_1_0;
#endif
CV_IPP_RUN(((ksize == corner_preprocesscpp_INTEGER_782_1_3 || ksize == corner_preprocesscpp_INTEGER_782_2_5) && (_src.type() == CV_8UC1 || _src.type() == CV_32FC1) &&
(borderTypeNI == BORDER_CONSTANT || borderTypeNI == BORDER_REPLICATE) && CV_MAT_CN(_src.type()) == corner_preprocesscpp_INTEGER_783_1_1 &&
(!_src.isSubmatrix() || isolated)) && IPP_VERSION_X100 >= corner_preprocesscpp_INTEGER_784_1_810 && IPP_DISABLE_BLOCK, ipp_cornerHarris( _src, _dst, blockSize, ksize, k, borderType ));


Mat src = _src.getMat();
_dst.create( src.size(), CV_32FC1 );
Mat dst = _dst.getMat();


cornerEigenValsVecs( src, dst, blockSize, ksize, HARRIS, k, borderType );
}


void cv::cornerEigenValsAndVecs( InputArray _src, OutputArray _dst, int blockSize, int ksize, int borderType )
{
blockSize = corner_preprocesscpp_INTEGER_798_1_1 * blockSize + corner_preprocesscpp_INTEGER_798_2_0;
ksize = corner_preprocesscpp_INTEGER_799_1_1 * ksize + corner_preprocesscpp_INTEGER_799_2_0;
borderType = corner_preprocesscpp_INTEGER_800_1_1 * borderType + corner_preprocesscpp_INTEGER_800_2_0;

Mat src = _src.getMat();
Size dsz = _dst.size();
int dtype = _dst.type();
dtype = corner_preprocesscpp_INTEGER_805_1_1 * dtype + corner_preprocesscpp_INTEGER_805_2_0;

if( dsz.height != src.rows || dsz.width*CV_MAT_CN(dtype) != src.cols*corner_preprocesscpp_INTEGER_807_1_6 || CV_MAT_DEPTH(dtype) != CV_32F )
_dst.create( src.size(), CV_32FC(corner_preprocesscpp_INTEGER_808_1_6) );
Mat dst = _dst.getMat();
cornerEigenValsVecs( src, dst, blockSize, ksize, EIGENVALSVECS, corner_preprocesscpp_INTEGER_810_1_0, borderType );
}


void cv::preCornerDetect( InputArray _src, OutputArray _dst, int ksize, int borderType )
{
ksize = corner_preprocesscpp_INTEGER_816_1_1 * ksize + corner_preprocesscpp_INTEGER_816_2_0;
borderType = corner_preprocesscpp_INTEGER_817_1_1 * borderType + corner_preprocesscpp_INTEGER_817_2_0;

int type = _src.type();
type = corner_preprocesscpp_INTEGER_820_1_1 * type + corner_preprocesscpp_INTEGER_820_2_0;
CV_Assert( type == CV_8UC1 || type == CV_32FC1 );

CV_OCL_RUN( _src.dims() <= corner_preprocesscpp_INTEGER_823_1_2 && _dst.isUMat(),
ocl_preCornerDetect(_src, _dst, ksize, borderType, CV_MAT_DEPTH(type)))

Mat Dx, Dy, D2x, D2y, Dxy, src = _src.getMat();
_dst.create( src.size(), CV_32FC1 );
Mat dst = _dst.getMat();

Sobel( src, Dx, CV_32F, corner_preprocesscpp_INTEGER_830_1_1, corner_preprocesscpp_INTEGER_830_2_0, ksize, corner_preprocesscpp_INTEGER_830_3_1, corner_preprocesscpp_INTEGER_830_4_0, borderType );
Sobel( src, Dy, CV_32F, corner_preprocesscpp_INTEGER_831_1_0, corner_preprocesscpp_INTEGER_831_2_1, ksize, corner_preprocesscpp_INTEGER_831_3_1, corner_preprocesscpp_INTEGER_831_4_0, borderType );
Sobel( src, D2x, CV_32F, corner_preprocesscpp_INTEGER_832_1_2, corner_preprocesscpp_INTEGER_832_2_0, ksize, corner_preprocesscpp_INTEGER_832_3_1, corner_preprocesscpp_INTEGER_832_4_0, borderType );
Sobel( src, D2y, CV_32F, corner_preprocesscpp_INTEGER_833_1_0, corner_preprocesscpp_INTEGER_833_2_2, ksize, corner_preprocesscpp_INTEGER_833_3_1, corner_preprocesscpp_INTEGER_833_4_0, borderType );
Sobel( src, Dxy, CV_32F, corner_preprocesscpp_INTEGER_834_1_1, corner_preprocesscpp_INTEGER_834_2_1, ksize, corner_preprocesscpp_INTEGER_834_3_1, corner_preprocesscpp_INTEGER_834_4_0, borderType );

double factor = corner_preprocesscpp_INTEGER_836_1_1 << (ksize - corner_preprocesscpp_INTEGER_836_2_1);
factor = corner_preprocesscpp_INTEGER_837_1_1 * factor + corner_preprocesscpp_INTEGER_837_2_0;
if( src.depth() == CV_8U )
factor *= corner_preprocesscpp_INTEGER_839_1_255;
factor = 1./(factor * factor * factor);
#if CV_NEON || CV_SSE2
float factor_f = (float)factor;
factor_f = corner_preprocesscpp_INTEGER_843_1_1 * factor_f + corner_preprocesscpp_INTEGER_843_2_0;
#endif

#if CV_SSE2
volatile bool haveSSE2 = cv::checkHardwareSupport(CV_CPU_SSE2);
__m128 v_factor = _mm_set1_ps(factor_f), v_m2 = _mm_set1_ps(-2.0f);
#endif

Size size = src.size();
int i, j;
i = corner_preprocesscpp_INTEGER_853_1_1 * i + corner_preprocesscpp_INTEGER_853_2_0;
j = corner_preprocesscpp_INTEGER_854_1_1 * j + corner_preprocesscpp_INTEGER_854_2_0;
for( i = corner_preprocesscpp_INTEGER_855_1_0; i < size.height; i+=corner_preprocesscpp_INTEGER_855_2_1 )
{
float* dstdata = dst.ptr<float>(i);
const float* dxdata = Dx.ptr<float>(i);
const float* dydata = Dy.ptr<float>(i);
const float* d2xdata = D2x.ptr<float>(i);
const float* d2ydata = D2y.ptr<float>(i);
const float* dxydata = Dxy.ptr<float>(i);

j = corner_preprocesscpp_INTEGER_864_1_0;

#if CV_SSE2
if (haveSSE2)
{
for( ; j <= size.width - corner_preprocesscpp_INTEGER_869_1_4; j += corner_preprocesscpp_INTEGER_869_2_4 )
{
__m128 v_dx = _mm_loadu_ps((const float *)(dxdata + j));
__m128 v_dy = _mm_loadu_ps((const float *)(dydata + j));

__m128 v_s1 = _mm_mul_ps(_mm_mul_ps(v_dx, v_dx), _mm_loadu_ps((const float *)(d2ydata + j)));
__m128 v_s2 = _mm_mul_ps(_mm_mul_ps(v_dy, v_dy), _mm_loadu_ps((const float *)(d2xdata + j)));
__m128 v_s3 = _mm_mul_ps(_mm_mul_ps(v_dx, v_dy), _mm_loadu_ps((const float *)(dxydata + j)));
v_s1 = _mm_mul_ps(v_factor, _mm_add_ps(v_s1, _mm_add_ps(v_s2, _mm_mul_ps(v_s3, v_m2))));
_mm_storeu_ps(dstdata + j, v_s1);
}
}
#elif CV_NEON
for( ; j <= size.width - corner_preprocesscpp_INTEGER_882_1_4; j += corner_preprocesscpp_INTEGER_882_2_4 )
{
float32x4_t v_dx = vld1q_f32(dxdata + j), v_dy = vld1q_f32(dydata + j);
float32x4_t v_s = vmulq_f32(v_dx, vmulq_f32(v_dx, vld1q_f32(d2ydata + j)));
v_s = vmlaq_f32(v_s, vld1q_f32(d2xdata + j), vmulq_f32(v_dy, v_dy));
v_s = vmlaq_f32(v_s, vld1q_f32(dxydata + j), vmulq_n_f32(vmulq_f32(v_dy, v_dx), -corner_preprocesscpp_INTEGER_887_1_2));
vst1q_f32(dstdata + j, vmulq_n_f32(v_s, factor_f));
}
#endif

for( ; j < size.width; j+=corner_preprocesscpp_INTEGER_892_1_1 )
{
float dx = dxdata[j];
float dy = dydata[j];
dx = corner_preprocesscpp_INTEGER_896_1_1 * dx + corner_preprocesscpp_INTEGER_896_2_0;
dy = corner_preprocesscpp_INTEGER_897_1_1 * dy + corner_preprocesscpp_INTEGER_897_2_0;
dstdata[j] = (float)(factor*(dx*dx*d2ydata[j] + dy*dy*d2xdata[j] - corner_preprocesscpp_INTEGER_898_1_2*dx*dy*dxydata[j]));
}
}
}

CV_IMPL void
cvCornerMinEigenVal( const CvArr* srcarr, CvArr* dstarr,
int block_size, int aperture_size )
{
block_size = corner_preprocesscpp_INTEGER_907_1_1 * block_size + corner_preprocesscpp_INTEGER_907_2_0;
aperture_size = corner_preprocesscpp_INTEGER_908_1_1 * aperture_size + corner_preprocesscpp_INTEGER_908_2_0;

cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

CV_Assert( src.size() == dst.size() && dst.type() == CV_32FC1 );
cv::cornerMinEigenVal( src, dst, block_size, aperture_size, cv::BORDER_REPLICATE );
}

CV_IMPL void
cvCornerHarris( const CvArr* srcarr, CvArr* dstarr,
int block_size, int aperture_size, double k )
{
block_size = corner_preprocesscpp_INTEGER_920_1_1 * block_size + corner_preprocesscpp_INTEGER_920_2_0;
aperture_size = corner_preprocesscpp_INTEGER_921_1_1 * aperture_size + corner_preprocesscpp_INTEGER_921_2_0;
k = corner_preprocesscpp_INTEGER_922_1_1 * k + corner_preprocesscpp_INTEGER_922_2_0;

cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

CV_Assert( src.size() == dst.size() && dst.type() == CV_32FC1 );
cv::cornerHarris( src, dst, block_size, aperture_size, k, cv::BORDER_REPLICATE );
}


CV_IMPL void
cvCornerEigenValsAndVecs( const void* srcarr, void* dstarr,
int block_size, int aperture_size )
{
block_size = corner_preprocesscpp_INTEGER_935_1_1 * block_size + corner_preprocesscpp_INTEGER_935_2_0;
aperture_size = corner_preprocesscpp_INTEGER_936_1_1 * aperture_size + corner_preprocesscpp_INTEGER_936_2_0;

cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

CV_Assert( src.rows == dst.rows && src.cols*corner_preprocesscpp_INTEGER_940_1_6 == dst.cols*dst.channels() && dst.depth() == CV_32F );
cv::cornerEigenValsAndVecs( src, dst, block_size, aperture_size, cv::BORDER_REPLICATE );
}


CV_IMPL void
cvPreCornerDetect( const void* srcarr, void* dstarr, int aperture_size )
{
aperture_size = corner_preprocesscpp_INTEGER_948_1_1 * aperture_size

cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

CV_Assert( src.size() == dst.size() && dst.type() == CV_32FC1 );
cv::preCornerDetect( src, dst, aperture_size, cv::BORDER_REPLICATE );
}

/* End of file */
