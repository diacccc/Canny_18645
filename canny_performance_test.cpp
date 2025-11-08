#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include "opencv2/core/hal/intrin.hpp"
#include <deque>

using namespace cv;
using namespace std;
using namespace chrono;


namespace cv
{

#define CANNY_PUSH(map, stack) *map = 2, stack.push_back(map)

#define CANNY_CHECK(m, high, map, stack) \
    if (m > high) \
        CANNY_PUSH(map, stack); \
    else \
        *map = 0

class customizedCanny : public ParallelLoopBody
{
public:
    customizedCanny(const Mat &_src, Mat &_map, std::deque<uchar*> &borderPeaksParallel,
                  int _low, int _high, int _aperture_size, bool _L2gradient) :
        src(_src), src2(_src), map(_map), _borderPeaksParallel(borderPeaksParallel),
        low(_low), high(_high), aperture_size(_aperture_size), L2gradient(_L2gradient)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        for(int i = 0; i < VTraits<v_int8>::vlanes(); ++i)
        {
            smask[i] = 0;
            smask[i + VTraits<v_int8>::vlanes()] = (schar)-1;
        }
        if (true)
            _map.create(src.rows + 2, (int)alignSize((size_t)(src.cols + CV_SIMD_WIDTH + 1), CV_SIMD_WIDTH), CV_8UC1);
        else
#endif
            _map.create(src.rows + 2, src.cols + 2,  CV_8UC1);
        map = _map;
        map.row(0).setTo(1);
        map.row(src.rows + 1).setTo(1);
        mapstep = map.cols;
        needGradient = true;
        cn = src.channels();
    }

    customizedCanny(const Mat &_dx, const Mat &_dy, Mat &_map, std::deque<uchar*> &borderPeaksParallel,
                  int _low, int _high, bool _L2gradient) :
        src(_dx), src2(_dy), map(_map), _borderPeaksParallel(borderPeaksParallel),
        low(_low), high(_high), aperture_size(0), L2gradient(_L2gradient)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        for(int i = 0; i < VTraits<v_int8>::vlanes(); ++i)
        {
            smask[i] = 0;
            smask[i + VTraits<v_int8>::vlanes()] = (schar)-1;
        }
        if (true)
            _map.create(src.rows + 2, (int)alignSize((size_t)(src.cols + CV_SIMD_WIDTH + 1), CV_SIMD_WIDTH), CV_8UC1);
        else
#endif
            _map.create(src.rows + 2, src.cols + 2,  CV_8UC1);
        map = _map;
        map.row(0).setTo(1);
        map.row(src.rows + 1).setTo(1);
        mapstep = map.cols;
        needGradient = false;
        cn = src.channels();
    }

    ~customizedCanny() {}

    customizedCanny& operator=(const customizedCanny&) { return *this; }
    void stage1(const Range &boundaries, Mat& dx, Mat& dy) const
    {
        const int rowStart = max(0, boundaries.start - 1), rowEnd = min(src.rows, boundaries.end + 1);
        double scale = 1.0;

        if(needGradient)
        {
            if (aperture_size == 7)
            {
                scale = 1 / 16.0;
            }
            Sobel(src.rowRange(rowStart, rowEnd), dx, CV_16S, 1, 0, aperture_size, scale, 0, BORDER_REPLICATE);
            Sobel(src.rowRange(rowStart, rowEnd), dy, CV_16S, 0, 1, aperture_size, scale, 0, BORDER_REPLICATE);
        }
    }

    void stage2 (const Range &boundaries, Mat& dx, Mat& dy, std::deque<uchar*> &stack, std::deque<uchar*> &borderPeaksLocal) const
    {
        AutoBuffer<short> dxMax(0), dyMax(0);
        const int rowStart = max(0, boundaries.start - 1), rowEnd = min(src.rows, boundaries.end + 1);
        int *_mag_p, *_mag_a, *_mag_n;
        short *_dx, *_dy, *_dx_a = NULL, *_dy_a = NULL, *_dx_n = NULL, *_dy_n = NULL;
        uchar *_pmap;
        double scale = 1.0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        AutoBuffer<int> buffer(3 * (mapstep * cn + CV_SIMD_WIDTH));
        _mag_p = alignPtr(buffer.data() + 1, CV_SIMD_WIDTH);
        _mag_a = alignPtr(_mag_p + mapstep * cn, CV_SIMD_WIDTH);
        _mag_n = alignPtr(_mag_a + mapstep * cn, CV_SIMD_WIDTH);
#else
        AutoBuffer<int> buffer(3 * (mapstep * cn));
        _mag_p = buffer.data() + 1;
        _mag_a = _mag_p + mapstep * cn;
        _mag_n = _mag_a + mapstep * cn;
#endif

        // For the first time when just 2 rows are filled and for left and right borders
        if(rowStart == boundaries.start)
            memset(_mag_n - 1, 0, mapstep * sizeof(int));
        else
            _mag_n[src.cols] = _mag_n[-1] = 0;

        _mag_a[src.cols] = _mag_a[-1] = _mag_p[src.cols] = _mag_p[-1] = 0;

        // calculate magnitude and angle of gradient, perform non-maxima suppression.
        // fill the map with one of the following values:
        //   0 - the pixel might belong to an edge
        //   1 - the pixel can not belong to an edge
        //   2 - the pixel does belong to an edge
        for (int i = rowStart; i <= boundaries.end; ++i)
        {
            // Scroll the ring buffer
            std::swap(_mag_n, _mag_a);
            std::swap(_mag_n, _mag_p);

            if(i < rowEnd)
            {
                // Next row calculation
                _dx = dx.ptr<short>(i - rowStart);
                _dy = dy.ptr<short>(i - rowStart);

                if (L2gradient)
                {
                    int j = 0, width = src.cols * cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                    for ( ; j <= width - VTraits<v_int16>::vlanes(); j += VTraits<v_int16>::vlanes())
                    {
                        v_int16 v_dx = vx_load((const short*)(_dx + j));
                        v_int16 v_dy = vx_load((const short*)(_dy + j));

                        v_int32 v_dxp_low, v_dxp_high;
                        v_int32 v_dyp_low, v_dyp_high;
                        v_expand(v_dx, v_dxp_low, v_dxp_high);
                        v_expand(v_dy, v_dyp_low, v_dyp_high);

                        v_store_aligned((int *)(_mag_n + j), v_add(v_mul(v_dxp_low, v_dxp_low), v_mul(v_dyp_low, v_dyp_low)));
                        v_store_aligned((int *)(_mag_n + j + VTraits<v_int32>::vlanes()), v_add(v_mul(v_dxp_high, v_dxp_high), v_mul(v_dyp_high, v_dyp_high)));
                    }
#endif
                    for ( ; j < width; ++j)
                        _mag_n[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];
                }
                else
                {
                    int j = 0, width = src.cols * cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                    for(; j <= width - VTraits<v_int16>::vlanes(); j += VTraits<v_int16>::vlanes())
                    {
                        v_int16 v_dx = vx_load((const short *)(_dx + j));
                        v_int16 v_dy = vx_load((const short *)(_dy + j));

                        v_dx = v_reinterpret_as_s16(v_abs(v_dx));
                        v_dy = v_reinterpret_as_s16(v_abs(v_dy));

                        v_int32 v_dx_ml, v_dy_ml, v_dx_mh, v_dy_mh;
                        v_expand(v_dx, v_dx_ml, v_dx_mh);
                        v_expand(v_dy, v_dy_ml, v_dy_mh);

                        v_store_aligned((int *)(_mag_n + j), v_add(v_dx_ml, v_dy_ml));
                        v_store_aligned((int *)(_mag_n + j + VTraits<v_int32>::vlanes()), v_add(v_dx_mh, v_dy_mh));
                    }
#endif
                    for ( ; j < width; ++j)
                        _mag_n[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
                }

                if(cn > 1)
                {
                    std::swap(_dx_n, _dx_a);
                    std::swap(_dy_n, _dy_a);

                    for(int j = 0, jn = 0; j < src.cols; ++j, jn += cn)
                    {
                        int maxIdx = jn;
                        for(int k = 1; k < cn; ++k)
                            if(_mag_n[jn + k] > _mag_n[maxIdx]) maxIdx = jn + k;

                        _mag_n[j] = _mag_n[maxIdx];
                        _dx_n[j] = _dx[maxIdx];
                        _dy_n[j] = _dy[maxIdx];
                    }

                    _mag_n[src.cols] = 0;
                }

                // at the very beginning we do not have a complete ring
                // buffer of 3 magnitude rows for non-maxima suppression
                if (i <= boundaries.start)
                    continue;
            }
            else
            {
                memset(_mag_n - 1, 0, mapstep * sizeof(int));

                if(cn > 1)
                {
                    std::swap(_dx_n, _dx_a);
                    std::swap(_dy_n, _dy_a);
                }
            }

            // From here actual src row is (i - 1)
            // Set left and right border to 1
#if (CV_SIMD || CV_SIMD_SCALABLE)
            if (true)
                _pmap = map.ptr<uchar>(i) + CV_SIMD_WIDTH;
            else
#endif
                _pmap = map.ptr<uchar>(i) + 1;

            _pmap[src.cols] =_pmap[-1] = 1;

            if(cn == 1)
            {
                _dx = dx.ptr<short>(i - rowStart - 1);
                _dy = dy.ptr<short>(i - rowStart - 1);
            }
            else
            {
                _dx = _dx_a;
                _dy = _dy_a;
            }

            const int TG22 = 13573;
            int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            {
                const v_int32 v_low = vx_setall_s32(low);
                const v_int8 v_one = vx_setall_s8(1);

                for (; j <= src.cols - VTraits<v_int8>::vlanes(); j += VTraits<v_int8>::vlanes())
                {
                    v_store_aligned((signed char*)(_pmap + j), v_one);
                    v_int8 v_cmp = v_pack(v_pack(v_gt(vx_load_aligned((const int *)(_mag_a + j)), v_low),
                                                 v_gt(vx_load_aligned((const int *)(_mag_a + j + VTraits<v_int32>::vlanes())), v_low)),
                                          v_pack(v_gt(vx_load_aligned((const int *)(_mag_a + j + 2 * VTraits<v_int32>::vlanes())), v_low),
                                                 v_gt(vx_load_aligned((const int *)(_mag_a + j + 3 * VTraits<v_int32>::vlanes())), v_low)));
                    while (v_check_any(v_cmp))
                    {
                        int l = v_scan_forward(v_cmp);
                        v_cmp = v_and(v_cmp, vx_load(smask + VTraits<v_int8>::vlanes() - 1 - l));
                        int k = j + l;

                        int m = _mag_a[k];
                        short xs = _dx[k];
                        short ys = _dy[k];
                        int x = (int)std::abs(xs);
                        int y = (int)std::abs(ys) << 15;

                        int tg22x = x * TG22;

                        if (y < tg22x)
                        {
                            if (m > _mag_a[k - 1] && m >= _mag_a[k + 1])
                            {
                                CANNY_CHECK(m, high, (_pmap+k), stack);
                            }
                        }
                        else
                        {
                            int tg67x = tg22x + (x << 16);
                            if (y > tg67x)
                            {
                                if (m > _mag_p[k] && m >= _mag_n[k])
                                {
                                    CANNY_CHECK(m, high, (_pmap+k), stack);
                                }
                            }
                            else
                            {
                                int s = (xs ^ ys) < 0 ? -1 : 1;
                                if(m > _mag_p[k - s] && m > _mag_n[k + s])
                                {
                                    CANNY_CHECK(m, high, (_pmap+k), stack);
                                }
                            }
                        }
                    }
                }
            }
#endif
            for (; j < src.cols; j++)
            {
                int m = _mag_a[j];

                if (m > low)
                {
                    short xs = _dx[j];
                    short ys = _dy[j];
                    int x = (int)std::abs(xs);
                    int y = (int)std::abs(ys) << 15;

                    int tg22x = x * TG22;

                    if (y < tg22x)
                    {
                        if (m > _mag_a[j - 1] && m >= _mag_a[j + 1])
                        {
                            CANNY_CHECK(m, high, (_pmap+j), stack);
                            continue;
                        }
                    }
                    else
                    {
                        int tg67x = tg22x + (x << 16);
                        if (y > tg67x)
                        {
                            if (m > _mag_p[j] && m >= _mag_n[j])
                            {
                                CANNY_CHECK(m, high, (_pmap+j), stack);
                                continue;
                            }
                        }
                        else
                        {
                            int s = (xs ^ ys) < 0 ? -1 : 1;
                            if(m > _mag_p[j - s] && m > _mag_n[j + s])
                            {
                                CANNY_CHECK(m, high, (_pmap+j), stack);
                                continue;
                            }
                        }
                    }
                }
                _pmap[j] = 1;
            }
        }        
    }

    void stage3(const Range &boundaries, std::deque<uchar*> &stack, std::deque<uchar*> &borderPeaksLocal) const
    {
        const int rowStart = max(0, boundaries.start - 1), rowEnd = min(src.rows, boundaries.end + 1);
        uchar *pmapLower = (rowStart == 0) ? map.data : (map.data + (boundaries.start + 2) * mapstep);
        uint pmapDiff = (uint)(((rowEnd == src.rows) ? map.datalimit : (map.data + boundaries.end * mapstep)) - pmapLower);

        while (!stack.empty())
        {
            uchar *m = stack.back();
            stack.pop_back();

            // Stops thresholding from expanding to other slices by sending pixels in the borders of each
            // slice in a queue to be serially processed later.
            if((unsigned)(m - pmapLower) < pmapDiff)
            {
                if (!m[-mapstep-1]) CANNY_PUSH((m-mapstep-1), stack);
                if (!m[-mapstep])   CANNY_PUSH((m-mapstep), stack);
                if (!m[-mapstep+1]) CANNY_PUSH((m-mapstep+1), stack);
                if (!m[-1])         CANNY_PUSH((m-1), stack);
                if (!m[1])          CANNY_PUSH((m+1), stack);
                if (!m[mapstep-1])  CANNY_PUSH((m+mapstep-1), stack);
                if (!m[mapstep])    CANNY_PUSH((m+mapstep), stack);
                if (!m[mapstep+1])  CANNY_PUSH((m+mapstep+1), stack);
            }
            else
            {
                borderPeaksLocal.push_back(m);
                ptrdiff_t mapstep2 = m < pmapLower ? mapstep : -mapstep;

                if (!m[-1])         CANNY_PUSH((m-1), stack);
                if (!m[1])          CANNY_PUSH((m+1), stack);
                if (!m[mapstep2-1]) CANNY_PUSH((m+mapstep2-1), stack);
                if (!m[mapstep2])   CANNY_PUSH((m+mapstep2), stack);
                if (!m[mapstep2+1]) CANNY_PUSH((m+mapstep2+1), stack);
            }
        }

        if(!borderPeaksLocal.empty())
        {
            AutoLock lock(mutex);
            _borderPeaksParallel.insert(_borderPeaksParallel.end(), borderPeaksLocal.begin(), borderPeaksLocal.end());
        }
    }
    void operator()(const Range &boundaries) const CV_OVERRIDE
    {

        Mat dx, dy;
        AutoBuffer<short> dxMax(0), dyMax(0);
        std::deque<uchar*> stack, borderPeaksLocal;
        const int rowStart = max(0, boundaries.start - 1), rowEnd = min(src.rows, boundaries.end + 1);
        int *_mag_p, *_mag_a, *_mag_n;
        short *_dx, *_dy, *_dx_a = NULL, *_dy_a = NULL, *_dx_n = NULL, *_dy_n = NULL;
        uchar *_pmap;
        double scale = 1.0;

        if(needGradient)
        {
            if (aperture_size == 7)
            {
                scale = 1 / 16.0;
            }
            Sobel(src.rowRange(rowStart, rowEnd), dx, CV_16S, 1, 0, aperture_size, scale, 0, BORDER_REPLICATE);
            Sobel(src.rowRange(rowStart, rowEnd), dy, CV_16S, 0, 1, aperture_size, scale, 0, BORDER_REPLICATE);
        }
        // _mag_p: previous row, _mag_a: actual row, _mag_n: next row
#if (CV_SIMD || CV_SIMD_SCALABLE)
        AutoBuffer<int> buffer(3 * (mapstep * cn + CV_SIMD_WIDTH));
        _mag_p = alignPtr(buffer.data() + 1, CV_SIMD_WIDTH);
        _mag_a = alignPtr(_mag_p + mapstep * cn, CV_SIMD_WIDTH);
        _mag_n = alignPtr(_mag_a + mapstep * cn, CV_SIMD_WIDTH);
#else
        AutoBuffer<int> buffer(3 * (mapstep * cn));
        _mag_p = buffer.data() + 1;
        _mag_a = _mag_p + mapstep * cn;
        _mag_n = _mag_a + mapstep * cn;
#endif

        // For the first time when just 2 rows are filled and for left and right borders
        if(rowStart == boundaries.start)
            memset(_mag_n - 1, 0, mapstep * sizeof(int));
        else
            _mag_n[src.cols] = _mag_n[-1] = 0;

        _mag_a[src.cols] = _mag_a[-1] = _mag_p[src.cols] = _mag_p[-1] = 0;

        // calculate magnitude and angle of gradient, perform non-maxima suppression.
        // fill the map with one of the following values:
        //   0 - the pixel might belong to an edge
        //   1 - the pixel can not belong to an edge
        //   2 - the pixel does belong to an edge
        for (int i = rowStart; i <= boundaries.end; ++i)
        {
            // Scroll the ring buffer
            std::swap(_mag_n, _mag_a);
            std::swap(_mag_n, _mag_p);

            if(i < rowEnd)
            {
                // Next row calculation
                _dx = dx.ptr<short>(i - rowStart);
                _dy = dy.ptr<short>(i - rowStart);

                if (L2gradient)
                {
                    int j = 0, width = src.cols * cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                    for ( ; j <= width - VTraits<v_int16>::vlanes(); j += VTraits<v_int16>::vlanes())
                    {
                        v_int16 v_dx = vx_load((const short*)(_dx + j));
                        v_int16 v_dy = vx_load((const short*)(_dy + j));

                        v_int32 v_dxp_low, v_dxp_high;
                        v_int32 v_dyp_low, v_dyp_high;
                        v_expand(v_dx, v_dxp_low, v_dxp_high);
                        v_expand(v_dy, v_dyp_low, v_dyp_high);

                        v_store_aligned((int *)(_mag_n + j), v_add(v_mul(v_dxp_low, v_dxp_low), v_mul(v_dyp_low, v_dyp_low)));
                        v_store_aligned((int *)(_mag_n + j + VTraits<v_int32>::vlanes()), v_add(v_mul(v_dxp_high, v_dxp_high), v_mul(v_dyp_high, v_dyp_high)));
                    }
#endif
                    for ( ; j < width; ++j)
                        _mag_n[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];
                }
                else
                {
                    int j = 0, width = src.cols * cn;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                    for(; j <= width - VTraits<v_int16>::vlanes(); j += VTraits<v_int16>::vlanes())
                    {
                        v_int16 v_dx = vx_load((const short *)(_dx + j));
                        v_int16 v_dy = vx_load((const short *)(_dy + j));

                        v_dx = v_reinterpret_as_s16(v_abs(v_dx));
                        v_dy = v_reinterpret_as_s16(v_abs(v_dy));

                        v_int32 v_dx_ml, v_dy_ml, v_dx_mh, v_dy_mh;
                        v_expand(v_dx, v_dx_ml, v_dx_mh);
                        v_expand(v_dy, v_dy_ml, v_dy_mh);

                        v_store_aligned((int *)(_mag_n + j), v_add(v_dx_ml, v_dy_ml));
                        v_store_aligned((int *)(_mag_n + j + VTraits<v_int32>::vlanes()), v_add(v_dx_mh, v_dy_mh));
                    }
#endif
                    for ( ; j < width; ++j)
                        _mag_n[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
                }

                if(cn > 1)
                {
                    std::swap(_dx_n, _dx_a);
                    std::swap(_dy_n, _dy_a);

                    for(int j = 0, jn = 0; j < src.cols; ++j, jn += cn)
                    {
                        int maxIdx = jn;
                        for(int k = 1; k < cn; ++k)
                            if(_mag_n[jn + k] > _mag_n[maxIdx]) maxIdx = jn + k;

                        _mag_n[j] = _mag_n[maxIdx];
                        _dx_n[j] = _dx[maxIdx];
                        _dy_n[j] = _dy[maxIdx];
                    }

                    _mag_n[src.cols] = 0;
                }

                // at the very beginning we do not have a complete ring
                // buffer of 3 magnitude rows for non-maxima suppression
                if (i <= boundaries.start)
                    continue;
            }
            else
            {
                memset(_mag_n - 1, 0, mapstep * sizeof(int));

                if(cn > 1)
                {
                    std::swap(_dx_n, _dx_a);
                    std::swap(_dy_n, _dy_a);
                }
            }

            // From here actual src row is (i - 1)
            // Set left and right border to 1
#if (CV_SIMD || CV_SIMD_SCALABLE)
            if (true)
                _pmap = map.ptr<uchar>(i) + CV_SIMD_WIDTH;
            else
#endif
                _pmap = map.ptr<uchar>(i) + 1;

            _pmap[src.cols] =_pmap[-1] = 1;

            if(cn == 1)
            {
                _dx = dx.ptr<short>(i - rowStart - 1);
                _dy = dy.ptr<short>(i - rowStart - 1);
            }
            else
            {
                _dx = _dx_a;
                _dy = _dy_a;
            }

            const int TG22 = 13573;
            int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            {
                const v_int32 v_low = vx_setall_s32(low);
                const v_int8 v_one = vx_setall_s8(1);

                for (; j <= src.cols - VTraits<v_int8>::vlanes(); j += VTraits<v_int8>::vlanes())
                {
                    v_store_aligned((signed char*)(_pmap + j), v_one);
                    v_int8 v_cmp = v_pack(v_pack(v_gt(vx_load_aligned((const int *)(_mag_a + j)), v_low),
                                                 v_gt(vx_load_aligned((const int *)(_mag_a + j + VTraits<v_int32>::vlanes())), v_low)),
                                          v_pack(v_gt(vx_load_aligned((const int *)(_mag_a + j + 2 * VTraits<v_int32>::vlanes())), v_low),
                                                 v_gt(vx_load_aligned((const int *)(_mag_a + j + 3 * VTraits<v_int32>::vlanes())), v_low)));
                    while (v_check_any(v_cmp))
                    {
                        int l = v_scan_forward(v_cmp);
                        v_cmp = v_and(v_cmp, vx_load(smask + VTraits<v_int8>::vlanes() - 1 - l));
                        int k = j + l;

                        int m = _mag_a[k];
                        short xs = _dx[k];
                        short ys = _dy[k];
                        int x = (int)std::abs(xs);
                        int y = (int)std::abs(ys) << 15;

                        int tg22x = x * TG22;

                        if (y < tg22x)
                        {
                            if (m > _mag_a[k - 1] && m >= _mag_a[k + 1])
                            {
                                CANNY_CHECK(m, high, (_pmap+k), stack);
                            }
                        }
                        else
                        {
                            int tg67x = tg22x + (x << 16);
                            if (y > tg67x)
                            {
                                if (m > _mag_p[k] && m >= _mag_n[k])
                                {
                                    CANNY_CHECK(m, high, (_pmap+k), stack);
                                }
                            }
                            else
                            {
                                int s = (xs ^ ys) < 0 ? -1 : 1;
                                if(m > _mag_p[k - s] && m > _mag_n[k + s])
                                {
                                    CANNY_CHECK(m, high, (_pmap+k), stack);
                                }
                            }
                        }
                    }
                }
            }
#endif
            for (; j < src.cols; j++)
            {
                int m = _mag_a[j];

                if (m > low)
                {
                    short xs = _dx[j];
                    short ys = _dy[j];
                    int x = (int)std::abs(xs);
                    int y = (int)std::abs(ys) << 15;

                    int tg22x = x * TG22;

                    if (y < tg22x)
                    {
                        if (m > _mag_a[j - 1] && m >= _mag_a[j + 1])
                        {
                            CANNY_CHECK(m, high, (_pmap+j), stack);
                            continue;
                        }
                    }
                    else
                    {
                        int tg67x = tg22x + (x << 16);
                        if (y > tg67x)
                        {
                            if (m > _mag_p[j] && m >= _mag_n[j])
                            {
                                CANNY_CHECK(m, high, (_pmap+j), stack);
                                continue;
                            }
                        }
                        else
                        {
                            int s = (xs ^ ys) < 0 ? -1 : 1;
                            if(m > _mag_p[j - s] && m > _mag_n[j + s])
                            {
                                CANNY_CHECK(m, high, (_pmap+j), stack);
                                continue;
                            }
                        }
                    }
                }
                _pmap[j] = 1;
            }
        }

        // Not for first row of first slice or last row of last slice
        uchar *pmapLower = (rowStart == 0) ? map.data : (map.data + (boundaries.start + 2) * mapstep);
        uint pmapDiff = (uint)(((rowEnd == src.rows) ? map.datalimit : (map.data + boundaries.end * mapstep)) - pmapLower);

        while (!stack.empty())
        {
            uchar *m = stack.back();
            stack.pop_back();

            // Stops thresholding from expanding to other slices by sending pixels in the borders of each
            // slice in a queue to be serially processed later.
            if((unsigned)(m - pmapLower) < pmapDiff)
            {
                if (!m[-mapstep-1]) CANNY_PUSH((m-mapstep-1), stack);
                if (!m[-mapstep])   CANNY_PUSH((m-mapstep), stack);
                if (!m[-mapstep+1]) CANNY_PUSH((m-mapstep+1), stack);
                if (!m[-1])         CANNY_PUSH((m-1), stack);
                if (!m[1])          CANNY_PUSH((m+1), stack);
                if (!m[mapstep-1])  CANNY_PUSH((m+mapstep-1), stack);
                if (!m[mapstep])    CANNY_PUSH((m+mapstep), stack);
                if (!m[mapstep+1])  CANNY_PUSH((m+mapstep+1), stack);
            }
            else
            {
                borderPeaksLocal.push_back(m);
                ptrdiff_t mapstep2 = m < pmapLower ? mapstep : -mapstep;

                if (!m[-1])         CANNY_PUSH((m-1), stack);
                if (!m[1])          CANNY_PUSH((m+1), stack);
                if (!m[mapstep2-1]) CANNY_PUSH((m+mapstep2-1), stack);
                if (!m[mapstep2])   CANNY_PUSH((m+mapstep2), stack);
                if (!m[mapstep2+1]) CANNY_PUSH((m+mapstep2+1), stack);
            }
        }

        // if(!borderPeaksLocal.empty())
        // {
        //     AutoLock lock(mutex);
        //     _borderPeaksParallel.insert(_borderPeaksParallel.end(), borderPeaksLocal.begin(), borderPeaksLocal.end());
        // }
    }

private:
    const Mat &src, &src2;
    Mat &map;
    std::deque<uchar*> &_borderPeaksParallel;
    int low, high, aperture_size;
    bool L2gradient, needGradient;
    ptrdiff_t mapstep;
    int cn;
    mutable Mutex mutex;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    schar smask[2*VTraits<v_int8>::max_nlanes];
#endif
};

} // namespace cv

/* End of file. */

struct PerformanceResult {
    int width;
    int height;
    int pixels;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double throughput_fps;
};

double calculateAverage(const vector<double>& times) {
    double sum = 0;
    for (double t : times) sum += t;
    return sum / times.size();
}

PerformanceResult benchmarkCanny(int width, int height, int iterations = 10000) {
    PerformanceResult result;
    result.width = width;
    result.height = height;
    result.pixels = width * height;
    
    // Generate synthetic grayscale image with random content
    Mat gray(height, width, CV_8UC1);
    randu(gray, Scalar(0), Scalar(255));
    
    // Add some shapes for more realistic edge detection
    for (int i = 0; i < min(10, width/50); i++) {
        circle(gray, Point(rand() % width, rand() % height), 
               rand() % (min(width, height)/10) + 10, Scalar(255), 2);
        rectangle(gray, 
                 Point(rand() % width, rand() % height),
                 Point(rand() % width, rand() % height),
                 Scalar(128), 2);
    }
    
    vector<double> times;

    Mat map;
    // Mat dx, dy;
    std::deque<uchar*> borderPeaksParallel, stack;
    cv::setNumThreads(1);    
    
    Mat dx, dy;
    // customizedCanny cc(gray, map, stack, 6, 10, 3, false);
    // cc.stage1(Range(0, gray.rows), dx, dy);
    // cc.stage2(Range(0, gray.rows), dx, dy, stack, borderPeaksParallel);
    // Warm-up runs
    for (int i = 0; i < 5; i++) {
        // Canny(gray, map, 50, 150, 3, false);
        // std::deque<uchar*> stack_copy = stack;
        // cc.stage3(Range(0, gray.rows), stack_copy, borderPeaksParallel);
        customizedCanny cc(gray, map, stack, 6, 10, 3, false);
        cc(Range(0, gray.rows));
    }
    
    // Actual benchmark
    for (int i = 0; i < iterations; i++) {
        auto start = high_resolution_clock::now();
        customizedCanny cc(gray, map, stack, 6, 10, 3, false);
        cc(Range(0, gray.rows));
        // Canny(gray, map, 50, 150, 3, false);
        
        // cc.stage3(Range(0, gray.rows), stack_copy, borderPeaksParallel);
        auto end = high_resolution_clock::now();
        duration<double, milli> elapsed = end - start;
        times.push_back(elapsed.count());
    }
    
    // Calculate statistics
    result.min_time_ms = *min_element(times.begin(), times.end());
    result.max_time_ms = *max_element(times.begin(), times.end());
    result.avg_time_ms = calculateAverage(times);
    result.throughput_fps = 1000.0 / result.avg_time_ms;
    
    return result;
}

void saveResultsToCSV(const vector<PerformanceResult>& results, const string& filename) {
    ofstream file(filename);
    
    // Write header
    file << "width,height,pixels,avg_time_ms,min_time_ms,max_time_ms,std_dev_ms,throughput_fps\n";
    
    // Write data
    for (const auto& result : results) {
        file << result.width << ","
             << result.height << ","
             << result.pixels << ","
             << fixed << setprecision(4)
             << result.avg_time_ms << ","
             << result.min_time_ms << ","
             << result.max_time_ms << ","
             << setprecision(2)
             << result.throughput_fps << "\n";
    }
    
    file.close();
    cout << "Results saved to: " << filename << endl;
}

void printResults(const vector<PerformanceResult>& results) {
    cout << "\n=== Canny Edge Detection Performance Results ===\n" << endl;
    cout << setw(12) << "Size" 
         << setw(15) << "Pixels"
         << setw(15) << "Avg (ms)"
         << setw(15) << "Min (ms)"
         << setw(15) << "Max (ms)"
         << setw(15) << "FPS" << endl;
    cout << string(102, '-') << endl;
    
    for (const auto& result : results) {
        cout << setw(5) << result.width << "x" << setw(5) << result.height
             << setw(15) << result.pixels
             << setw(15) << fixed << setprecision(3) << result.avg_time_ms
             << setw(15) << result.min_time_ms
             << setw(15) << result.max_time_ms
             << setw(15) << setprecision(2) << result.throughput_fps
             << endl;
    }
    cout << endl;
}



int main(int argc, char** argv) {
    int iterations = 100;

    if (argc > 1) {
        iterations = atoi(argv[1]);
    }
    
    // Define test sizes
    vector<pair<int, int>> testSizes = {
        {128, 128},
        {128, 256},
        {256, 256},
        {512, 512},
        {1024, 1024},
        {2048, 2048}
    };
    
    vector<PerformanceResult> results;
    
    cout << "OpenCV Canny Edge Detection Performance Test" << endl;
    cout << "Iterations per size: " << iterations << endl;
    cout << "Running benchmarks...\n" << endl;
    
    // Run benchmarks for each size
    for (const auto& size : testSizes) {
        cout << "Testing size " << size.first << "x" << size.second << "... " << flush;
        
        PerformanceResult result = benchmarkCanny(size.first, size.second, iterations);
        results.push_back(result);
        
        cout << "Done! (" << fixed << setprecision(3) << result.avg_time_ms << " ms)" << endl;
    }
    
    // Print results
    printResults(results);
    
    // Save to CSV
    saveResultsToCSV(results, "canny_performance.csv");
    
    cout << "Run 'python3 plot_performance.py' to generate performance plots." << endl;
    
    return 0;
}