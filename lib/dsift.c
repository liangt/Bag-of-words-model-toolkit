#include "dsift.h"

typedef void*(*allocator_t)(int, int*);  // 动态内存分配

VL_EXPORT void denseSIFT( float* img, int imgWidth, int imgHeight, int step, int binSize, allocator_t allocator){
	VlDsiftFilter* vlf = vl_dsift_new_basic(imgWidth, imgHeight, step, binSize);
	vl_dsift_process(vlf, img);
	int numFrames = vlf->numFrames;
	int num = 128 * numFrames;
	int shape1[] = {numFrames, 128} ;
	int shape2[] = {numFrames, 2} ;
	float* descrs = (float*)allocator(2, shape1);
	float* frames = (float*)allocator(2, shape2);
	for(int i=0; i<num; i++)
		descrs[i] = vlf->descrs[i];
	for(int i=0; i<numFrames; i++){
        num = i << 1;
		frames[num] = vlf->frames[i].x;
		frames[num+1] = vlf->frames[i].y;
	}
	vl_dsift_delete(vlf);
}
