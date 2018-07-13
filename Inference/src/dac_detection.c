#include "darknet.h"

#define W			640
#define H			360

typedef struct img_args {
	char* path;
	float* data;
	int bsize;
}img_args;

typedef struct det_batch_args {
	void* net;
	int w, h;
	int mini_batch;
	float* data;
	int* pd_bboxes;
}det_batch_args;

typedef struct img_batch_args {
	char** paths;
	float* data;
	int batch;
	int bsize;
}img_batch_args;

void* load_model(char* cfgfile, char* weightfile) {
	cuda_set_device(0);
	return load_network(cfgfile, weightfile, 0);
}

void detect(void *net, int *pd_bboxes, int imgw, int imgh, int batch, float *data) {
	network_predict(net, data);
	get_focal_bboxes_parallel(net, pd_bboxes, imgw, imgh, batch);
	//get_focal_bboxes(net, pd_bboxes, imgw, imgh, batch);
}


void *load_imgae_t(void *args) {
	img_args* ptr = (img_args *)args;
	load_dac_image(ptr->path, ptr->data);
	return 0;
}

void load_image_parallel(char **paths, float *data, int bsize, int batch) {
	pthread_t *threads = (pthread_t *)calloc(batch, sizeof(pthread_t));
	img_args *args = (img_args *)calloc(batch, sizeof(img_args));
	int i, d_offset = 0;
	for (i = 0; i < batch; ++i) {
		args[i].path = paths[i];
		args[i].data = data + d_offset;
		args[i].bsize = bsize;
		pthread_create(threads + i, 0, load_imgae_t, args + i);
		d_offset += bsize;
	}
	for (i = 0; i < batch; ++i) {
		pthread_join(threads[i], 0);
	}
	free(threads);
	free(args);
}

void load_image_serial(char **paths, float *data, int bsize, int batch) {
	int i, d_offset = 0;
	for (i = 0; i < batch; ++i) {
		load_dac_image(paths[i], data + d_offset);
		d_offset += bsize;
	}
}


void *detect_batch(void* args) {
	det_batch_args* ptr = (det_batch_args *)args;
	detect(ptr->net, ptr->pd_bboxes, ptr->w, ptr->h, ptr->mini_batch, ptr->data);
	return 0;
}

void *load_image_batch(void* args) {
	img_batch_args* ptr = (img_batch_args *)args;
	int i, offset = 0;
	for (i = 0; i < ptr->batch; ++i) {
		load_dac_image((ptr->paths)[i], ptr->data + offset);
		offset += ptr->bsize;
	}
	return 0;
}

int *detect_all_parallel(void *net_ptr, void *paths_ptr, int tot_img) {
  network *net = (network *)net_ptr;
	char **paths = (char **)paths_ptr;
	int w = W, h = H, bsize = W*H*3;
	int *res_bboxes = calloc(4*tot_img, sizeof(int));
	// inference
  int batch = net->batch;
	int n = tot_img / batch + 1;
	int i, b_offset = batch, p_offset = 0, mini_batch = 1;  // mini_batch relative to image loader
	float **data = calloc(2, sizeof(float*));
	data[0] = calloc(batch*bsize, sizeof(float));
	data[1] = calloc(batch*bsize, sizeof(float));
	// load the first batch
	load_image_parallel(paths, data[0], bsize, batch);
	pthread_t pd_thread, im_thread;
	for (i = 1; i < n; ++i) {
		// set mini-batch size
		if (i == n - 1) mini_batch = tot_img % batch;
		else mini_batch = batch;
		// inference
		det_batch_args det_batch_arg = {
			.net = net,
			.w = w,
			.h = h,
			.mini_batch = batch,
			.data = data[~i & 1],
			.pd_bboxes = res_bboxes + p_offset
		};
		pthread_create(&pd_thread, 0, detect_batch, &det_batch_arg);
		// load data
		img_batch_args img_batch_arg = {
			.paths = paths + b_offset,
			.data = data[i & 1],
			.bsize = bsize,
			.batch = mini_batch
		};
		pthread_create(&im_thread, 0, load_image_batch, &img_batch_arg);
		pthread_join(im_thread, 0);
		pthread_join(pd_thread, 0);
		
		p_offset += (batch << 2);
		b_offset += batch;
	}
	if (tot_img % batch != 0) {
		det_batch_args det_batch_arg = {
			.net = net,
			.w = w,
			.h = h,
			.mini_batch = mini_batch,
			.data = data[~n & 1],
			.pd_bboxes = res_bboxes + p_offset
		};
		pthread_create(&pd_thread, 0, detect_batch, &det_batch_arg);
		pthread_join(pd_thread, 0);
	}
	free(data[0]);
	free(data[1]);
	free(data);
	return res_bboxes;
}

// #define PARALLEL
int *detect_all_batch(void *net_ptr, void *paths_ptr, int tot_img) {
  network *net = (network *)net_ptr;
	char **paths = (char **)paths_ptr;
	int w = W, h = H, bsize = W*H*3;
	int *res_bboxes = calloc(4*tot_img, sizeof(int));
	// inference
  int batch = net->batch;
	int n = tot_img / batch + 1;
	int i, mini_batch = 1;  // mini_batch relative to image loader
	float *data = calloc(batch*bsize, sizeof(float));
	int b_offset = 0;
	for (i = 0; i < n; ++i) {
		// set mini-batch size
		if (i == n - 1) mini_batch = tot_img % batch;
		else mini_batch = batch;
		// load data serial
#ifdef PARALLEL
		load_image_parallel(paths + b_offset, data, bsize, mini_batch);
#else
		load_image_serial(paths + b_offset, data, bsize, mini_batch);
#endif
		// inference
		detect(net, res_bboxes + i*4*batch, w, h, mini_batch, data);
		b_offset += batch;
	}
	free(data);
	// return predictions
	return res_bboxes;
}
