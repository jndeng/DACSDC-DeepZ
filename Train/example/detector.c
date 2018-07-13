#include "darknet.h"
#include <assert.h>
#include <unistd.h>

#define W 640
#define H 360

typedef struct img_args {
	char* path;
	float* data;
	int bsize;
}img_args;

typedef struct det_batch_args {
	network* net;
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


void detect(network *net, int *pd_bboxes, int imgw, int imgh, int batch, float *data) {
	network_predict(net, data);
	get_focal_bboxes_in_parallel(net, pd_bboxes, imgw, imgh, batch);
}

void* detect_batch(void* args) {
	det_batch_args* ptr = (det_batch_args *)args;
	detect(ptr->net, ptr->pd_bboxes, ptr->w, ptr->h, ptr->mini_batch, ptr->data);
	return 0;
}

void* load_imgae_t(void* args) {
	img_args* ptr = (img_args *)args;
	image im = load_image(ptr->path, 0, 0, 3);
	memcpy(ptr->data, im.data, ptr->bsize*sizeof(float));
	free_image(im);
	return 0;
}

void load_image_in_parallel(char **paths, float *data, int bsize, int batch) {
	pthread_t *threads = (pthread_t *)calloc(batch, sizeof(pthread_t));
	img_args *args = (img_args *)calloc(batch, sizeof(img_args));
	int i;
	for (i = 0; i < batch; ++i) {
		args[i].path = paths[i];
		args[i].data = data + i*bsize;
		args[i].bsize = bsize;
		pthread_create(threads + i, 0, load_imgae_t, args + i);
	}
	for (i = 0; i < batch; ++i) {
		pthread_join(threads[i], 0);
	}
	free(threads);
	free(args);
}

void* load_image_batch(void* args) {
	img_batch_args* ptr = (img_batch_args *)args;
	int i;
	for (i = 0; i < ptr->batch; ++i) {
		image im = load_image((ptr->paths)[i], 0, 0, 3);
		memcpy(ptr->data + i*ptr->bsize, im.data, ptr->bsize*sizeof(float));
		free_image(im);
	}
	return 0;
}

int *detect_all_in_parallel(network *net, char **paths, int tot_img, int batch) {
	int w = W, h = H, bsize = W*H*3;
	int *res_bboxes = calloc(tot_img << 2, sizeof(int));
	// inference
	int n = tot_img / batch + 1;
	int i, b_offset = batch, p_offset = 0, mini_batch = 1;  // mini_batch related to image loader
	float **data = calloc(2, sizeof(float*));
	data[0] = calloc(batch*bsize, sizeof(float));
	data[1] = calloc(batch*bsize, sizeof(float));
	// load the first batch
	load_image_in_parallel(paths, data[0], bsize, batch);
	pthread_t pd_thread, im_thread;
	for (i = 1; i < n; ++i) {
		// set mini-batch size
		if (i == n - 1) mini_batch = tot_img % batch;
		else mini_batch = batch;

		// inference
		det_batch_args det_batch_arg = { 0 };
		det_batch_arg.net = net;
		det_batch_arg.w = w;
		det_batch_arg.h = h;
		det_batch_arg.mini_batch = batch;
		det_batch_arg.data = data[~i & 1];
		det_batch_arg.pd_bboxes = res_bboxes + p_offset;
		pthread_create(&pd_thread, 0, detect_batch, &det_batch_arg);

		// load data (for next batch)
		img_batch_args img_batch_arg = { 0 };
		img_batch_arg.paths = paths + b_offset;
		img_batch_arg.data = data[i & 1];
		img_batch_arg.bsize = bsize;
		img_batch_arg.batch = mini_batch;
		pthread_create(&im_thread, 0, load_image_batch, &img_batch_arg);

		pthread_join(im_thread, 0);
		pthread_join(pd_thread, 0);

		p_offset += (batch << 2);
		b_offset += batch;
	}
	if (tot_img % batch != 0) {
		det_batch_args det_batch_arg = { 0 };
		det_batch_arg.net = net;
		det_batch_arg.w = w;
		det_batch_arg.h = h;
		det_batch_arg.mini_batch = mini_batch;
		det_batch_arg.data = data[~n & 1];
		det_batch_arg.pd_bboxes = res_bboxes + p_offset;
		pthread_create(&pd_thread, 0, detect_batch, &det_batch_arg);
		pthread_join(pd_thread, 0);
	}
	free(data[0]);
	free(data[1]);
	free(data);
	return res_bboxes;
}

double validate_detector(char *cfg, char *weights, char *va_dir, char *va_gt_dir) {
	// load validation data path
	list *val_img_plist = get_paths(va_dir);
	char **val_img_paths = (char **)list_to_array(val_img_plist);
	int tot_val_image = val_img_plist->size;
	free_list(val_img_plist);
	int *gt_bboxes = calloc(tot_val_image << 2, sizeof(int));
	FILE *fp = fopen(va_gt_dir, "r");
	int i;
	for (i = 0; i < tot_val_image; ++i)
		fscanf(fp, "%d %d %d %d", &gt_bboxes[(i << 2) | 0], &gt_bboxes[(i << 2) | 1], &gt_bboxes[(i << 2) | 2], &gt_bboxes[(i << 2) | 3]);
	fclose(fp);

	// load validation network
	network *net = load_network(cfg, weights, 0);

	// inference 
	int *res_bboxes = detect_all_in_parallel(net, val_img_paths, tot_val_image, net->batch);

	// validate IoU
	double tot_iou = 0.0;
	for (i = 0; i < tot_val_image; ++i) {
		boundingbox gt_bbox = { 0 };
		gt_bbox.x1 = gt_bboxes[(i << 2) | 0];
		gt_bbox.y1 = gt_bboxes[(i << 2) | 1];
		gt_bbox.x2 = gt_bboxes[(i << 2) | 2];
		gt_bbox.y2 = gt_bboxes[(i << 2) | 3];
		boundingbox pd_bbox = { 0 };
		pd_bbox.x1 = res_bboxes[(i << 2) | 0];
		pd_bbox.y1 = res_bboxes[(i << 2) | 1];
		pd_bbox.x2 = res_bboxes[(i << 2) | 2];
		pd_bbox.y2 = res_bboxes[(i << 2) | 3];
		double cur_iou = calc_iou(pd_bbox, gt_bbox);
		tot_iou += cur_iou;
	}

	// free memory
	free(val_img_paths);
	free(gt_bboxes);
	free(res_bboxes);
	free_network(net);

  // return average IoU
	return tot_iou / tot_val_image;
}

void test_detector(char *cfg, char *weights, float *data, int *pd_bbox) {
	// load network
	network *net = load_network(cfg, weights, 0);
	// inference (assume the shape of input image is correct)
	detect(net, pd_bbox, W, H, 1, data);
}

void test_detector_from_file(char *cfg, char *weights, char *img_src, int *pd_bbox) {
	// load image and resize to fit the shape of network input
	image img = load_image(img_src, W, H, 3);
	test_detector(cfg, weights, img.data, pd_bbox);
	free_image(img);
}

void train_detector(char *tr_cfg, char *va_cfg, char *weights, char *tr_dir, char *va_dir, char *va_gt_dir, char *model_name, char *model_dir, char *log_dir, int clear, float avg_loss) {

#ifdef GPU
	cuda_set_device(gpu_index);
#endif

	srand(time(0));
	int seed = rand();
	srand(seed);
	network *net = load_network(tr_cfg, weights, clear);

	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);

	// initialize data loader
	data train, buffer;
	layer l = net->layers[net->n - 1];

	load_args args = get_base_args(net);
	args.coords = l.coords;
	args.n = net->batch*net->subdivisions;  // batchsize
	args.classes = l.classes;
	args.jitter = l.jitter;
	args.num_boxes = l.max_boxes;
	args.d = &buffer;

	// fetch training data paths
	args.type = net->random ? DETECTION_DATA : DAC_DATA;
	args.threads = args.n;  // batchsize (1 image/thread)

	if (DAC_DATA == args.type) {  // DAC_DATA
								  // random & balanced batch data
		int i;
		list *save_plist = get_paths(tr_dir);
		char **save_paths = (char **)list_to_array(save_plist);
		int tot_cls = save_plist->size;
		printf("tot cls: %d\n", tot_cls);
		free_list(save_plist);
		assert(args.threads == tot_cls);
		class_paths *cls_paths = calloc(tot_cls, sizeof(class_paths));
		for (i = 0; i < tot_cls; ++i) {
			list *train_img_plist = get_paths(save_paths[i]);
			cls_paths[i].paths = (char **)list_to_array(train_img_plist);
			cls_paths[i].size = train_img_plist->size;
			free_list(train_img_plist);
		}
		free(save_paths);
		args.cls_paths = cls_paths;
	}
	else {  // DETECTION_DATA
			// random & unbalanced batch data
		list *plist = get_paths(tr_dir);
		char **paths = (char **)list_to_array(plist);
		args.m = plist->size;
		args.paths = paths;
		free_list(plist);
	}

	char tmp_weights[256], log_file[256], best_weights[256];
	double max_avg_iou = -1.0;
	int cur_batch = 1;
	int valid_detector = 1;

	printf("avg loss: %f\n", avg_loss);

	// load the first batch of data
	pthread_t load_thread = load_data(args);

	while (cur_batch <= net->max_batches) {
		pthread_join(load_thread, 0);
		train = buffer;
		load_thread = load_data(args);

		float cur_loss = train_network(net, train);
		free_data(train);

		if (avg_loss < 0) avg_loss = cur_loss;
		avg_loss = avg_loss*.9 + cur_loss*.1;

		printf("%d: %.5f cur, %.5f avg, %f rate\n", cur_batch, cur_loss, avg_loss, get_current_rate(net));

		if ((1000 == cur_batch) || (0 == cur_batch % 20000)) {
			// BUG: found memory leak when enable validation
			if (valid_detector) { // validate model
				// save current model
				sprintf(tmp_weights, "%s/%s_tmp.weights", model_dir, model_name);
				save_weights(net, tmp_weights);
				// validate current model
				double cur_avg_iou = validate_detector(va_cfg, tmp_weights, va_dir, va_gt_dir);
				// save the best model up to now
				if (cur_avg_iou > max_avg_iou) {
					max_avg_iou = cur_avg_iou;
					sprintf(best_weights, "%s/%s_best.weights", model_dir, model_name);
					save_weights(net, best_weights);
				}
				// log the valiation result
				sprintf(log_file, "%s/%s.log", log_dir, model_name);
				FILE *fp = fopen(log_file, "a");
				fprintf(fp, "batch %d, cur avg loss %f, cur avg iou %f, best avg iou %f, lr %f\n", cur_batch, avg_loss, cur_avg_iou, max_avg_iou, get_current_rate(net));
				fclose(fp);
			}
			else {
				sprintf(tmp_weights, "%s/%s_%d.weights", model_dir, model_name, cur_batch);
				save_weights(net, tmp_weights);
			}
		}
		cur_batch += 1;
	}
}

void run_detector(int argc, char **argv) {
	char *tr_cfg = find_char_arg(argc, argv, "-tr_cfg", 0);
	char *va_cfg = find_char_arg(argc, argv, "-va_cfg", 0);
	char *weights = find_char_arg(argc, argv, "-weights", 0);
	char *tr_dir = find_char_arg(argc, argv, "-tr_dir", 0);
	char *va_dir = find_char_arg(argc, argv, "-va_dir", 0);
	char *va_gt_dir = find_char_arg(argc, argv, "-va_gt_dir", 0);
	char *model_name = find_char_arg(argc, argv, "-model_name", 0);
	char *model_dir = find_char_arg(argc, argv, "-model_dir", 0);
	char *log_dir = find_char_arg(argc, argv, "-log_dir", 0);
	float avg_loss = find_float_arg(argc, argv, "-avg_loss", -1.0);

	if (0 == strcmp(argv[2], "train"))
		train_detector(tr_cfg, va_cfg, weights, tr_dir, va_dir, va_gt_dir, model_name, model_dir, log_dir, 1, avg_loss);
	else if (0 == strcmp(argv[2], "valid"))
		printf("avg iou: %f\n", validate_detector(va_cfg, weights, va_dir, va_gt_dir));
	else if (0 == strcmp(argv[2], "model"))
		parse_network_cfg(tr_cfg);
}
