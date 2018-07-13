#include "focal_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

LOSS_TYPE get_focal_loss_type(char *s) {
	if (strcmp(s, "l2") == 0) return L2;
	if (strcmp(s, "ce") == 0) return CE;
	if (strcmp(s, "fl") == 0) return FL;
	if (strcmp(s, "sl1") == 0) return SL1;
	fprintf(stderr, "Couldn't find loss type function %s, going with L2 loss\n", s);
	return L2;
}

STRATEGY_TYPE get_focal_strategy_type(char *s) {
	if (strcmp(s, "yolo2") == 0) return YOLO2;
	if (strcmp(s, "ssd") == 0) return SSD;
	if (strcmp(s, "rcnn") == 0) return RCNN;
	fprintf(stderr, "Couldn't find strategy %s, going with yolo2 strategy\n", s);
	return YOLO2;
}

layer make_focal_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int coords, int max_boxes, char *pos_loss_type_s, char *con_loss_type_s, char *strategy_type_s) {
	layer l = { 0 };
	l.type = FOCAL;

	l.pos_loss_type = get_focal_loss_type(pos_loss_type_s);
	l.con_loss_type = get_focal_loss_type(con_loss_type_s);

	l.strategy_type = get_focal_strategy_type(strategy_type_s);

	l.n = n;
	l.total = total;
	l.mask = mask;

	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = n*(coords + 1);
	l.out_w = l.w;
	l.out_h = l.h;
	l.out_c = l.c;
	l.classes = classes;
	l.coords = coords;
	l.max_boxes = max_boxes;
	l.cost = calloc(1, sizeof(float));
	l.con_cost = calloc(1, sizeof(float));
	l.pos_cost = calloc(1, sizeof(float));
	l.biases = calloc(total*2, sizeof(float));
	l.outputs = l.h*l.w*l.c;
	l.inputs = l.outputs;
	l.truths = max_boxes*(coords + 1);
	l.delta = calloc(batch*l.outputs, sizeof(float));
	l.output = calloc(batch*l.outputs, sizeof(float));
	int i;
	for (i = 0; i < total*2; ++i)
		l.biases[i] = .5;

	l.forward = forward_focal_layer;
	l.backward = backward_focal_layer;
#ifdef GPU
	l.forward_gpu = forward_focal_layer_gpu;
	l.backward_gpu = backward_focal_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

	fprintf(stderr, "focal\n");
	srand(0);

	return l;
}

box get_focal_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride, STRATEGY_TYPE strategy_type) {
	box b;
	if (YOLO2 == strategy_type) {
		b.x = (i + x[index + 0*stride]) / w;
		b.y = (j + x[index + 1*stride]) / h;
		b.w = exp(x[index + 2*stride]) * biases[2*n] / w;
		b.h = exp(x[index + 3*stride]) * biases[2*n + 1] / h;
	} else if (SSD == strategy_type) {
		b.x = (x[index + 0*stride] * biases[2*n] + i + 0.5) / w;
		b.y = (x[index + 1*stride] * biases[2*n + 1] + j + 0.5) / h;
		b.w = exp(x[index + 2*stride]) * biases[2*n] / w;
		b.h = exp(x[index + 3*stride]) * biases[2*n + 1] / h;
	}
	return b;
}

int focal_entry_index(layer l, int batch, int location, int entry) {
	int n =   location / (l.w*l.h);
	int loc = location % (l.w*l.h);
	return batch*l.outputs + n*l.w*l.h*(l.coords + l.classes + 1) + entry*l.w*l.h + loc;
}

void forward_focal_layer(const layer l, network net) {
	int i, j, b, t, n;
	memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			if (YOLO2 == l.strategy_type) {
				int index = focal_entry_index(l, b, n*l.w*l.h, 0);
				activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
			}
			int index = focal_entry_index(l, b, n*l.w*l.h, l.coords);
			activate_array(l.output + index, l.w*l.h, LOGISTIC);
		}
	}
#endif

	if (!net.train) return;

	memset(l.delta, 0, l.outputs*l.batch*sizeof(float));
	float *loss = calloc(l.batch*l.outputs, sizeof(float));
	*(l.cost) = 0;
	*(l.con_cost) = 0;
	*(l.pos_cost) = 0;

	if (YOLO2 == l.strategy_type) {
		for (b = 0; b < l.batch; ++b) {
			for (j = 0; j < l.h; ++j) {
				for (i = 0; i < l.w; ++i) {
					for (n = 0; n < l.n; ++n) {
						int box_index = focal_entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
						box pred = get_focal_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, l.w*l.h, l.strategy_type);
						float best_iou = 0;
						for (t = 0; t < l.max_boxes; ++t) {
							box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
							if (!truth.x) break;
							int truth_cls = net.truth[t*(l.coords + 1) + b*l.truths + 4];
							if (truth_cls < l.class_low || truth_cls >= l.class_high) break;

							float iou = box_iou(pred, truth);
							if (iou > best_iou)
								best_iou = iou;
						}
						int obj_index = focal_entry_index(l, b, n*l.w*l.h + j*l.w + i, l.coords);
						delta_con(l.output, 0, obj_index, l.focal_alpha, l.focal_gamma, loss, l.delta, l.noobject_scale, l.con_loss_type);
						if (best_iou > l.ignore_thresh) {
							l.delta[obj_index] = 0;
							loss[obj_index] = 0;
						}
					}
				}
			}
			for (t = 0; t < l.max_boxes; ++t) {
				box truth = float_to_box(net.truth + t*(l.coords + 1) + b*l.truths, 1);
				if (!truth.x) break;
				float truth_size = truth.w*truth.h*640*360;
				if (truth_size < l.size_low || truth_size > l.size_high) break;
				int truth_cls = net.truth[t*(l.coords + 1) + b*l.truths + 4];
				if (truth_cls < l.class_low || truth_cls >= l.class_high) break;

				float best_iou = 0;
				int best_n = 0;
				i = truth.x*l.w;
				j = truth.y*l.h;
				box truth_shift = truth;
				truth_shift.x = truth_shift.y = 0;
				for (n = 0; n < l.total; ++n) {
					box pred = { 0 };
					pred.w = l.biases[2*n] / l.w;
					pred.h = l.biases[2*n + 1] / l.h;
					float iou = box_iou(pred, truth_shift);
					if (iou > best_iou) {
						best_iou = iou;
						best_n = n;
					}
				}

				int mask_n = int_index(l.mask, best_n, l.n);
				if (mask_n >= 0) {
					int box_index = focal_entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
					delta_pos(l.output, truth, l.biases, best_n, box_index, i, j, l.w, l.h, loss, l.delta, l.coord_scale*(2 - truth.w*truth.h), l.w*l.h, l.pos_loss_type, l.strategy_type);

					int obj_index = focal_entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, l.coords);
					delta_con(l.output, 1, obj_index, 1 - l.focal_alpha, l.focal_gamma, loss, l.delta, l.object_scale, l.con_loss_type);
				}
			}
		}
	} else if (SSD == l.strategy_type) {
		// can only support single feature map bbox extraction
		assert(l.max_boxes == 1);  // consider only one gt bbox in one single image
		indexbox *neg_boxes = calloc(l.w*l.h*l.n, sizeof(indexbox));
		for (b = 0; b < l.batch; ++b) {
			// get the gt bbox first
			box truth_box = float_to_box(net.truth + b*l.truths, 1);
			if (!truth_box.x)  // in case there is no proper gt bbox in an image
				continue;  

			// process positive examples
			int tot_neg = 0, tot_pos = 0;
			float max_iou = -1;
			indexbox max_box = { 0 };
			for (j = 0; j < l.h; ++j) {
				for (i = 0; i < l.w; ++i) {
					for (n = 0; n < l.n; ++n) {
						int index = n*l.w*l.h + j*l.w + i;
						box default_box = {
							.x = (i + 0.5) / l.w,
							.y = (j + 0.5) / l.h,
							.w = l.biases[2*n] / l.w,
							.h = l.biases[2*n + 1] / l.h
						};
						float cur_iou = box_iou(default_box, truth_box);
						if (cur_iou > max_iou) {
							max_iou = cur_iou;
							indexbox tmp_box = { .i = i, .j = j, .n = n };
							max_box = tmp_box;
						}
						if (cur_iou > l.pos_thresh) {  // positive example
							int obj_index = focal_entry_index(l, b, index, l.coords);
							delta_con(l.output, 1, obj_index, l.focal_alpha, l.focal_gamma, loss, l.delta, l.object_scale, l.con_loss_type);
							int box_index = focal_entry_index(l, b, index, 0);
							delta_pos(l.output, truth_box, l.biases, n, box_index, i, j, l.w, l.h, loss, l.delta, l.coord_scale, l.w*l.h, l.pos_loss_type, l.strategy_type);
							tot_pos += 1;
						}
						if (cur_iou < l.neg_thresh) {  // potential negative example
							int obj_index = focal_entry_index(l, b, index, l.coords);
							indexbox tmp_box = { 
								.conf = l.output[obj_index],
								.i = i,.j = j,.n = n 
							};
							neg_boxes[tot_neg] = tmp_box;
							tot_neg += 1;
						}
					}
				}
			}
			if (max_iou < l.pos_thresh) {
				// in case no iou > l.pos_thresh, need to guranteed at least one positive example
				int index = max_box.n*l.w*l.h + max_box.j*l.w + max_box.i;
				int obj_index = focal_entry_index(l, b, index, l.coords);
				delta_con(l.output, 1, obj_index, l.focal_alpha, l.focal_gamma, loss, l.delta, l.object_scale, l.con_loss_type);
				int box_index = focal_entry_index(l, b, index, 0);
				delta_pos(l.output, truth_box, l.biases, max_box.n, box_index, max_box.i, max_box.j, l.w, l.h, loss, l.delta, l.coord_scale, l.w*l.h, l.pos_loss_type, l.strategy_type);
				tot_pos += 1;
			}

			// process negative examples
			sort_bbox(neg_boxes, tot_neg);  // sort all negative bboxes according to confidence
			int max_index = max_box.n*l.w*l.h + max_box.j*l.w + max_box.i;
			int sel_neg = MMIN(tot_pos*3, tot_neg);
			for (i = 0, j = 0; i < tot_neg && j < sel_neg; ++i) {
				indexbox cur_box = neg_boxes[i];
				int cur_index = cur_box.n*l.w*l.h + cur_box.j*l.w + cur_box.i;
				if (cur_index == max_index)
					continue;
				int obj_index = focal_entry_index(l, b, cur_index, l.coords);
				delta_con(l.output, 0, obj_index, l.focal_alpha, l.focal_gamma, loss, l.delta, l.noobject_scale, l.con_loss_type);
				++j;
			}
			
		}
		free(neg_boxes);
	} else if (RCNN == l.strategy_type) {

	} else {}
	calc_tot_loss(loss, l);
	free(loss);
}

void backward_focal_layer(const layer l, network net) {}

void delta_con(float *output, int label, int index, float alpha, float gamma, float *loss, float *delta, int scale, LOSS_TYPE loss_type) {
	/// Note1: output[] have been activated
	float p = (label ? output[index] : (1 - output[index]));
	if (FL == loss_type) {
		loss[index] = scale * focal_loss(p, alpha, gamma);
		delta[index] = scale * delta_focal_loss(p, label, alpha, gamma);
	} else if (L2 == loss_type) {
		loss[index] = scale * l2_loss(label, output[index]);
		delta[index] = scale * delta_l2_loss(label, output[index]) * logistic_gradient(output[index]);
	} else if (CE == loss_type) {
		loss[index] = scale * crossentropy_loss(p);
		delta[index] = scale * delta_crossentropy_loss(p, label);
	} else {}
}

void delta_pos(float *output, box truth, float *biases, int n, int index, int i, int j, int w, int h, float *loss, float *delta, float scale, int stride, LOSS_TYPE loss_type, STRATEGY_TYPE strategy_type) {
	// select strategy type
	if (YOLO2 == strategy_type) {
		float tx = truth.x*w - i;
		float ty = truth.y*h - j;
		float tw = log(truth.w*w / biases[2 * n]);
		float th = log(truth.h*h / biases[2 * n + 1]);
		// select loss type
		if (L2 == loss_type) {
			loss[index + 0 * stride] = scale * l2_loss(tx, output[index + 0 * stride]);
			loss[index + 1 * stride] = scale * l2_loss(ty, output[index + 1 * stride]);
			loss[index + 2 * stride] = scale * l2_loss(tw, output[index + 2 * stride]);
			loss[index + 3 * stride] = scale * l2_loss(th, output[index + 3 * stride]);
			delta[index + 0 * stride] = scale * delta_l2_loss(tx, output[index + 0 * stride]) * logistic_gradient(output[index + 0 * stride]);
			delta[index + 1 * stride] = scale * delta_l2_loss(ty, output[index + 1 * stride]) * logistic_gradient(output[index + 1 * stride]);
			delta[index + 2 * stride] = scale * delta_l2_loss(tw, output[index + 2 * stride]);
			delta[index + 3 * stride] = scale * delta_l2_loss(th, output[index + 3 * stride]);
		} else if (SL1 == loss_type) {
			loss[index + 0 * stride] = scale * smooth_l1_loss(tx, output[index + 0 * stride]);
			loss[index + 1 * stride] = scale * smooth_l1_loss(ty, output[index + 1 * stride]);
			loss[index + 2 * stride] = scale * smooth_l1_loss(tw, output[index + 2 * stride]);
			loss[index + 3 * stride] = scale * smooth_l1_loss(th, output[index + 3 * stride]);
			delta[index + 0 * stride] = scale * delta_smooth_l1_loss(tx, output[index + 0 * stride]) * logistic_gradient(output[index + 0 * stride]);
			delta[index + 1 * stride] = scale * delta_smooth_l1_loss(ty, output[index + 1 * stride]) * logistic_gradient(output[index + 1 * stride]);
			delta[index + 2 * stride] = scale * delta_smooth_l1_loss(tw, output[index + 2 * stride]);
			delta[index + 3 * stride] = scale * delta_smooth_l1_loss(th, output[index + 3 * stride]);
		} else {}
	} else if (SSD == strategy_type) {
		float tx = (truth.x*w - (i + 0.5)) / biases[2 * n];
		float ty = (truth.y*h - (j + 0.5)) / biases[2 * n + 1];
		float tw = log(truth.w*w / biases[2 * n]);
		float th = log(truth.h*h / biases[2 * n + 1]);
		// select loss type
		if (L2 == loss_type) {
			loss[index + 0 * stride] = scale * l2_loss(tx, output[index + 0 * stride]);
			loss[index + 1 * stride] = scale * l2_loss(ty, output[index + 1 * stride]);
			loss[index + 2 * stride] = scale * l2_loss(tw, output[index + 2 * stride]);
			loss[index + 3 * stride] = scale * l2_loss(th, output[index + 3 * stride]);
			delta[index + 0 * stride] = scale * delta_l2_loss(tx, output[index + 0 * stride]);
			delta[index + 1 * stride] = scale * delta_l2_loss(ty, output[index + 1 * stride]);
			delta[index + 2 * stride] = scale * delta_l2_loss(tw, output[index + 2 * stride]);
			delta[index + 3 * stride] = scale * delta_l2_loss(th, output[index + 3 * stride]);
		} else if (SL1 == loss_type) {
			loss[index + 0 * stride] = scale * smooth_l1_loss(tx, output[index + 0 * stride]);
			loss[index + 1 * stride] = scale * smooth_l1_loss(ty, output[index + 1 * stride]);
			loss[index + 2 * stride] = scale * smooth_l1_loss(tw, output[index + 2 * stride]);
			loss[index + 3 * stride] = scale * smooth_l1_loss(th, output[index + 3 * stride]);
			delta[index + 0 * stride] = scale * delta_smooth_l1_loss(tx, output[index + 0 * stride]);
			delta[index + 1 * stride] = scale * delta_smooth_l1_loss(ty, output[index + 1 * stride]);
			delta[index + 2 * stride] = scale * delta_smooth_l1_loss(tw, output[index + 2 * stride]);
			delta[index + 3 * stride] = scale * delta_smooth_l1_loss(th, output[index + 3 * stride]);
		} else {}
	}
	else if (RCNN == strategy_type) {
	
	} else {}
}

void calc_tot_loss(float *loss, layer l) {
	int b, n, i;
	float cl = 0, pl = 0;
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			int index = focal_entry_index(l, b, n*l.w*l.h, 0);
			for (i = 0; i < l.coords*l.w*l.h; ++i)
				pl += *(loss + index + i);
			index = focal_entry_index(l, b, n*l.w*l.h, l.coords);
			for (i = 0; i < l.w*l.h; ++i)
				cl += *(loss + index + i);
		}
	}
	*(l.pos_cost) = pl;
	*(l.con_cost) = cl;
	*(l.cost) = pl + cl;
}

void get_focal_bboxes(network *net, int *pd_bboxes, int w, int h, int batch) {
	int b, i, j, n;
	for (b = 0; b < batch; ++b) {
		float max_conf = -1.0;
		box max_box = { 0 };
		for (j = 0; j < net->n; ++j) {
			layer l = net->layers[j];
			if (l.type != FOCAL || l.onlyforward)
				continue;
			for (i = 0; i < l.w*l.h; ++i) {
				int row = i / l.w;
				int col = i % l.w;
				for (n = 0; n < l.n; ++n) {
					int obj_index = focal_entry_index(l, b, n*l.w*l.h + i, l.coords);
					float cur_conf = l.output[obj_index];
					if (cur_conf > max_conf) {
						max_conf = cur_conf;
						int box_index = focal_entry_index(l, b, n*l.w*l.h + i, 0);
						max_box = get_focal_box(l.output, l.biases, l.mask[n], box_index, col, row, l.w, l.h, l.w*l.h, l.strategy_type);
					}
				}
			}
		}
		pd_bboxes[(b << 2) | 0] = (max_box.x - max_box.w / 2.0) * w;
		pd_bboxes[(b << 2) | 1] = (max_box.y - max_box.h / 2.0) * h;
		pd_bboxes[(b << 2) | 2] = (max_box.x + max_box.w / 2.0) * w;
		pd_bboxes[(b << 2) | 3] = (max_box.y + max_box.h / 2.0) * h;
		if (pd_bboxes[(b << 2) | 0] < 0) pd_bboxes[(b << 2) | 0] = 0;
		if (pd_bboxes[(b << 2) | 1] < 0) pd_bboxes[(b << 2) | 1] = 0;
		if (pd_bboxes[(b << 2) | 2] > w - 1) pd_bboxes[(b << 2) | 2] = w - 1;
		if (pd_bboxes[(b << 2) | 3] > h - 1) pd_bboxes[(b << 2) | 3] = h - 1;
	}
}



typedef struct box_batch_args {
	network *net;
	int netw, neth;
	int batch;
	int *pd_bboxes;
}box_batch_args;

void *get_boxes_batch(void *args) {
	box_batch_args* ptr = (box_batch_args*)args;
	network *net = ptr->net;
	int netw = ptr->netw, neth = ptr->neth, batch = ptr->batch;
	int *pd_bboxes = ptr->pd_bboxes;

	int i, j, n;
	float max_conf = -1.0;
	box max_box = { 0 };
	for (j = 0; j < net->n; ++j) {
		layer l = net->layers[j];
		if (l.type != FOCAL) continue;
		for (i = 0; i < l.w*l.h; ++i) {
			int row = i / l.w;
			int col = i % l.w;
			for (n = 0; n < l.n; ++n) {
				int obj_index = focal_entry_index(l, batch, n*l.w*l.h + i, l.coords);
				float cur_conf = l.output[obj_index];
				if (cur_conf > max_conf) {
					max_conf = cur_conf;
					int box_index = focal_entry_index(l, batch, n*l.w*l.h + i, 0);
					max_box = get_focal_box(l.output, l.biases, l.mask[n], box_index, col, row, l.w, l.h, l.w*l.h, l.strategy_type);
				}
			}
		}
	}
	pd_bboxes[0] = (max_box.x - max_box.w / 2.0) * netw;
	pd_bboxes[1] = (max_box.y - max_box.h / 2.0) * neth;
	pd_bboxes[2] = (max_box.x + max_box.w / 2.0) * netw;
	pd_bboxes[3] = (max_box.y + max_box.h / 2.0) * neth;
	if (pd_bboxes[0] < 0) pd_bboxes[0] = 0;
	if (pd_bboxes[1] < 0) pd_bboxes[1] = 0;
	if (pd_bboxes[2] > netw - 1) pd_bboxes[2] = netw - 1;
	if (pd_bboxes[3] > neth - 1) pd_bboxes[3] = neth - 1;
	return 0;
}

void get_focal_bboxes_parallel(network *net, int *pd_bboxes, int netw, int neth, int batch) {
	pthread_t *threads = (pthread_t*)calloc(batch, sizeof(pthread_t));
	box_batch_args *args = (box_batch_args*)calloc(batch, sizeof(box_batch_args));
	int i, offset = 0;
	for (i = 0; i < batch; ++i) {
		args[i].net = net;
		args[i].netw = netw;
		args[i].neth = neth;
		args[i].batch = i;
		args[i].pd_bboxes = pd_bboxes + offset;
		pthread_create(threads + i, 0, get_boxes_batch, args + i);
		offset += 4;
	}
	for (i = 0; i < batch; ++i) {
		pthread_join(threads[i], 0);
	}
	free(threads);
	free(args);
}



#ifdef GPU

void forward_focal_layer_gpu(const layer l, network net) {
	copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
	int b, n;
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			if (YOLO2 == l.strategy_type) {
				int index = focal_entry_index(l, b, n*l.w*l.h, 0);
				activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
			}
			int index = focal_entry_index(l, b, n*l.w*l.h, l.coords);
			activate_array_gpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
		}
	}
	if (!net.train || l.onlyforward) {
		cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
		return;
	}
	cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
	forward_focal_layer(l, net);
	if (!net.train) return;
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_focal_layer_gpu(const layer l, network net) {
	axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}

#endif