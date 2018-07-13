#ifndef FOCAL_LAYER_H
#define FOCAL_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_focal_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int coords, int max_boxes, char *loss_type_s, char *con_loss_type_s, char *strategy_type_s);
void forward_focal_layer(const layer l, network net);
void backward_focal_layer(const layer l, network net);

box get_focal_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride, STRATEGY_TYPE strategy_type);
int focal_entry_index(layer l, int batch, int location, int entry);

// added
void delta_con(float *output, int label, int index, float alpha, float gamma, float *loss, float *delta, int scale, LOSS_TYPE loss_type);
void delta_pos(float *output, box truth, float *biases, int n, int index, int i, int j, int w, int h, float *loss, float *delta, float scale, int stride, LOSS_TYPE loss_type, STRATEGY_TYPE strategy_type);
void calc_tot_loss(float *loss, layer l);


#ifdef GPU
void forward_focal_layer_gpu(const layer l, network net);
void backward_focal_layer_gpu(layer l, network net);
#endif

#endif
