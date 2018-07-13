#include "upsample_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

/* Only support GPU version */

layer make_upsample_layer(int batch, int w, int h, int c, int out_w, int out_h) {
    layer l = {0};
    l.type = UPSAMPLE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = out_w;
    l.out_h = out_h;
    l.out_c = c;

    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;
    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;

    l.forward = forward_upsample_layer;
    l.backward = backward_upsample_layer;
    #ifdef GPU
    l.forward_gpu = forward_upsample_layer_gpu;
    l.backward_gpu = backward_upsample_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    fprintf(stderr, "upsample                %4d x%4d x%4d   ->  %4d x%4d x%4d\n", w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_upsample_layer(layer *l, int w, int h) {}

void forward_upsample_layer(const layer l, network net) {}

void backward_upsample_layer(const layer l, network net) {}

#ifdef GPU
void forward_upsample_layer_gpu(const layer l, network net)
{
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    upsample_gpu(l.batch, l.c, 1, net.input_gpu, l.w, l.h, l.output_gpu, l.out_w, l.out_h);
/*
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if(l.reverse){
        upsample_gpu(l.output_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input_gpu);
    }else{
        upsample_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_gpu);
    }
*/
}

void backward_upsample_layer_gpu(const layer l, network net)
{
    upsample_gpu(l.batch, l.c, 0, net.delta_gpu, l.w, l.h, l.delta_gpu, l.out_w, l.out_h);
    
/*
    if(l.reverse){
        upsample_gpu(l.delta_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, net.delta_gpu);
    }else{
        upsample_gpu(net.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta_gpu);
    }
*/
}
#endif
