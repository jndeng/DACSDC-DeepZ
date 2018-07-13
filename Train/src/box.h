#ifndef BOX_H
#define BOX_H
#include "darknet.h"

#define MMAX(a,b) (((a) > (b)) ? (a) : (b))
#define MMIN(a,b) (((a) < (b)) ? (a) : (b))

typedef struct{
    float dx, dy, dw, dh;
} dbox;

float box_rmse(box a, box b);
dbox diou(box a, box b);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
