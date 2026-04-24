#pragma once
#include <vector>

typedef struct anchor {
  float x, y, w, h;
} anchor;

typedef struct point{
  float x,y;
}point;
typedef struct box {
  float x1, y1, x2, y2;
  std::vector<point> circles;
  float score;
} box;

extern std::vector<anchor> anc;

