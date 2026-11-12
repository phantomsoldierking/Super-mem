/* Sanity workload: STREAM-like triad over 64 MiB arrays.
 * Expect ~1 L2 miss per 64 B line per array and dirty write-backs for `a`. */
#include <stdio.h>
#include <stdlib.h>

#define N (8u << 20)

int main(void) {
  double *a = malloc(N * sizeof *a), *b = malloc(N * sizeof *b), *c = malloc(N * sizeof *c);
  for (unsigned i = 0; i < N; i++) { b[i] = i; c[i] = 2.0 * i; }
  for (int rep = 0; rep < 4; rep++)
    for (unsigned i = 0; i < N; i++) a[i] = b[i] + 3.0 * c[i];
  printf("%f\n", a[N / 2]);
  return 0;
}
