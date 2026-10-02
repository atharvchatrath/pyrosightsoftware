/* Minimal unit-test helpers: no dependencies, works on host and on target. */
#ifndef PS_TEST_H
#define PS_TEST_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int ps_test_failures;
static int ps_test_checks;

#define CHECK(cond) do { ps_test_checks++; if (!(cond)) { ps_test_failures++; \
    fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); } } while (0)
#define CHECK_NEAR(a, b, tol) do { ps_test_checks++; double _a = (a), _b = (b); \
    if (fabs(_a - _b) > (tol)) { ps_test_failures++; \
    fprintf(stderr, "%s:%d: CHECK_NEAR failed: %s = %g, expected %g +- %g\n", \
            __FILE__, __LINE__, #a, _a, _b, (double)(tol)); } } while (0)
#define RUN(fn) do { fn(); } while (0)
#define TEST_MAIN_END() do { printf("%s: %d checks, %d failures\n", __FILE__, ps_test_checks, \
    ps_test_failures); return ps_test_failures ? 1 : 0; } while (0)

/* Deterministic PRNG for tests (xorshift32) and a gaussian from it. */
static unsigned ps_test_rng = 2463534242u;
static inline float ps_test_uniform(void)
{
    ps_test_rng ^= ps_test_rng << 13; ps_test_rng ^= ps_test_rng >> 17; ps_test_rng ^= ps_test_rng << 5;
    return (ps_test_rng & 0xFFFFFF) / 16777216.0f;
}
static inline float ps_test_gauss(void)
{
    float u1 = ps_test_uniform() + 1e-7f, u2 = ps_test_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

#endif
