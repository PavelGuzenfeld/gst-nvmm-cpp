#ifndef NVMM_TEST_HARNESS_H
#define NVMM_TEST_HARNESS_H

// Minimal self-registering unit-test harness.
//
// Define a test with `TEST(name) { ... ASSERT_*; }` — it registers itself via a
// static constructor and runs before main(). Provide your own main() that prints
// a title and ends with `return tests_failed > 0 ? 1 : 0;`.
//
// ASSERT_* THROW on failure (they do not `return`). This is load-bearing: the
// TEST wrapper counts a pass only if test_##name() returns without throwing, so a
// failing assert must unwind into the catch. A *returning* assert would fall
// through to the PASS path and double-count — printing both "FAIL" and "PASS" and
// bumping both counters (the bug this shared harness exists to prevent).
//
// For the alternative explicit-`PASS()` style (test fns call PASS() at their end,
// asserts `return`), see the RUN_TEST harness inlined in tests that use it; that
// style is correct because the early `return` skips the trailing PASS().

#include <cstdio>
#include <stdexcept>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    struct test_reg_##name { test_reg_##name() { \
        printf("  TEST %s ... ", #name); \
        try { test_##name(); printf("PASS\n"); tests_passed++; } \
        catch (...) { printf("FAIL (exception)\n"); tests_failed++; } \
    } } test_reg_inst_##name; \
    static void test_##name()

#define ASSERT_TRUE(expr) do { \
    if (!(expr)) { printf("FAIL at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
                    throw std::runtime_error("assertion failed"); } } while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { printf("FAIL at %s:%d: %s != %s\n", __FILE__, __LINE__, #a, #b); \
                       throw std::runtime_error("assertion failed"); } } while(0)

#define ASSERT_NOT_NULL(ptr) ASSERT_TRUE((ptr) != nullptr)

/* Float comparison within eps (test file must include <cmath>). */
#define ASSERT_NEAR(a, b, eps) \
    ASSERT_TRUE(std::fabs((double)(a) - (double)(b)) <= (eps))

#endif  // NVMM_TEST_HARNESS_H
