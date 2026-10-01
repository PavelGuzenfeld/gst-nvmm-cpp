#ifndef NVMM_TEST_HARNESS_H
#define NVMM_TEST_HARNESS_H

#include <cstdio>
#include <stdexcept>

static int tests_passed = 0;
static int tests_failed = 0;

/// Runs from a static constructor, before main(). ASSERT_* throw rather than
/// return, so a failed assert can never fall through to the PASS count.
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

#define ASSERT_NEAR(a, b, eps) \
    ASSERT_TRUE(std::fabs((double)(a) - (double)(b)) <= (eps))

#endif
