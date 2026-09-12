/// Unit tests for KalmanBox (kalman_box.hpp): the 8D constant-velocity filter.
/// Reference values generated from the SAMURAI Python KalmanFilter
/// (scripts/kf_ref.py) — these MUST match to validate the C++ port.

#include "kalman_box.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

#include "test_harness.h"

namespace {

static void check_mean(const nvmm::KalmanBox &kf, const double e[8], const char *tag)
{
    const auto &m = kf.mean();
    for (int i = 0; i < 8; ++i) {
        if (std::fabs(m[i] - e[i]) > 1e-4) {
            printf("FAIL %s mean[%d]: got %.8g want %.8g\n", tag, i, m[i], e[i]);
            tests_failed++;
            return;
        }
    }
    tests_passed++;
    printf("  %s mean ... PASS\n", tag);
}

static void check_cov_diag(const nvmm::KalmanBox &kf, const double e[8], const char *tag)
{
    const auto &c = kf.covariance();
    for (int i = 0; i < 8; ++i) {
        if (std::fabs(c[i][i] - e[i]) > 1e-4) {
            printf("FAIL %s cov_diag[%d]: got %.8g want %.8g\n", tag, i, c[i][i], e[i]);
            tests_failed++;
            return;
        }
    }
    tests_passed++;
    printf("  %s cov_diag ... PASS\n", tag);
}

TEST(matches_python_reference) {
    nvmm::KalmanBox kf;

    kf.initiate(100, 200, 14, 8);
    const double init_mean[8] = {100, 200, 14, 8, 0, 0, 0, 0};
    const double init_cov[8]  = {1.96, 0.64, 1.96, 0.64, 0.765625, 0.25, 0.765625, 0.25};
    check_mean(kf, init_mean, "INIT");
    check_cov_diag(kf, init_cov, "INIT");

    kf.predict(1.0);
    const double p1_mean[8] = {100, 200, 14, 8, 0, 0, 0, 0};
    const double p1_cov[8]  = {3.215625, 1.05, 3.215625, 1.05, 0.77328125, 0.2525, 0.77328125, 0.2525};
    check_mean(kf, p1_mean, "PRED1");
    check_cov_diag(kf, p1_cov, "PRED1");

    kf.update(110, 205, 15, 9);
    const double u_mean[8] = {108.677686, 204.338843, 14.8677686, 8.867768595,
                              2.066115702, 1.033057851, 0.2066115702, 0.2066115702};
    const double u_cov[8]  = {0.4252066116, 0.1388429752, 0.4252066116, 0.1388429752,
                              0.6150942665, 0.2008471074, 0.6150942665, 0.2008471074};
    check_mean(kf, u_mean, "UPD");
    check_cov_diag(kf, u_cov, "UPD");

    kf.predict(2.0);
    const double p2_mean[8] = {112.8099174, 206.4049587, 15.28099174, 9.280991736,
                               2.066115702, 1.033057851, 0.2066115702, 0.2066115702};
    const double p2_cov[8]  = {3.843168713, 1.27105611, 3.843168713, 1.27105611,
                               0.6237290534, 0.2039188777, 0.6237290534, 0.2039188777};
    check_mean(kf, p2_mean, "PRED2");
    check_cov_diag(kf, p2_cov, "PRED2");

    double gd = kf.gating_distance(130, 212, 16, 9);
    ASSERT_NEAR(gd, 87.98068236, 1e-3);
}

// Each coordinate (cx, cy, w, h) carries its own independent position/velocity
// pair, so a covariance term between two different coordinates is structurally
// zero, not merely small — hence `== 0.0` and not a tolerance. chol4's
// off-diagonal branches and the accumulation loops in both triangular solves are
// unreachable for as long as this holds, which is why mutating them changes no
// observable value. Should this ever go red, those branches become live and need
// tests of their own.
TEST(covariance_never_correlates_two_coordinates) {
    auto no_cross_terms = [](const nvmm::KalmanBox &f) {
        const auto &c = f.covariance();
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j)
                if (i % 4 != j % 4) ASSERT_TRUE(c[i][j] == 0.0);
    };
    nvmm::KalmanBox kf;
    kf.initiate(100, 200, 14, 8); no_cross_terms(kf);
    kf.predict(1.0);              no_cross_terms(kf);
    kf.update(110, 205, 15, 9);   no_cross_terms(kf);
    kf.predict(2.0);              no_cross_terms(kf);
    kf.update(130, 215, 17, 11);  no_cross_terms(kf);
}

// gating_distance runs the residual through chol4 and a forward solve. The
// covariance is diagonal (see the test above), so the answer must equal the
// closed form sum d_i^2 / S_ii, S = cov[0..3][0..3] + diag(r^2) — an oracle that
// shares no code with the factorisation it checks. The state is driven until a
// pivot falls below 1: chol4 guards the pivot against being *negative*, and
// nothing else in the suite reaches a state where a guard against "small"
// instead would behave differently.
TEST(gating_distance_matches_the_closed_form_at_a_pivot_below_one) {
    nvmm::KalmanBox kf;
    kf.initiate(100, 200, 14, 8);
    kf.predict(1.0);
    kf.update(110, 205, 15, 9);
    kf.predict(2.0);
    kf.update(130, 215, 17, 11);
    kf.predict(0.5);

    const double meas[4] = {131, 216, 17, 11};
    const auto &m = kf.mean();
    const auto &c = kf.covariance();
    const double sp = 1.0 / 20.0;   // kStdWPos, kalman_box.hpp
    const double r[4] = {sp * m[2], sp * m[3], sp * m[2], sp * m[3]};

    double expect = 0.0, smallest = c[0][0] + r[0] * r[0];
    for (int i = 0; i < 4; ++i) {
        const double s = c[i][i] + r[i] * r[i];
        const double d = meas[i] - m[i];
        expect += d * d / s;
        smallest = std::fmin(smallest, s);
    }
    ASSERT_TRUE(smallest <= 1.0);

    // Both sides are ~10 operations on values of order 10, so the two orderings
    // may disagree by a few ulp of the result; 16 is the budget, measured worst
    // case is under 1.
    const double budget = 16 * std::numeric_limits<double>::epsilon() * expect;
    ASSERT_NEAR(kf.gating_distance(meas[0], meas[1], meas[2], meas[3]), expect, budget);
}

TEST(box_accessor_center_form) {
    nvmm::KalmanBox kf;
    kf.initiate(50, 60, 10, 6);
    double cx, cy, w, h; kf.box(cx, cy, w, h);
    ASSERT_TRUE(cx == 50 && cy == 60 && w == 10 && h == 6);
    ASSERT_TRUE(kf.initiated());
}

}  // namespace

int main() {
    printf("=== KalmanBox Tests (vs SAMURAI Python reference) ===\n");
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
