/// Unit test for analytics/persistence_gate.hpp (pure C++, no OpenCV).
#include "persistence_gate.hpp"
#include "test_harness.h"

#include <vector>

namespace {

using nvmm::track::Detection;
using nvmm::track::PersistenceGate;
using nvmm::track::PersistenceParams;

std::vector<Detection> one(float x, float y, bool supported) {
    return { Detection{x, y, 0.9f, supported} };
}

// persistent + supported every frame -> confirms once age>=min_age AND support>=min_support.
TEST(persistent_supported_confirms_then_latches) {
    PersistenceGate g{PersistenceParams{}};   // min_age=6, min_support=4
    int first = -1;
    for (int f = 1; f <= 5; f++) ASSERT_TRUE(g.update(one(100, 100, true)) == -1);  // not yet
    first = g.update(one(100, 100, true));     // frame 6: age=6, support=6 -> confirm
    ASSERT_TRUE(first == 0);
    ASSERT_TRUE(g.locked());
    ASSERT_TRUE(g.update(one(101, 100, true)) == 0);   // stays locked, tracks the detection
}

// persistent but never supported -> never confirms.
TEST(unsupported_never_confirms) {
    PersistenceGate g{PersistenceParams{}};
    for (int f = 1; f <= 30; f++) ASSERT_TRUE(g.update(one(100, 100, false)) == -1);
    ASSERT_TRUE(!g.locked());
}

// support that flickers (never min_support in a row) -> never confirms.
TEST(flickering_support_never_confirms) {
    PersistenceGate g{PersistenceParams{}};
    for (int f = 1; f <= 40; f++)
        ASSERT_TRUE(g.update(one(100, 100, f % 2 == 0)) == -1);  // supported every other frame
    ASSERT_TRUE(!g.locked());
}

}  // namespace

int main() {
    printf("== analytics/persistence_gate ==\n");
    return tests_failed > 0 ? 1 : 0;
}
