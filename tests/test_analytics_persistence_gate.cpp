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

TEST(persistent_supported_confirms_then_latches) {
    PersistenceGate g{PersistenceParams{}};
    int first = -1;
    for (int f = 1; f <= 5; f++) ASSERT_TRUE(g.update(one(100, 100, true)) == -1);
    first = g.update(one(100, 100, true));
    ASSERT_TRUE(first == 0);
    ASSERT_TRUE(g.locked());
    ASSERT_TRUE(g.update(one(101, 100, true)) == 0);
}

TEST(unsupported_never_confirms) {
    PersistenceGate g{PersistenceParams{}};
    for (int f = 1; f <= 30; f++) ASSERT_TRUE(g.update(one(100, 100, false)) == -1);
    ASSERT_TRUE(!g.locked());
}

TEST(flickering_support_never_confirms) {
    PersistenceGate g{PersistenceParams{}};
    for (int f = 1; f <= 40; f++)
        ASSERT_TRUE(g.update(one(100, 100, f % 2 == 0)) == -1);
    ASSERT_TRUE(!g.locked());
}

}

int main() {
    printf("== analytics/persistence_gate ==\n");
    return tests_failed > 0 ? 1 : 0;
}
