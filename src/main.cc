#include "helloWindow.h"
#include <matplot/matplot.h>

void test02() {
    std::vector<std::vector<int>> data = {
        {45, 60, 32}, {43, 54, 76}, {32, 94, 68}, {23, 95, 58}};
    matplot::heatmap(data);

    matplot::show();
}

int main(int argc, char const *argv[]) {
    // test01(); // Hello Window
    test02(); // Matplot++
    return 0;
}