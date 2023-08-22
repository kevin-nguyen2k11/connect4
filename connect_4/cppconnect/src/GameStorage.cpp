#include "GameStorage.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <iostream>
#include <tuple>
#include <array>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>

//int test[5] = {1,2,3,4,5};
//
//int thing()
//{
//    namespace nb = nanobind;
//    size_t shape[1] = {5};
//    return nb::ndarray<nb::numpy, int, nb::shape<nb::any>> data(test, 1, shape);
//}
//
//int add(int a, int b) { return a + b; }
//
//NB_MODULE(my_ext, m) {
//    m.def("add", &add);
//}
//
//std::vector<std::vector<int>> data {{ 0, 2, 3, 4 }, { 5, 6, 7, 8 }};
//std::vector<int> data { 0, 2, 3, 4, 5, 6, 7, 8 };
//auto data = new std::vector<int>{ 0, 2, 3, 4, 5, 6, 7, 8 };
//
//
//template<class T>
//nanobind::capsule create_owner(T* a)
//{
//    return nanobind::capsule{a, [](void* p) noexcept {
//        delete (std::vector<T>*) p;
//    }};
//}

struct Temp {
    std::vector<int> vec_1;
    std::vector<int> vec_2;
};

template<class T, size_t... Is>
using Array = nanobind::ndarray<nanobind::numpy, T, nanobind::shape<Is...>, nanobind::c_contig>;

std::tuple<Array<int,2,nanobind::any>,Array<int,2,nanobind::any>> test()
{
    namespace nb = nanobind;
    size_t shape[2] = { 2, 4 };

    Temp* temp = new Temp();

    std::vector<int> data1 { 0, 2, 3, 4, 5, 6, 7, 8 };
    std::vector<int> data2 { 0, 2, 3, 4, 5, 6, 7, 8 };

    temp->vec_1 = std::move(data1);
    temp->vec_2 = std::move(data2);

    nb::capsule deleter(temp, [](void *p) noexcept {
        delete (Temp *) p;
    });

    return {{temp->vec_1.data(), 2, shape, deleter},
            {temp->vec_2.data(), 2, shape, deleter}};
}

NB_MODULE(player, m) {
    m.def("test", &test);
}

//NB_MODULE(player, m) {
//    namespace nb = nanobind;
//    m.def("ret_numpy", []() {
//        size_t shape[2] = { 2, 4 };
//
//        Temp* temp = new Temp();
//
//        std::vector<int> data1 { 0, 2, 3, 4, 5, 6, 7, 8 };
//        std::vector<int> data2 { 0, 2, 3, 4, 5, 6, 7, 8 };
//
//        temp->vec_1 = std::move(data1);
//        temp->vec_2 = std::move(data2);
//
//        nb::capsule deleter(temp, [](void *p) noexcept {
//            delete (Temp *) p;
//        });
//
////        nb::ndarray<nb::numpy, int, nb::shape<2, nb::any>> thing{data->data(), 2, shape, owner};
//
//        return std::make_pair(
//                nb::ndarray<nb::numpy, int>{temp->vec_1.data(), 2, shape, deleter},
//                nb::ndarray<nb::numpy, int>{temp->vec_2.data(), 2, shape, deleter});
//    });
//}
