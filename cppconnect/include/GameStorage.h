#ifndef C___CONNECT_4_GAMESTORAGE_H
#define C___CONNECT_4_GAMESTORAGE_H

#include "Config.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <array>
#include <vector>

class GameStorage {
public:
    template<class T, size_t... Is>
    using Array = nanobind::ndarray<nanobind::numpy, T, nanobind::shape<Is...>, nanobind::c_contig>;

    void save_game(Array<int, 2, nanobind::any>);

    void get_batch(Array<int, Config::batch_size, Config::height, Config::width, 2>,
                   Array<float, Config::batch_size, Config::width>,
                   Array<int, Config::batch_size, 1>) const;

private:

    std::array<Array<int, 2, nanobind::any>, Config::max_saved_games> moves{};
    std::array<Array<float, nanobind::any, Config::width>, Config::max_saved_games> policies{};
};

#endif //C___CONNECT_4_GAMESTORAGE_H
