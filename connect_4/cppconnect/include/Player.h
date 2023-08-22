#ifndef C___CONNECT_4_PLAYER_H
#define C___CONNECT_4_PLAYER_H

#include "Config.h"
#include "Tree_search.h"
#include "Game_controller.h"
#include "Random.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <vector>
#include <array>
#include <tuple>
#include <string>

class Self_player {
public:
    template<class T, size_t... Is>
    using Array = nanobind::ndarray<nanobind::numpy, T, nanobind::shape<Is...>, nanobind::c_contig>;

    explicit Self_player(const std::string& directory) : tree{directory,true} {}

    void load_model(const std::string& directory) { tree.load_model(directory); }

    std::tuple<Array<int, nanobind::any, 2>, Array<float, nanobind::any, Config::width>, int> play_game()
    {
        Temp* const temp = new Temp{};
        std::vector<int>* const moves{&temp->move_history};
        std::vector<float>* const policies{&temp->policy_history};
        Game_controller game{};
        tree.game = &game;
        while (!game.is_over) {
            game.make_move(tree.choose_move(policies),moves);
        }
        tree.reset();

        nanobind::capsule deleter(temp,[](void* p) noexcept {
            delete (Temp*) p;
        });

        const size_t move_shape[2]{temp->move_history.size()/2,2};
        const size_t policy_shape[2]{temp->policy_history.size()/Config::width,Config::width};

        return {{moves->data(),2,move_shape,deleter},
                {policies->data(),2,policy_shape,deleter},
                game.win_state};
    }

private:
    struct Temp {
        Temp() { move_history.reserve(Config::max_moves*2); policy_history.reserve(Config::max_moves*Config::width); }

        std::vector<int> move_history;
        std::vector<float> policy_history;
    };

    Tree tree;
};

std::tuple<int, int> evaluate(const std::string& latest_directory, const std::string& best_directory)
{
    int wins{0};
    int draws{0};
    std::array<Tree, 2> trees{Tree{best_directory,false},Tree{latest_directory,false}};
    for (int _{0}; _<Config::num_eval_games; ++_) {
        Game_controller game{};
        trees[0].game = trees[1].game = &game;
        int current_player{Random::get(0,1)};
        while (!game.is_over) {
            const int move{trees[current_player].choose_move(nullptr,0)};
            current_player ^= 1;
            trees[current_player].opponent_move(move);
            game.make_move(move);
        }
        trees[0].reset();
        trees[1].reset();
        if (game.win_state == 0) {
            draws += 1;
        }
        else if (!current_player) {
            wins += 1;
        }
    }
    return {wins,draws};
}

#endif //C___CONNECT_4_PLAYER_H
