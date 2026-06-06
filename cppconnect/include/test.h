#ifndef C___CONNECT_4_TEST_H
#define C___CONNECT_4_TEST_H

#include "Config.h"
#include "Tree_search.h"
#include "Game_controller.h"
#include "Random.h"
#include <vector>
#include <array>
#include <tuple>
#include <string>

class Self_player {
public:
    explicit Self_player(const std::string& directory) : tree{directory, true} {}

    void load_model(const std::string& directory) { tree.load_model(directory); }

    std::tuple<std::vector<int>, std::vector<float>, int> play_game()
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

        const size_t move_shape[2]{temp->move_history.size()/2,2};
        const size_t policy_shape[2]{temp->policy_history.size()/Config::width,Config::width};

        return {*moves,
                *policies,
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

#endif //C___CONNECT_4_TEST_H
