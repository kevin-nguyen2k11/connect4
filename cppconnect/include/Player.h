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
#include <nanobind/stl/array.h>
#include <vector>
#include <array>
#include <tuple>
#include <string>

class Self_player {
public:
    template<class T, size_t... Is>
    using Array = nanobind::ndarray<nanobind::numpy, T, nanobind::shape<Is...>, nanobind::c_contig>;

    explicit Self_player(std::string &&directory) : tree{std::move(directory), true} {}

    void load_model(std::string &&directory) { tree.load_model(std::move(directory)); }

    std::tuple<Array<int, nanobind::any, 2>, Array<float, nanobind::any, Config::width>, int> play_game()
    {
        Temp* const temp = new Temp{};
        std::vector<int>* const moves{&temp->move_history};
        std::vector<float>* const policies{&temp->policy_history};
        Game_controller game{};
        tree.game = &game;
        while (!tree.is_over()) {
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
                tree.game_result};
    }

private:
    struct Temp {
        std::vector<int> move_history;
        std::vector<float> policy_history;
    };

    Tree tree;
};

void evaluate(std::string& best_directory, std::string& latest_directory,
              std::atomic_int& played_games, std::atomic_int& wins, std::atomic_int& draws,
              std::array<std::atomic_int, 2>& lengths)
{
    std::array<Tree, 2> trees{Tree{std::move(best_directory),false},Tree{std::move(latest_directory),false}};
    while (true) {
        const int i{played_games.fetch_add(1,std::memory_order_relaxed)};
        if (i>=Config::num_eval_games) { break; }
        Game_controller game{};
        trees[0].game = trees[1].game = &game;
        int current_player{i%2};
        int length{0};
        while (!trees[0].is_over() && !trees[1].is_over()) {
            const int move{trees[current_player].choose_move(nullptr,0)};
            current_player ^= 1;
            trees[current_player].opponent_move(move);
            game.make_move(move,nullptr);
            length++;
        }
        trees[0].reset();
        trees[1].reset();

        lengths[i%2].fetch_add(length,std::memory_order_relaxed);
        const int result{trees[!current_player].game_result};
        if (result == 0) {
            draws.fetch_add(1,std::memory_order_relaxed);
            std::cout << "draw " << i%2 << std::endl;
        }
        else if (!current_player&&(result==1) || current_player&&(result==-1)) {
            wins.fetch_add(1,std::memory_order_relaxed);
            std::cout << "win " << i%2 << std::endl;
        }
        else {
            std::cout << "loss " << i%2 << std::endl;
        }
    }
}

std::tuple<int,int,int,int> evaluator(std::string&& best_directory, std::string&& latest_directory)
{
    std::atomic_int played_games{0};
    std::atomic_int wins{0};
    std::atomic_int draws{0};
    std::array<std::atomic_int, 2> lengths{{0,0}};
    std::array<std::thread, Config::num_eval_threads> threads;
    for (int i{0}; i<Config::num_eval_threads; ++i) {
        threads[i] = std::thread{evaluate,std::ref(best_directory),
                                 std::ref(latest_directory),std::ref(played_games),
                                 std::ref(wins),std::ref(draws),std::ref(lengths)};
    }
    for (auto& thread:threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    int win_count{wins.load(std::memory_order_relaxed)};
    int draw_count{draws.load(std::memory_order_relaxed)};
    std::cout << "wins: " << win_count << " losses: " << Config::num_eval_games-win_count-draw_count << std::endl;
    return {win_count,draw_count,lengths[0].load(std::memory_order_relaxed),lengths[1].load(std::memory_order_relaxed)};
}

std::array<int, Config::num_eval_games> evaluate_tournament(std::string& a, std::string& b)
{
    std::array<Tree, 2> trees{Tree{std::move(a),false},Tree{std::move(b),false}};
    std::array<int, Config::num_eval_games> results{};
    for (int i{0}; i<Config::num_eval_games; ++i) {
        Game_controller game{};
        trees[0].game = trees[1].game = &game;
        int current_player{i%2};
        while (!trees[0].is_over() && !trees[1].is_over()) {
            const int move{trees[current_player].choose_move(nullptr,0)};
            current_player ^= 1;
            trees[current_player].opponent_move(move);
            game.make_move(move,nullptr);
        }
        trees[0].reset();
        trees[1].reset();
        const int result{trees[!current_player].game_result};
        if (result == 0) {
            results[i] = 1;
        }
        else {
            results[i] = (!current_player == i%2) ? 2 : 0;
        }
    }
    return results;
}

//std::tuple<int, int> evaluate(const std::string& latest_directory, const std::string& best_directory)
//{
//    int wins{0};
//    int draws{0};
//    std::array<Tree, 2> trees{Tree{best_directory,false},Tree{latest_directory,false}};
//    for (int i{0}; i<Config::num_eval_games; ++i) {
//        Game_controller game{};
//        trees[0].game = trees[1].game = &game;
//        int current_player{i%2};
//        while (!game.is_over) {
//            const int move{trees[current_player].choose_move(nullptr,0)};
//            current_player ^= 1;
//            trees[current_player].opponent_move(move);
//            game.make_move(move);
//        }
//        trees[0].reset();
//        trees[1].reset();
//        if (game.win_state == 0) {
//            draws += 1;
//        }
//        else if (!current_player) {
//            wins += 1;
//        }
//    }
//    return {wins,draws};
//}

#endif //C___CONNECT_4_PLAYER_H
