#ifndef C___CONNECT_4_PLAYER_H
#define C___CONNECT_4_PLAYER_H

#include "Config.h"
#include "Tree_search.h"
#include "Game_controller.h"
#include "Random.h"
#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <atomic>
#include <thread>

class Self_player {
public:
    explicit Self_player(std::string&& directory) : tree{std::move(directory),true} {}

    void load_model(std::string&& directory) { tree.load_model(std::move(directory)); }

    std::tuple<std::vector<int>, std::vector<float>, int> play_game()
    {
        Game_controller game{};
        std::vector<int> move_history;
        std::vector<float> policy_history;
        tree.game = &game;
        while (!tree.is_over()) {
            game.make_move(tree.choose_move(&policy_history),&move_history);
        }
        tree.reset();
        return {std::move(move_history),std::move(policy_history),tree.game_result};
    }

private:

    Tree tree;
};

void evaluate(std::string& best_directory, std::string& latest_directory,
              std::atomic_int& played_games, std::atomic_int& wins, std::atomic_int& draws)
{
    std::array<Tree, 2> trees{Tree{std::move(best_directory),false},Tree{std::move(latest_directory),false}};
    int i{0};
    while (played_games.fetch_add(1,std::memory_order_release)<Config::num_eval_games) {
        Game_controller game{};
        trees[0].game = trees[1].game = &game;
        int current_player{i%2};
        while (!trees[0].is_over() && !trees[1].is_over()) {
            const int move{trees[current_player].choose_move(nullptr,0)};
            current_player ^= 1;
            trees[current_player].opponent_move(move);
            game.make_move(move, nullptr);
        }
        trees[0].reset();
        trees[1].reset();
        const int result{trees[!current_player].game_result};
        if (result == 0) {
            draws.fetch_add(1,std::memory_order_release);
            std::cout << "draw" << std::endl;
        }
        else if (!current_player&&(result==1) || current_player&&(result==-1)) {
            wins.fetch_add(1,std::memory_order_release);
            std::cout << "win" << std::endl;
        }
        else {
            std::cout << "loss" << std::endl;
        }
        ++i;
    }
}

std::tuple<int,int> evaluator(std::string&& best_directory, std::string&& latest_directory)
{
    std::atomic_int played_games{0};
    std::atomic_int wins{0};
    std::atomic_int draws{0};
    std::array<std::thread, Config::num_threads> threads;
    for (int i{0}; i<Config::num_eval_threads; ++i) {
        threads[i] = std::thread{evaluate,std::ref(best_directory),
                                 std::ref(latest_directory),std::ref(played_games),
                                 std::ref(wins),std::ref(draws)};
    }
    for (auto& thread:threads) {
        thread.join();
    }
    std::cout << "win, losses: " << wins << " " << Config::num_eval_games-wins-draws << std::endl;
    return {wins,draws};
}

#endif //C___CONNECT_4_PLAYER_H