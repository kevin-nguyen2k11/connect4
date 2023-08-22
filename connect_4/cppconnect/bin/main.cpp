#include "Ndarray.h"
#include "Random.h"
#include "Player.h"
#include "Game_controller.h"
#include <random>
#include <iostream>
#include <unistd.h>
#include <atomic>
#include <thread>
#include <functional>
#include <cmath>
#include <cppflow/cppflow.h>
//#include <nanobind/nanobind.h>
#include <memory>
#include <vector>
#include <array>
#include <random>
#include <limits>
#include <filesystem>
#include <string>

//void producer(int& input, int& results, std::atomic_int& count, std::atomic_bool& ready, std::atomic_bool& input_ready)
//{
//    std::cout << "producing\n";
//    sleep(2);
//    input = 10;
//    int prev = count.fetch_add(1, std::memory_order_release);
//    if (prev==4) {
//        input_ready.store(true, std::memory_order_relaxed);
//        input_ready.notify_one();
//    }
//    ready.wait(false, std::memory_order_acquire);
//    std::cout << "got results: " << results << '\n';
//
//}
//
//void consumer(std::array<int, 5>& input, std::array<int, 5>& results, std::atomic_int& count, std::atomic_bool& ready, std::atomic_bool& input_ready)
//{
//    input_ready.wait(false, std::memory_order_acquire);
//    std::cout << "done waiting: ";
//    for (const auto& i:input) {
//        std::cout << i << " ";
//    }
//    std::cout << '\n';
//    results = {1,2,3,4,5};
//    ready.store(true, std::memory_order_release);
//    ready.notify_all();
//    std::cout << "distributed\n";
//}

//template<class T, size_t... Is>
//using Array = nanobind::ndarray<nanobind::numpy, T, nanobind::shape<Is...>, nanobind::c_contig>;

int main()
{
//    std::vector<int> test{0,1,2,3,4,5,6,7};
//    size_t move_shape[2]{2,2};
//    Array<int,2,2> thing{test.data(),2,move_shape};
//    std::cout << thing;

//    Ndarray<float> test{{2,2}};
//    for (int i{0}; i<2; ++i) {
//        for (int j{0}; j<2; ++j) {
//            test[{i,j}] = 1;
//        }
//    }
//    std::cout << test[{1,1}];

//    const std::string thing{"/Users/kevinnguyen/Documents/Python files/connect_4/models/best"};
//    Self_player player{thing};
//    auto outputs = player.play_game(true);

    Game_controller game{};
    while (!game.is_over) {
        int move;
        std::cout << "Please make a move: ";
        std::cin >> move;
        game.make_move(move,nullptr);
    }

//    std::vector<cppflow::tensor> inputs = {10, cppflow::fill({2,2}, 0.0)};
//    for(auto& elem:inputs) {
//        std::cout << elem << '\n';
//    }


//    auto input = cppflow::fill({10, 6, 7, 2}, 1.0f);
//    cppflow::model model{"/Users/kevinnguyen/Documents/Python files/connect_4/models/best"};
//
//    std::vector<cppflow::tensor> output = model({{"serving_default_board:0", input}},
//                        {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"});
//
//    std::cout << "output_1: " << output[0] << std::endl;
//    std::vector<float> tho{output[1].get_data<float>()};
//    for(auto& elem:tho) {
//        std::cout << "output_2: " << elem << std::endl;
//    }

//    Ndarray<int> thing{{2,4}};
//    thing[{1,3}] = 1;
//    thing.print();

//    std::vector<int> data1 { 0, 2, 3, 4, 5, 6, 7, 8 };
//    std::vector<int64_t> shape {2,4};
//    cppflow::tensor thing{data1, shape};
//    std::cout << thing;


//    std::array<int, 5> input{0,0,0,0,0};
//    std::array<int, 5> results{0,0,0,0,0};
//    std::atomic_int count{0};
//    std::atomic_bool ready;
//    std::atomic_bool input_ready;
//    std::thread consumer_thread{consumer, std::ref(input), std::ref(results), std::ref(count), std::ref(ready), std::ref(input_ready)};
//    std::array<std::thread, 5> threads{};
//    for (int i=0; i<5; ++i) {
//        threads[i] = std::thread{producer, std::ref(input[i]), std::ref(results[i]), std::ref(count), std::ref(ready), std::ref(input_ready)};
//    }
//    consumer_thread.join();
//    for (auto& thread:threads) {
//        thread.join();
//    }

}