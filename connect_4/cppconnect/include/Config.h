#ifndef C___CONNECT_4_CONFIG_H
#define C___CONNECT_4_CONFIG_H

#include <array>
#include <string>

namespace Config
{
    //Game specific settings
    inline constexpr int width{7};
    inline constexpr int height{6};
    inline constexpr int win_length{4};
    inline constexpr int max_moves{width*height};
    inline constexpr std::array<std::array<int, 2>, 4> dirs{{{1,0},{1,1},{0,1},{-1,1}}};

    //Self play
    inline constexpr int max_saved_games{30000};
    inline constexpr int num_high_temp_moves{5};
    inline constexpr int num_simulations{100};
    inline constexpr int num_threads{10};
//    inline constexpr std::string_view buffer_directory{"/Users/kevinnguyen/Documents/Python files/connect_4/data"};

    //Root prior exploration noise
    inline constexpr float alpha{0.1};
    inline constexpr float epsilon{0.25};

    //UCB formula
    inline constexpr float c_base{500};
    inline constexpr float c_init{1.05};

    //Evaluation
    inline constexpr int num_eval_games{100};
    inline constexpr float win_margin{0.5};

    //Training
    inline constexpr int batch_size{128};

    //Model settings
//    inline constexpr std::string_view model_directory{"/Users/kevinnguyen/Documents/Python files/connect_4/models"};
}

#endif //C___CONNECT_4_CONFIG_H
