#ifndef C___CONNECT_4_TREE_SEARCH_H
#define C___CONNECT_4_TREE_SEARCH_H

#include "Config.h"
#include "Game_controller.h"
#include <cppflow/cppflow.h>
#include <vector>
#include <memory>
#include <atomic>
#include <array>
#include <thread>

class Node {
public:
    template<class T>
    using Array = std::array<T, Config::width>;

    Node() = default;
    Node(int m, Node* p) : move{m}, parent{p} {}

    void create_children(const Ndarray<float>& priors, int thread, float value, std::vector<bool>&& legals);

    inline void increment_visit(int n = 1);

    inline void update(float value);

    [[nodiscard]]
    int choose_child() const;

    [[nodiscard]]
    inline int choose_move(int temperature) const;

    [[nodiscard]]
    Array<float> get_policy() const;

    inline void add_noise();

    bool expanded{false};
    bool terminal{false};
    int win_state{0};
    Node* parent{nullptr}; // was unique_ptr, but just needs to be ptr right?
    Array<std::unique_ptr<Node>> children;
    std::atomic_bool flag{false};

private:
    [[nodiscard]]
    Array<float> get_mean_action() const;

    [[nodiscard]]
    Array<float> get_appraisal() const;

    [[nodiscard]]
    inline int high_temp_move() const;

    [[nodiscard]]
    int low_temp_move() const;

    Array<int> child_visits{};
    Array<float> child_priors;
    Array<float> child_total_actions{};
    std::vector<bool> legal_moves;
    const int move{};
};

class Player {
public:
    virtual int choose_move(std::vector<float>*, int) {};

    virtual bool is_over() const {};

    virtual void opponent_move(int) {};

    Game_controller* game;
};

class Tree: public Player {
public:
    Tree(std::string&& directory, bool n) : model{new cppflow::model{std::move(directory)}}, noise{n}
    {
        evaluator = std::thread(&Tree::evaluate,this);
        for (int i{0}; i<Config::num_threads; ++i) {
            searchers[i] = std::thread{&Tree::search,this,i};
        }
    }

    Tree(const Tree&) = delete;

    Tree& operator=(const Tree&) = delete;

    Tree(Tree&&) = delete;

    Tree& operator=(Tree&&) = delete;

    ~Tree();

    void load_model(std::string&& directory) { delete model; model = new cppflow::model{std::move(directory)}; }

    void reset() { game_result = root->win_state; root = std::make_unique<Node>(); }

    int choose_move(std::vector<float>* policy_history, int temperature = 1) override;

    bool is_over() const override { return root->terminal; }

    void opponent_move(int move) override;

    const Game_controller* game;
    int game_result;

private:
    void evaluate();

    void search(int thread_id);

    Ndarray<float> inputs{{Config::num_threads,Config::height,Config::width,2}};
    Ndarray<float> output_policies{{Config::width,Config::num_threads}};
    std::vector<float> output_values{};
    std::atomic_int ready_count{0};
    std::atomic_bool input_ready{false};
    std::atomic_bool output_ready{true};
    std::atomic_bool done{false};
    cppflow::model* model;
    std::unique_ptr<Node> root{std::make_unique<Node>()};
    std::array<std::thread, Config::num_threads> searchers;
    std::thread evaluator;
    const bool noise{};
    std::atomic_bool exit{false};
};

#endif //C___CONNECT_4_TREE_SEARCH_H
