## An implementation of DeepMind's AlphaGo Zero strategy to play Connect 4

Silver, D., Schrittwieser, J., Simonyan, K. et al. [Mastering the game of Go without human knowledge](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf). Nature 550, 354–359 (2017). https://doi.org/10.1038/nature24270

This project was undertaken with the goal of better understanding the concepts put forth in the above paper by attempting to generalize it to a different game. The result is an AI that learns how to play the board game Connect 4 from scratch, without any prior human knowledge (heuristics, expert moves etc.)

- Python multiprocessing for concurrent training, self-play, and evaluation
- Multithreaded C++ code for computationally intensive Monte Carlo tree search
- Convolutional neural network built with TensorFlow

### Why Connect 4?

Similarly to Go, Connect 4 is a two-player, adversarial, perfect information game. However, it is much simpler; the total board size is 7x6 compared to a full-sized 19x19 Go board, with a maximum of only 7 possible moves in any given state. In addition, all relevant information is encoded in the current board state (the order of previous moves does not matter), unlike Go where removing pieces is possible and repetition of previous states is prohibited (ko rule). This simplicity means Connect 4 was [weakly solved as early as 1988](https://tromp.github.io/c4/connect4_thesis.pdf) using knowledge-based approaches, and [strongly solved in 1995](https://tromp.github.io/c4/c4.html) for legal 8-ply positions. Nevertheless, it is not so simple as to be trivial; there are about [1.6e13 legal positions](https://web.mit.edu/sp.268/www/2010/connectFourSlides.pdf), and a [strong solution for every board state](https://github.com/markus7800/Connect4-Strong-Solver) was only achieved in 2025 and takes ~47 hours and 128 GB RAM. This was a good level of complexity for this project, which ultimately achieved strong play in around a day running on one laptop.

## Training your own model
### Requirements / dependencies

## Comparing different hyperparameters

## Demonstration

## Possible extensions
