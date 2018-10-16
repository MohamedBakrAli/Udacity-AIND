![Simulated Annealing](SA_animation.gif)
simulated annealing is a probablistic technique used for finding an approximate solution to an optimization problem--one of the simplest "gradient-free" optimization techniques. In this exercise you will check your understanding by implementing [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) to solve the [Traveling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem) (TSP) between US state capitals.  Briefly, the TSP is an optimization problem that seeks to find the shortest path passing through every city exactly once.  In our example the TSP path is defined to start and end in the same city (so the path is a closed loop).
Image Source: [Simulated Annealing - By Kingpin13 (Own work) [CC0], via Wikimedia Commons (Attribution not required)](https://commons.wikimedia.org/wiki/File:Hill_Climbing_with_Simulated_Annealing.gif)

## Code:
  0. Implement the `simulated_annealing()` main loop function in Section II
  0. Complete the `TravelingSalesmanProblem` class by implementing the `successors()` and `get_value()` methods in section III
  0. Complete the `schedule()` function to define the temperature schedule in Section IV
  0. Use the completed algorithm and problem description to experiment with simulated annealing to solve larger TSP instances on the map of US capitals
  ## Authors
* **Mohamed Bakr** [MohamedBakrAli](https://github.com/MohamedBakrAli)
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
