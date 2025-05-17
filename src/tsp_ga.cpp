#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <limits>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "tsp_ga.hpp"

using namespace std;

// Random number generator
random_device rd;
mt19937 gen(rd());

// Generate a set of random cities
vector<City> generateCities(size_t numCities) {
    vector<City> cities;
    uniform_real_distribution<> dis(0.0, 100.0);
    
    for (size_t i = 0; i < numCities; ++i) {
        City city;
        city.id = i;
        city.x = dis(gen);
        city.y = dis(gen);
        cities.push_back(city);
    }
    
    return cities;
}

// Initialize a random population of routes
vector<Route> initializePopulation(const vector<City>& cities, size_t populationSize) {
    vector<Route> population;
    
    for (size_t i = 0; i < populationSize; ++i) {
        Route route;
        route.path.resize(cities.size());
        
        // Create a path visiting all cities in random order
        for (size_t j = 0; j < cities.size(); ++j) {
            route.path[j] = j;
        }
        
        // Shuffle the path (except for the starting city if needed)
        shuffle(route.path.begin(), route.path.end(), gen);
        
        population.push_back(route);
    }
    
    return population;
}

// Calculate the total distance of a route
double calculateDistance(const vector<City>& cities, const vector<int>& path) {
    double totalDistance = 0.0;
    
    for (size_t i = 0; i < path.size() - 1; ++i) {
        int cityA = path[i];
        int cityB = path[i + 1];
        
        double dx = cities[cityA].x - cities[cityB].x;
        double dy = cities[cityA].y - cities[cityB].y;
        totalDistance += sqrt(dx * dx + dy * dy);
    }
    
    // Add distance from last city back to first city
    int firstCity = path.front();
    int lastCity = path.back();
    double dx = cities[lastCity].x - cities[firstCity].x;
    double dy = cities[lastCity].y - cities[firstCity].y;
    totalDistance += sqrt(dx * dx + dy * dy);
    
    return totalDistance;
}

// Evaluate fitness for all routes in the population
void evaluatePopulation(vector<Route>& population, const vector<City>& cities) {
    for (auto& route : population) {
        route.distance = calculateDistance(cities, route.path);
        // Fitness is inverse of distance (shorter distance = higher fitness)
        route.fitness = 1.0 / route.distance;
    }
}

// Tournament selection to pick a parent
Route tournamentSelection(const vector<Route>& population, size_t tournamentSize) {
    uniform_int_distribution<size_t> dis(0, population.size() - 1);
    Route best = population[dis(gen)]; // Random starting candidate
    
    for (size_t i = 1; i < tournamentSize; ++i) {
        size_t randomIndex = dis(gen);
        if (population[randomIndex].fitness > best.fitness) {
            best = population[randomIndex];
        }
    }
    
    return best;
}

// Order crossover (OX) operator
vector<int> orderCrossover(const vector<int>& parent1, const vector<int>& parent2) {
    size_t size = parent1.size();
    vector<int> child(size, -1);
    
    // Select a random subsequence from parent1
    uniform_int_distribution<size_t> dis(0, size - 1);
    size_t start = dis(gen);
    size_t end = dis(gen);
    
    if (start > end) {
        swap(start, end);
    }
    
    // Copy the selected segment from parent1 to child
    for (size_t i = start; i <= end; ++i) {
        child[i] = parent1[i];
    }
    
    // Fill the remaining positions with cities from parent2 in the order they appear
    size_t j = (end + 1) % size;
    for (size_t i = 0; i < size; ++i) {
        // Check if city from parent2 is already in child
        int city = parent2[(end + 1 + i) % size];
        if (find(child.begin(), child.end(), city) == child.end()) {
            child[j] = city;
            j = (j + 1) % size;
            while (j >= start && j <= end) {
                j = (j + 1) % size;
            }
        }
    }
    
    return child;
}

// Mutation operator (swap mutation)
void mutate(vector<int>& path, double mutationRate) {
    uniform_real_distribution<> dis(0.0, 1.0);
    
    // Apply mutation with probability mutationRate
    if (dis(gen) < mutationRate) {
        uniform_int_distribution<size_t> indexDis(0, path.size() - 1);
        size_t i = indexDis(gen);
        size_t j = indexDis(gen);
        
        // Ensure distinct indices
        while (i == j) {
            j = indexDis(gen);
        }
        
        // Swap two cities
        swap(path[i], path[j]);
    }
}

// Create the next generation using selection, crossover, and mutation
vector<Route> createNextGeneration(const vector<Route>& currentGen, const vector<City>& cities, const GAParams& params) {
    vector<Route> nextGen;
    
    // Elitism: Copy best individuals directly to next generation
    vector<Route> sortedCurrentGen = currentGen;
    sort(sortedCurrentGen.begin(), sortedCurrentGen.end(), [](const Route& a, const Route& b) { return a.fitness > b.fitness; });
    
    for (size_t i = 0; i < params.eliteCount && i < sortedCurrentGen.size(); ++i) {
        nextGen.push_back(sortedCurrentGen[i]);
    }
    
    // Create rest of the population
    while (nextGen.size() < params.populationSize) {
        // Select parents
        Route parent1 = tournamentSelection(currentGen, params.tournamentSize);
        Route parent2 = tournamentSelection(currentGen, params.tournamentSize);
        
        // Create child
        Route child;
        
        // Apply crossover with probability crossoverRate
        uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(gen) < params.crossoverRate) {
            child.path = orderCrossover(parent1.path, parent2.path);
        } else {
            // No crossover, just copy one parent
            child.path = parent1.path;
        }
        
        // Apply mutation
        mutate(child.path, params.mutationRate);
        
        // Add to next generation
        nextGen.push_back(child);
    }
    
    // Evaluate fitness of new generation
    evaluatePopulation(nextGen, cities);
    
    return nextGen;
}

// Find the best route in the population
Route findBestRoute(const vector<Route>& population) {
    return *max_element(population.begin(), population.end(), [](const Route& a, const Route& b) { return a.fitness < b.fitness; });
}

// Print route details
void printRoute(const Route& route, const vector<City>& cities) {
    cout << "Route: ";
    for (int cityId : route.path) {
        cout << cityId << " -> ";
    }
    cout << route.path.front() << " (return to start)" << endl;
    
    cout << "Distance: " << fixed << setprecision(2) << route.distance << endl;
    
    cout << "City coordinates:" << endl;
    for (int cityId : route.path) {
        cout << "City " << cityId << ": (" 
             << fixed << setprecision(2) << cities[cityId].x << ", " 
             << cities[cityId].y << ")" << endl;
    }
}

// Visualize the route on a 2D grid
void visualizeRoute(const Route& route, const vector<City>& cities) {
    cout << "\nRoute Visualization (X-Y Coordinate System):" << endl;
    
    // Find min and max coordinates to determine grid boundaries
    double minX = numeric_limits<double>::max();
    double maxX = numeric_limits<double>::min();
    double minY = numeric_limits<double>::max();
    double maxY = numeric_limits<double>::min();
    
    for (const auto& city : cities) {
        minX = min(minX, city.x);
        maxX = max(maxX, city.x);
        minY = min(minY, city.y);
        maxY = max(maxY, city.y);
    }
    
    // Add a small margin
    minX -= 2.0;
    maxX += 2.0;
    minY -= 2.0;
    maxY += 2.0;
    
    // Grid dimensions
    const int gridWidth = 60;
    const int gridHeight = 30;
    
    // Create the grid
    vector<vector<char>> grid(gridHeight, vector<char>(gridWidth, ' '));
    
    // Scale factors to map coordinates to grid
    double xScale = (gridWidth - 1) / (maxX - minX);
    double yScale = (gridHeight - 1) / (maxY - minY);
    
    // Map each city to a grid position
    vector<pair<int, int>> gridPositions;
    for (const auto& city : cities) {
        int gridX = static_cast<int>((city.x - minX) * xScale);
        int gridY = static_cast<int>((maxY - city.y) * yScale); // Invert Y for display
        
        gridX = max(0, min(gridWidth - 1, gridX));
        gridY = max(0, min(gridHeight - 1, gridY));
        
        gridPositions.push_back({gridX, gridY});
        
        // Mark the city on the grid
        grid[gridY][gridX] = 'O';
    }
    
    // Draw the route on the grid
    for (size_t i = 0; i < route.path.size(); ++i) {
        int cityA = route.path[i];
        int cityB = route.path[(i + 1) % route.path.size()];
        
        int x1 = gridPositions[cityA].first;
        int y1 = gridPositions[cityA].second;
        int x2 = gridPositions[cityB].first;
        int y2 = gridPositions[cityB].second;
        
        // Draw a simplified line between cities
        drawLine(grid, x1, y1, x2, y2);
    }
    
    // Add city numbers to the grid
    for (size_t i = 0; i < cities.size(); ++i) {
        int x = gridPositions[i].first;
        int y = gridPositions[i].second;
        
        // Convert city ID to digits
        string cityId = to_string(i);
        if (cityId.length() == 1) {
            grid[y][x] = cityId[0];
        } else {
            // For multi-digit city IDs, just mark with '#'
            grid[y][x] = '#';
        }
    }
    
    // Print the grid
    // Top border
    cout << '+';
    for (int i = 0; i < gridWidth; ++i) cout << '-';
    cout << '+' << endl;
    
    // Grid content
    for (int y = 0; y < gridHeight; ++y) {
        cout << '|';
        for (int x = 0; x < gridWidth; ++x) {
            cout << grid[y][x];
        }
        cout << '|' << endl;
    }
    
    // Bottom border
    cout << '+';
    for (int i = 0; i < gridWidth; ++i) cout << '-';
    cout << '+' << endl;
    
    // Legend
    cout << "Legend: [0-9] - Cities, '*' - Route paths" << endl;
}

// Helper function to draw a line between two points
void drawLine(vector<vector<char>>& grid, int x1, int y1, int x2, int y2) {
    // Bresenham's line algorithm (simplified)
    const bool steep = (abs(y2 - y1) > abs(x2 - x1));
    
    if (steep) {
        swap(x1, y1);
        swap(x2, y2);
    }
    
    if (x1 > x2) {
        swap(x1, x2);
        swap(y1, y2);
    }
    
    const int dx = x2 - x1;
    const int dy = abs(y2 - y1);
    const int ystep = (y1 < y2) ? 1 : -1;
    
    int error = dx / 2;
    int y = y1;
    
    for (int x = x1; x <= x2; ++x) {
        if (steep) {
            if (grid[x][y] != '0' && grid[x][y] != '1' && grid[x][y] != '2' &&
                grid[x][y] != '3' && grid[x][y] != '4' && grid[x][y] != '5' &&
                grid[x][y] != '6' && grid[x][y] != '7' && grid[x][y] != '8' &&
                grid[x][y] != '9' && grid[x][y] != '#') {
                grid[x][y] = '*';
            }
        } else {
            if (grid[y][x] != '0' && grid[y][x] != '1' && grid[y][x] != '2' &&
                grid[y][x] != '3' && grid[y][x] != '4' && grid[y][x] != '5' &&
                grid[y][x] != '6' && grid[y][x] != '7' && grid[y][x] != '8' &&
                grid[y][x] != '9' && grid[y][x] != '#') {
                grid[y][x] = '*';
            }
        }
        
        error -= dy;
        if (error < 0) {
            y += ystep;
            error += dx;
        }
    }
}

// Main function to run the genetic algorithm
void runGeneticAlgorithm() {
    cout << "Running Genetic Algorithm for TSP..." << endl;
    
    // Set parameters
    GAParams params;
    params.populationSize = 100;
    params.generations = 500;
    params.mutationRate = 0.02;
    params.crossoverRate = 0.8;
    params.tournamentSize = 5;
    params.eliteCount = 2;
    
    // Generate cities
    size_t numCities = 15;
    vector<City> cities = generateCities(numCities);
    
    cout << "Created " << numCities << " random cities." << endl;
    
    // Initialize population
    vector<Route> population = initializePopulation(cities, params.populationSize);
    evaluatePopulation(population, cities);
    
    Route bestRoute = findBestRoute(population);
    cout << "Initial best distance: " << fixed << setprecision(2) << bestRoute.distance << endl;
    
    // Evolution loop
    auto startTime = chrono::high_resolution_clock::now();
    
    for (size_t generation = 0; generation < params.generations; ++generation) {
        // Create new generation
        population = createNextGeneration(population, cities, params);
        
        // Find the best route in this generation
        Route generationBest = findBestRoute(population);
        
        // Update overall best if needed
        if (generationBest.fitness > bestRoute.fitness) {
            bestRoute = generationBest;
        }
        
        // Print progress every 50 generations
        if (generation % 50 == 0 || generation == params.generations - 1) {
            cout << "Generation " << generation << ": Best distance = "<< fixed << setprecision(2) << bestRoute.distance << endl;
        }
    }
    
    auto endTime = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    
    cout << "\nEvolution completed in " << duration << " ms." << endl;
    cout << "\nFinal best route:" << endl;
    printRoute(bestRoute, cities);
    
    // Visualize the best route
    visualizeRoute(bestRoute, cities);

    // Compare with brute force approach
    compareWithBruteForce(numCities);
}

// Function to calculate and compare the theoretical time needed for brute force approach
void compareWithBruteForce(size_t numCities) {
    cout << "\n-----------------------------------------------------" << endl;
    cout << "PERFORMANCE COMPARISON: GENETIC ALGORITHM vs BRUTE FORCE" << endl;
    cout << "-----------------------------------------------------" << endl;
    
    // Calculate factorial approximation for large numbers
    auto factorial = [](size_t n) -> double {
        if (n <= 20) {
            // Exact calculation for smaller numbers
            unsigned long long result = 1;
            for (size_t i = 2; i <= n; ++i) {
                result *= i;
            }
            return static_cast<double>(result);
        } else {
            // Stirling's approximation for larger factorials
            // n! ≈ sqrt(2πn) * (n/e)^n
            constexpr double e = 2.71828182845904523536;
            constexpr double pi = 3.14159265358979323846;
            
            double logFactorial = 0.5 * log(2 * pi * n) + n * (log(n) - 1);
            return exp(logFactorial);
        }
    };
    
    // Calculate number of possible routes: (n-1)!/2
    // Division by 2 because routes can be traversed in either direction
    double numRoutes = factorial(numCities - 1) / 2;
    
    // Estimate time per route evaluation (in nanoseconds)
    // This is a reasonable estimate based on modern CPU performance
    constexpr double nanosPerRoute = 100.0; // 100 nanoseconds per route evaluation
    
    // Total estimated time in various units
    double totalNanoseconds = numRoutes * nanosPerRoute;
    double totalSeconds = totalNanoseconds / 1e9;
    double totalMinutes = totalSeconds / 60;
    double totalHours = totalMinutes / 60;
    double totalDays = totalHours / 24;
    double totalYears = totalDays / 365.25;
    
    // Print comparison results
    cout << "Number of cities: " << numCities << endl;
    cout << "Number of possible routes: " << scientific << setprecision(4) << numRoutes << fixed << endl;
    
    cout << "\nEstimated time for brute force solution:" << endl;
    
    if (totalSeconds < 1) {
        cout << "  " << setprecision(4) << totalNanoseconds << " nanoseconds" << endl;
    } else if (totalMinutes < 1) {
        cout << "  " << setprecision(4) << totalSeconds << " seconds" << endl;
    } else if (totalHours < 1) {
        cout << "  " << setprecision(4) << totalMinutes << " minutes" << endl;
    } else if (totalDays < 1) {
        cout << "  " << setprecision(4) << totalHours << " hours" << endl;
    } else if (totalYears < 1) {
        cout << "  " << setprecision(4) << totalDays << " days" << endl;
    } else {
        cout << "  " << setprecision(4) << totalYears << " years" << endl;
    }
    
    // Compare with genetic algorithm
    size_t populationSize = 100;
    size_t generations = 500;
    double gaEvaluations = populationSize * generations;
    double gaTimeNanoseconds = gaEvaluations * nanosPerRoute;
    double gaTimeSeconds = gaTimeNanoseconds / 1e9;
    
    cout << "\nGenetic Algorithm approach:" << endl;
    cout << "  Population size: " << populationSize << endl;
    cout << "  Generations: " << generations << endl;
    cout << "  Total evaluations: " << gaEvaluations << endl;
    cout << "  Estimated time: " << setprecision(4) << gaTimeSeconds << " seconds" << endl;
    
    // Calculate speedup
    double speedup = totalSeconds / gaTimeSeconds;
    cout << "\nSpeedup factor: " << scientific << setprecision(4) << speedup << "x faster" << fixed << endl;
    
    // Print human-readable explanation
    cout << "\nEXPLANATION:" << endl;
    if (numCities <= 10) {
        cout << "For " << numCities << " cities, the brute force approach is feasible." << endl;
    } else if (numCities <= 15) {
        cout << "For " << numCities << " cities, the brute force approach is challenging but possible on modern hardware." << endl;
    } else {
        cout << "For " << numCities << " cities, the brute force approach is practically impossible." << endl;
    }
    
    cout << "The genetic algorithm provides a near-optimal solution in a fraction of the time." << endl;
    cout << "With " << numCities << " cities, it is approximately " << setprecision(0) << speedup;
    cout << " times faster than checking all possible routes." << endl;
    cout << "-----------------------------------------------------\n" << endl;
}