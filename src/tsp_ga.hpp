#pragma once
#include <vector>
#include <cstddef>
using namespace std;

// Define structures for our GA
struct City {
    int id;
    double x;
    double y;
};

struct Route {
    vector<int> path;
    double fitness;
    double distance;
};

// GA Parameters
struct GAParams {
    size_t populationSize;
    size_t generations;
    double mutationRate;
    double crossoverRate;
    size_t tournamentSize;
    size_t eliteCount;
};

// Main function to run the GA
void runGeneticAlgorithm();

// TSP GA functions
vector<City> generateCities(size_t numCities);
vector<Route> initializePopulation(const vector<City>& cities, size_t populationSize);
double calculateDistance(const vector<City>& cities, const vector<int>& path);
void evaluatePopulation(vector<Route>& population, const vector<City>& cities);
Route tournamentSelection(const vector<Route>& population, size_t tournamentSize);
vector<int> orderCrossover(const vector<int>& parent1, const vector<int>& parent2);
void mutate(vector<int>& path, double mutationRate);
vector<Route> createNextGeneration(const vector<Route>& currentGen, const vector<City>& cities, const GAParams& params);
Route findBestRoute(const vector<Route>& population);
void printRoute(const Route& route, const vector<City>& cities);
void visualizeRoute(const Route& route, const vector<City>& cities);
void drawLine(vector<vector<char>>& grid, int x1, int y1, int x2, int y2);
void compareWithBruteForce(size_t numCities);